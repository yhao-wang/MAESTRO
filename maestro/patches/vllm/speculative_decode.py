"""vLLM speculative-decoding patches for OPD rollouts.

The patch provides the SKD top-k baseline, Trigger-Stop, and MAESTRO while
keeping the implementation independent of any training frontend.

Usage: add this directory to PYTHONPATH and set MAESTRO_RUNTIME_VLLM_PATCH=1.
Patches are applied via sitecustomize.py or explicit apply_patches() call.
"""

from __future__ import annotations

import logging
import json
import math
import os
import random
import re
import socket
import struct
from typing import Any

import torch

logger = logging.getLogger(__name__)

PLACEHOLDER_TOKEN_ID = -1

_PATCHED = False
_OPD_STATS: dict[str, torch.Tensor] = {}
_TEACHER_MASK_BY_REQUEST_ID: dict[str, list[bool]] = {}
_TOKEN_EVENTS_BY_REQUEST_ID: dict[str, list[int]] = {}
_POSITION_EVENTS_BY_REQUEST_ID: dict[str, list[int]] = {}
_OVERLAP_EVENTS_BY_REQUEST_ID: dict[str, list[float]] = {}
_TRACE_EVENTS_BY_REQUEST_ID: dict[str, list[dict[str, Any]]] = {}
_TAKEOVER_TOKENS_REMAINING: dict[str, int] = {}
_TAKEOVER_PARAGRAPHS_REMAINING: dict[str, int] = {}
_TAKEOVER_MAX_TAKEOVERS: dict[str, int] = {}
_TAKEOVER_TOKEN_LIMIT: dict[str, int] = {}
_TAKEOVER_PARAGRAPH_LIMIT: dict[str, int] = {}
_COMPLETED_TAKEOVERS: dict[str, int] = {}
# post-takeover tail cap (2026-09-21): once a rollout EXHAUSTS its takeover quota
# and resumes the student (resume_student), the student may emit at most this many
# more tokens before we force an internal stop.  This replaces the old
# generate-then-truncate tail cap in the agent loop: instead of decoding the whole
# response and cutting it afterwards, we stop generation at last_teacher_end+tail_max
# so the wasted tail is never decoded.  One int per in-flight rollout; popped when
# the cap fires or the request otherwise stops.  User口径 2026-09-21: 纯硬停, 不做 \n\n 对齐.
_POST_TAKEOVER_TAIL_REMAINING: dict[str, int] = {}
# per-rollout count of low-PDS "bonus" takeovers granted after quota exhaustion
_LOW_PDS_BONUS_USED: dict[str, int] = {}
_PDS_WINDOW_VALUES: dict[str, list[float]] = {}
_PDS_PARAGRAPH_VALUES: dict[str, list[float]] = {}
_PDS_BAD_WINDOW_STREAK: dict[str, int] = {}
_PDS_FALLBACK_STREAK: dict[str, int] = {}
_PDS_LAST_ROLLING_MEAN: dict[str, float] = {}
_TAKEOVER_RECOVERY_VALUES: dict[str, list[float]] = {}
_TAKEOVER_RECOVERY_SEEN: dict[str, int] = {}
_TAKEOVER_RECOVERY_GOOD_STREAK: dict[str, int] = {}
_TAKEOVER_RECOVERY_READY: dict[str, bool] = {}
_TAKEOVER_COOLDOWN_REMAINING: dict[str, int] = {}
_PARAGRAPH_BOUNDARY_IDS_CACHE: dict[str, torch.Tensor] = {}
# CPU-side (device-free) mirror of the paragraph-boundary token ids, populated the
# first time _get_paragraph_boundary_token_ids scans the vocab.  The \n\n-watchdog
# early stop reads this to measure "tokens since the last \n\n" without a device.
_PARAGRAPH_BOUNDARY_ID_SET: frozenset[int] | None = None
_INTERNAL_STOP_ID_CACHE: dict[str, int | None] = {}
# ---- repetition early stop (generation-side) --------------------------------
#
# Why this exists: the agent loop can only rewind a response *after* it has been
# generated, and a confident loop is exactly the case the PDS trigger cannot see
# (student and teacher agree, so PDS stays ~1 and never triggers).  Without a
# generation-side stop the student therefore decodes its entire 16384-token
# budget on every leg, and the agent loop then throws nearly all of it away
# (A1/0.6B measured removed_suffix_mean = 12878 of ~19k decoded tokens, i.e.
# ~70% of the decode was paid for and dropped -> 44 min/step).
#
# The fix is *not* to chunk generation from the agent loop: the relay state
# machine (takeover ledger, cooldown, paragraph budget, absolute-position
# teacher masks) is keyed by request id, so splitting one response into several
# requests would silently reset it and de-align the masks.  Instead the sampler
# -- which already sees every emitted token of the live sequence and already has
# a working "end this request now" channel (`internal_stop_output`) -- watches
# for a stable repetition and ends the request at the loop onset.  The agent loop
# then rewinds a few dozen tokens instead of thirteen thousand.
_REPEAT_EARLY_STATE: dict[str, dict[str, Any]] = {}
_REPEAT_EARLY_LOG_LIMIT = 8
_REPEAT_EARLY_LOGGED = 0
_VLLM_PORT_COUNTER = 0
_DYNAMIC_BUDGET_LOGGED = False

_RANDOM_SUFFIX_RE = re.compile(r"-[0-9a-f]{8}$")
_MAESTRO_STEP_RE = re.compile(r"^relaystep(-?\d+)-")


def pop_rollout_trace(request_id: str) -> dict[str, Any] | None:
    """Pop per-token trace for an in-process eval request."""
    if request_id is None:
        return None
    rid = _strip_vllm_suffix(str(request_id))
    mask = _TEACHER_MASK_BY_REQUEST_ID.pop(rid, None)
    tokens = _TOKEN_EVENTS_BY_REQUEST_ID.pop(rid, None)
    positions = _POSITION_EVENTS_BY_REQUEST_ID.pop(rid, None)
    overlaps = _OVERLAP_EVENTS_BY_REQUEST_ID.pop(rid, None)
    events = _TRACE_EVENTS_BY_REQUEST_ID.pop(rid, None)
    if mask is None and tokens is None and positions is None and overlaps is None and events is None:
        return None
    return {
        "request_id": rid,
        "teacher_mask": mask,
        "tokens": tokens,
        "positions": positions,
        "topk_overlaps": overlaps,
        "events": events or [],
    }


_MASK_IPC_SOCK: socket.socket | None = None


def _get_mask_ipc_sock() -> socket.socket | None:
    global _MASK_IPC_SOCK
    sock_path = os.environ.get("MAESTRO_RUNTIME_MASK_IPC_SOCKET")
    if not sock_path:
        return None
    if _MASK_IPC_SOCK is None:
        _MASK_IPC_SOCK = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        _MASK_IPC_SOCK.connect(sock_path)
    return _MASK_IPC_SOCK


def _send_mask_chunk_ipc(
    request_id: str,
    mask_chunk: list[bool],
    token_chunk: list[int],
    position_chunk: list[int] | None = None,
    action_token_chunk: list[int] | None = None,
    teacher_action_logp_chunk: list[float] | None = None,
    topk_overlap_chunk: list[float] | None = None,
    pds_chunk: list[float] | None = None,
    teacher_mass_coverage_chunk: list[float] | None = None,
    local_bhattacharyya_chunk: list[float] | None = None,
    student_p_teacher_top1_chunk: list[float] | None = None,
    teacher_student_logp_gap_top1_chunk: list[float] | None = None,
) -> None:
    """Send emitted-token/mask events to the server process via Unix datagram."""
    global _MASK_IPC_SOCK
    if not request_id or not mask_chunk:
        return
    pds_metric_chunks = [
        pds_chunk,
        teacher_mass_coverage_chunk,
        local_bhattacharyya_chunk,
        student_p_teacher_top1_chunk,
        teacher_student_logp_gap_top1_chunk,
    ]
    has_pds_metrics = any(chunk is not None for chunk in pds_metric_chunks)
    malformed = (
        len(mask_chunk) != len(token_chunk) or len(mask_chunk) > 255
        or position_chunk is None
        or len(position_chunk) != len(mask_chunk)
        or (action_token_chunk is not None and len(action_token_chunk) != len(mask_chunk))
        or (teacher_action_logp_chunk is not None and len(teacher_action_logp_chunk) != len(mask_chunk))
        or ((action_token_chunk is None) != (teacher_action_logp_chunk is None))
        or (topk_overlap_chunk is not None and len(topk_overlap_chunk) != len(mask_chunk))
        or (has_pds_metrics and topk_overlap_chunk is None)
    )
    if has_pds_metrics:
        malformed = malformed or any(chunk is None or len(chunk) != len(mask_chunk) for chunk in pds_metric_chunks)
    if malformed:
        raise RuntimeError(
            "malformed OPD mask IPC chunk: "
            f"rid={request_id} masks={len(mask_chunk)} tokens={len(token_chunk)} "
            f"positions={None if position_chunk is None else len(position_chunk)} "
            f"actions={None if action_token_chunk is None else len(action_token_chunk)} "
            f"action_logps={None if teacher_action_logp_chunk is None else len(teacher_action_logp_chunk)} "
            f"topk_overlap={None if topk_overlap_chunk is None else len(topk_overlap_chunk)} "
            f"pds={None if pds_chunk is None else len(pds_chunk)}"
        )
    sock = _get_mask_ipc_sock()
    if sock is None:
        return
    n = len(mask_chunk)
    mask_bytes = bytes(1 if x else 0 for x in mask_chunk)
    token_bytes = struct.pack(f"!{n}i", *[int(x) for x in token_chunk])
    position_bytes = struct.pack(f"!{n}i", *[int(x) for x in position_chunk])
    overlap_bytes = (
        struct.pack(f"!{n}f", *[float(x) for x in topk_overlap_chunk])
        if topk_overlap_chunk is not None
        else b""
    )
    if has_pds_metrics:
        pds_bytes = b"".join(
            struct.pack(f"!{n}f", *[float(x) for x in chunk])
            for chunk in [
                topk_overlap_chunk,
                pds_chunk,
                teacher_mass_coverage_chunk,
                local_bhattacharyya_chunk,
                student_p_teacher_top1_chunk,
                teacher_student_logp_gap_top1_chunk,
            ]
            if chunk is not None
        )
        if action_token_chunk is not None and teacher_action_logp_chunk is not None:
            action_bytes = struct.pack(f"!{n}i", *[int(x) for x in action_token_chunk])
            action_logp_bytes = struct.pack(f"!{n}f", *[float(x) for x in teacher_action_logp_chunk])
            body = (
                b"OPD7" + bytes([n]) + mask_bytes + token_bytes + position_bytes
                + action_bytes + action_logp_bytes + pds_bytes
            )
        else:
            body = b"OPD6" + bytes([n]) + mask_bytes + token_bytes + position_bytes + pds_bytes
    elif action_token_chunk is not None and teacher_action_logp_chunk is not None and topk_overlap_chunk is not None:
        action_bytes = struct.pack(f"!{n}i", *[int(x) for x in action_token_chunk])
        action_logp_bytes = struct.pack(f"!{n}f", *[float(x) for x in teacher_action_logp_chunk])
        body = (
            b"OPD5" + bytes([n]) + mask_bytes + token_bytes + position_bytes
            + action_bytes + action_logp_bytes + overlap_bytes
        )
    elif action_token_chunk is not None and teacher_action_logp_chunk is not None:
        action_bytes = struct.pack(f"!{n}i", *[int(x) for x in action_token_chunk])
        action_logp_bytes = struct.pack(f"!{n}f", *[float(x) for x in teacher_action_logp_chunk])
        body = b"OPD3" + bytes([n]) + mask_bytes + token_bytes + position_bytes + action_bytes + action_logp_bytes
    elif topk_overlap_chunk is not None:
        body = b"OPD4" + bytes([n]) + mask_bytes + token_bytes + position_bytes + overlap_bytes
    else:
        # OPD2 carries absolute response positions. This lets the server place
        # each mask bit directly instead of guessing an offset from token text.
        body = b"OPD2" + bytes([n]) + mask_bytes + token_bytes + position_bytes
    payload = request_id.encode("utf-8") + b"\0" + body
    try:
        sock.send(payload)
    except Exception:
        try:
            _MASK_IPC_SOCK.close()
        except Exception:
            pass
        _MASK_IPC_SOCK = None


def _send_stop_event_ipc(request_id: str, event: dict[str, Any]) -> None:
    """Send an internal trigger-stop event to the rollout server."""
    global _MASK_IPC_SOCK
    if not request_id or not event:
        return
    sock = _get_mask_ipc_sock()
    if sock is None:
        return
    try:
        body = b"OPS1" + json.dumps(event, ensure_ascii=False).encode("utf-8")
        payload = request_id.encode("utf-8") + b"\0" + body
        sock.send(payload)
    except Exception:
        try:
            _MASK_IPC_SOCK.close()
        except Exception:
            pass
        _MASK_IPC_SOCK = None


def _strip_vllm_suffix(internal_rid: str) -> str:
    return _RANDOM_SUFFIX_RE.sub("", internal_rid)


# Rule (stated by the PI on 2026-09-17, implemented here): a repetition always
# earns a teacher takeover, whether or not the paragraph/PDS quota is spent; it
# is bounded only by the *single* takeover budget (paragraphs per takeover,
# token cap, adaptive exit, cooldown).  The paragraph channel itself must stay
# dead once its quota is gone.
#
# The agent-loop repeat recovery continues the student under a brand new vLLM
# request every round, and the whole takeover ledger is keyed by the request id,
# so without this the paragraph channel would silently get a fresh quota on
# every continuation leg and could fire far more than MAESTRO_MAX_TAKEOVERS
# times per rollout.  The agent loop encodes the teacher legs the rollout has
# already spent in the continuation request id
# ("relayrepeatcontinue<legs>-<uuid>"); seeding the ledger from it keeps the
# paragraph channel honest across legs.  The repetition early stop is untouched
# by this: it never consults the quota.
_REPEAT_CONTINUE_SPENT_RE = re.compile(r"^relayrepeatcontinue(\d+)-")


def _seed_spent_takeovers(ext: str | None) -> int | None:
    """Pre-load the takeover ledger for a repeat-recovery continuation leg."""
    if not ext:
        return None
    match = _REPEAT_CONTINUE_SPENT_RE.match(ext)
    if match is None:
        return None
    if ext in _COMPLETED_TAKEOVERS:
        return None
    spent = int(match.group(1))
    if spent <= 0:
        return None
    _COMPLETED_TAKEOVERS[ext] = spent
    _stats_add("maestro_seeded_spent_takeovers", 1)
    return spent


def _flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "y", "on", "sample"}


def _float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _exit_min_takeover_tokens(request_id: str | None, default: int) -> int:
    """Resolve an optional step-aware linear schedule for adaptive exit."""
    mode = os.environ.get("MAESTRO_EXIT_MIN_TAKEOVER_SCHEDULE", "").strip().lower()
    if mode not in {"linear", "linear_step"} or not request_id:
        return max(default, 0)
    match = _MAESTRO_STEP_RE.match(_strip_vllm_suffix(str(request_id)))
    if match is None:
        return max(default, 0)
    step = int(match.group(1))
    start_step = _int("MAESTRO_EXIT_MIN_DECAY_START_STEP", 20)
    end_step = _int("MAESTRO_EXIT_MIN_DECAY_END_STEP", 50)
    start_value = max(_int("MAESTRO_EXIT_MIN_DECAY_START_VALUE", default), 0)
    end_value = max(_int("MAESTRO_EXIT_MIN_DECAY_END_VALUE", default), 0)
    if end_step <= start_step:
        return end_value if step > start_step else start_value
    if step <= start_step:
        return start_value
    if step >= end_step:
        return end_value
    progress = (step - start_step) / (end_step - start_step)
    return max(int(round(start_value + progress * (end_value - start_value))), 0)


def _dynamic_takeover_budget(
    trigger_pds: float | None,
    default_paragraphs: int,
    default_tokens: int,
    default_max_takeovers: int,
) -> tuple[int, int, int]:
    """Increase takeover paragraphs linearly as PDS difficulty rises."""
    if not _flag("MAESTRO_LINEAR_PDS_BUDGET", False):
        return default_paragraphs, default_tokens, default_max_takeovers
    trigger_threshold = _float("MAESTRO_PDS_TRIGGER_THRESHOLD", 0.14)
    step = max(_float("MAESTRO_PDS_DIFFICULTY_STEP", 0.02), 1e-6)
    base_paragraphs = max(_int("MAESTRO_PDS_BASE_PARAGRAPHS", default_paragraphs), 1)
    max_paragraphs = max(_int("MAESTRO_PDS_MAX_PARAGRAPHS", base_paragraphs), base_paragraphs)
    paragraphs = base_paragraphs
    if trigger_pds is not None and math.isfinite(trigger_pds) and trigger_pds >= trigger_threshold:
        extra = max(int(math.floor((trigger_pds - trigger_threshold + 1e-9) / step)), 0)
        paragraphs = min(base_paragraphs + extra, max_paragraphs)
    _log_dynamic_budget_priority(
        default_paragraphs,
        default_tokens,
        default_max_takeovers,
        (paragraphs, default_tokens, default_max_takeovers),
        (base_paragraphs, default_tokens, default_max_takeovers),
    )
    return paragraphs, default_tokens, default_max_takeovers


def _log_dynamic_budget_priority(
    global_paragraphs: int,
    global_tokens: int,
    global_max_takeovers: int,
    hard: tuple[int, int, int],
    normal: tuple[int, int, int],
) -> None:
    """Make the shadowing explicit instead of a silent no-op.

    With MAESTRO_LINEAR_PDS_BUDGET=1 the per-request caps win, so the global
    MAESTRO_MAX_TAKEOVERS / PARAGRAPHS_PER_TAKEOVER / MAX_TAKEOVER_TOKENS are
    inert for every triggered request.  Teams have lost days to this, so it is
    logged loudly once per process (and the effective values are dumped into
    run_config.md by the launcher).
    """
    global _DYNAMIC_BUDGET_LOGGED
    if _DYNAMIC_BUDGET_LOGGED:
        return
    _DYNAMIC_BUDGET_LOGGED = True
    shadowed = []
    if global_max_takeovers not in (hard[2], normal[2]):
        shadowed.append(f"MAESTRO_MAX_TAKEOVERS={global_max_takeovers}")
    if global_paragraphs not in (hard[0], normal[0]):
        shadowed.append(f"MAESTRO_PARAGRAPHS_PER_TAKEOVER={global_paragraphs}")
    if global_tokens not in (hard[1], normal[1]):
        shadowed.append(f"MAESTRO_MAX_TAKEOVER_TOKENS={global_tokens}")
    message = (
        "[opd-budget] dynamic budget ON: per-request caps win -> hard(paragraphs=%s tokens=%s max_takeovers=%s) "
        "normal(paragraphs=%s tokens=%s max_takeovers=%s)"
    )
    if shadowed:
        logger.warning(
            message + " | SHADOWED (ignored) globals: %s",
            hard[0], hard[1], hard[2], normal[0], normal[1], normal[2], ", ".join(shadowed),
        )
    else:
        logger.warning(
            message + " | globals match one of the tiers, no shadowing detected",
            hard[0], hard[1], hard[2], normal[0], normal[1], normal[2],
        )


def _pds_segment_score(values: list[float], mode: str) -> float:
    """Aggregate a paragraph PDS segment without changing the rolling default."""
    finite = [float(v) for v in values if math.isfinite(float(v))]
    if not finite:
        return float("nan")
    if mode not in {"paragraph_weighted", "weighted", "hybrid"}:
        return float(sum(finite) / len(finite))
    alpha = min(max(_float("MAESTRO_PDS_PREFIX_ALPHA", 0.5), 0.0), 2.0)
    tau = max(_float("MAESTRO_PDS_PREFIX_TAU", 12.0), 1.0)
    weights = [1.0 + alpha * math.exp(-i / tau) for i in range(len(finite))]
    return float(sum(w * v for w, v in zip(weights, finite)) / sum(weights))


def _stats_enabled() -> bool:
    return _flag("MAESTRO_RUNTIME_SOURCE_METRICS", True)


def _trace_enabled() -> bool:
    return _flag("MAESTRO_RUNTIME_TRACE", False)


def _trace_mode() -> str:
    mode = os.environ.get("MAESTRO_RUNTIME_TRACE_MODE", "full").strip().lower()
    if mode in {"mask", "masks", "position", "positions", "minimal", "light"}:
        return "mask"
    return "full"


def _trace_file_writer_enabled() -> bool:
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return int(torch.distributed.get_rank()) == 0
    except Exception:
        return True
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank not in (None, ""):
        try:
            return int(local_rank) == 0
        except ValueError:
            return False
    return True


def _stats_add(key: str, value: torch.Tensor | int | float, device: torch.device | None = None) -> None:
    if not _stats_enabled():
        return
    with torch.no_grad():
        if isinstance(value, torch.Tensor):
            val = value.detach().float().sum()
        else:
            val = torch.tensor(float(value), dtype=torch.float32, device=device)
        if key in _OPD_STATS:
            _OPD_STATS[key] = _OPD_STATS[key].to(device=val.device) + val
        else:
            _OPD_STATS[key] = val


def pop_opd_stats() -> dict[str, float]:
    stats = {key: float(value.detach().float().cpu().item()) for key, value in _OPD_STATS.items()}
    _OPD_STATS.clear()
    return stats


def _get_internal_stop_token_id(vocab_size: int) -> int | None:
    """Resolve the internal stop token used only to make vLLM finish a request."""
    env_val = os.environ.get("MAESTRO_RUNTIME_INTERNAL_STOP_TOKEN_ID")
    tok_path = os.environ.get("MAESTRO_RUNTIME_TOKENIZER_PATH") or os.environ.get(
        "MAESTRO_RUNTIME_TARGET_MODEL"
    ) or os.environ.get("STUDENT_MODEL")
    cache_key = f"{env_val or ''}:{tok_path or ''}:{vocab_size}"
    if cache_key in _INTERNAL_STOP_ID_CACHE:
        return _INTERNAL_STOP_ID_CACHE[cache_key]

    stop_id: int | None = None
    if env_val not in (None, ""):
        try:
            stop_id = int(env_val)
        except ValueError:
            logger.warning("[opd-rollout] invalid MAESTRO_RUNTIME_INTERNAL_STOP_TOKEN_ID=%r", env_val)
            stop_id = None
    elif tok_path:
        try:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
            eos = getattr(tok, "eos_token_id", None)
            if isinstance(eos, (list, tuple)):
                eos = eos[0] if eos else None
            if eos is not None:
                stop_id = int(eos)
        except Exception:
            logger.exception("[opd-rollout] failed to resolve internal stop token from tokenizer")

    if stop_id is not None and not (0 <= stop_id < vocab_size):
        logger.warning(
            "[opd-rollout] internal stop token id out of range: stop_id=%s vocab_size=%s",
            stop_id,
            vocab_size,
        )
        stop_id = None
    logger.warning("[opd-rollout] internal stop token id=%s", stop_id)
    _INTERNAL_STOP_ID_CACHE[cache_key] = stop_id
    return stop_id


_REPEAT_EARLY_CFG_CACHE: dict[str, Any] | None = None


def _repeat_early_stop_config() -> dict[str, Any]:
    """Sampler-side repetition early stop (see _REPEAT_EARLY_STATE for why).

    Defaults follow ``MAESTRO_REPEAT_RECOVERY``: an early stop is only useful
    when the agent loop is allowed to rewind, so enabling recovery also enables
    it, and an explicit ``MAESTRO_REPEAT_EARLY_STOP`` overrides either way.
    """
    global _REPEAT_EARLY_CFG_CACHE
    if _REPEAT_EARLY_CFG_CACHE is not None:
        return _REPEAT_EARLY_CFG_CACHE
    cfg = {
        "enabled": _flag("MAESTRO_REPEAT_EARLY_STOP", _flag("MAESTRO_REPEAT_RECOVERY", False)),
        "min_run": max(_int("MAESTRO_REPEAT_EARLY_MIN_RUN", _int("MAESTRO_REPEAT_MIN_RUN", 24)), 4),
        "max_period": max(_int("MAESTRO_REPEAT_EARLY_MAX_PERIOD", 32), 1),
        "min_period_repeats": max(_int("MAESTRO_REPEAT_EARLY_MIN_PERIOD_REPEATS", 3), 2),
        "stride": max(_int("MAESTRO_REPEAT_EARLY_CHECK_STRIDE", 8), 1),
        # Generalised (period-free) repetition: fires when everything after the
        # last *newly seen* n-gram covers >= min_span tokens, i.e. the response has
        # been echoing already-seen material for a long stretch.  This is the only
        # channel that catches sentence-level echo whose period exceeds the tandem
        # bound (probe row 11466: period 283 tokens, tandem sees it at token 16192).
        "general": _flag("MAESTRO_REPEAT_GENERAL", False),
        "general_ngram": max(_int("MAESTRO_REPEAT_GENERAL_NGRAM", 16), 2),
        "general_min_span": max(_int("MAESTRO_REPEAT_GENERAL_MIN_SPAN", 192), 8),
        # \n\n-watchdog: end the request when the tail runs this many tokens with
        # no paragraph break.  Pattern-agnostic (catches single-word / digit
        # runaways the tandem + general rules miss) and, being the dominant tail-
        # length source, it is also the biggest rollout-throughput lever.  The
        # agent-loop rewind (repeat_recovery.find_no_boundary_tail_start) uses the
        # SAME token set and threshold, so a request stopped here is always
        # rewound to the last \n\n rather than handed back untouched.
        "noeos_gap": _flag("MAESTRO_NOEOS_GAP", False),
        "noeos_gap_max": max(_int("MAESTRO_NOEOS_GAP_MAX_TOKENS", 3072), 8),
    }
    _REPEAT_EARLY_CFG_CACHE = cfg
    logger.warning("[opd-rollout] repeat early-stop effective cfg=%r", cfg)
    return cfg


def _repeat_early_stop_reset(request_id: str | None) -> None:
    if request_id is None:
        return
    _REPEAT_EARLY_STATE.pop(request_id, None)


def _repeat_period_match(tail: list[int], max_period: int, min_run: int, min_repeats: int) -> int:
    """Smallest stable tandem period: ``repeats`` copies of a ``k``-token block.

    ``repeats`` is raised until the matched region covers ``min_run`` tokens, so a
    short cycle (``beef $2$``) needs more copies than a long one, but both have to
    cover at least ``min_run`` tokens before the sampler acts.  The two O(1) index
    probes prune almost every candidate before any list slicing happens.
    """
    n = len(tail)
    for k in range(1, max_period + 1):
        repeats = max(min_repeats, -(-min_run // k))
        span = k * repeats
        if n < span:
            continue
        if tail[-1] != tail[-1 - k] or tail[-2] != tail[-2 - k]:
            continue
        block = tail[-k:]
        ok = True
        for r in range(2, repeats + 1):
            if tail[-r * k : -(r - 1) * k] != block:
                ok = False
                break
        if ok:
            return k
    return 0


def _repeat_early_stop_observe(
    request_id: str, tokens: list[int], cfg: dict[str, Any]
) -> dict[str, Any] | None:
    """Feed the tokens emitted by one verify step; report a repetition hit once.

    Two signals, both cheap: a run of one identical token (the dominant A1
    failure mode, ``' \\\\' x 282``) tracked with an O(1) counter, and a tandem
    periodicity check run every ``stride`` tokens.  Detection only has to be
    *early* and unbiased -- the exact onset is still computed by the agent loop
    with the same ``find_repeat_start`` used for the rewind.
    """
    state = _REPEAT_EARLY_STATE.get(request_id)
    if state is None:
        if len(_REPEAT_EARLY_STATE) > 200000:
            # Requests that end normally (EOS / length) are never seen again, and
            # this module has no per-request teardown hook, so bound the table
            # instead of letting it grow without limit across a long run.
            _REPEAT_EARLY_STATE.clear()
        state = {
            "tail": [],
            "run": 0,
            "last": None,
            "seen": 0,
            "gtail": [],
            "gh": 0,
            "gj": -1,
            "glast_novel": -1,
            "gseen": set(),
            "nlgap": 0,
        }
        _REPEAT_EARLY_STATE[request_id] = state
    tail: list[int] = state["tail"]
    run = int(state["run"])
    last = state["last"]
    seen = int(state["seen"])
    min_run = int(cfg["min_run"])
    max_period = int(cfg["max_period"])
    stride = int(cfg["stride"])
    min_repeats = int(cfg["min_period_repeats"])
    keep = max(max_period * 8, min_run * 4)
    # generalised echo
    g_enabled = bool(cfg.get("general"))
    g_n = int(cfg.get("general_ngram", 16))
    g_min_span = int(cfg.get("general_min_span", 192))
    g_tail = state["gtail"]
    g_hash = int(state["gh"])
    g_gram = int(state["gj"])
    g_last_novel = int(state["glast_novel"])
    g_seen = state["gseen"]
    # \n\n-watchdog channel: tokens since the last paragraph-boundary token.
    noeos_enabled = bool(cfg.get("noeos_gap")) and _PARAGRAPH_BOUNDARY_ID_SET is not None
    noeos_gap_max = int(cfg.get("noeos_gap_max", 3072))
    noeos_bset = _PARAGRAPH_BOUNDARY_ID_SET if noeos_enabled else None
    nlgap = int(state.get("nlgap", 0))
    _MASK = (1 << 64) - 1
    _BASE = 0x100000001B3
    _POW = pow(_BASE, g_n, 1 << 64)
    hit: dict[str, Any] | None = None
    for raw_tok in tokens:
        tok = int(raw_tok)
        run = run + 1 if tok == last else 1
        last = tok
        tail.append(tok)
        seen += 1
        if noeos_enabled:
            nlgap = 0 if tok in noeos_bset else nlgap + 1
        if len(tail) > keep:
            del tail[:-keep]
        if hit is not None:
            continue
        if min_run > 1 and run >= min_run:
            hit = {"kind": "run", "period": 1, "matches": int(run), "tail_len": len(tail)}
            continue
        if g_enabled:
            # O(1) rolling window: hash the last g_n tokens; a gram that has never
            # been seen resets the echo counter, so only a sustained echo fires.
            val = (tok & 0xFFFFFFFF) + 1
            g_tail.append(tok)
            if len(g_tail) > g_n:
                left = g_tail.pop(0)
                # Horner: shift first, then subtract the term that left the window
                # (it sits at BASE**g_n after the shift), otherwise the hash is off
                # by one factor of BASE and identical windows never collide.
                g_hash = (g_hash * _BASE + val - ((left & 0xFFFFFFFF) + 1) * _POW) & _MASK
            elif len(g_tail) == g_n:
                g_hash = 0
                for t in g_tail:
                    g_hash = (g_hash * _BASE + ((t & 0xFFFFFFFF) + 1)) & _MASK
            if len(g_tail) == g_n:
                g_gram += 1
                if g_hash in g_seen:
                    pass
                else:
                    g_seen.add(g_hash)
                    g_last_novel = g_gram
                echo_span = seen - (g_last_novel + 1)
                if echo_span >= g_min_span:
                    hit = {
                        "kind": "general",
                        "period": int(echo_span),
                        "matches": int(echo_span),
                        "tail_len": len(tail),
                    }
                    continue
        if noeos_enabled and nlgap >= noeos_gap_max:
            hit = {
                "kind": "noeos_gap",
                "period": int(nlgap),
                "matches": int(nlgap),
                "tail_len": len(tail),
            }
            continue
        if max_period > 1 and seen % stride == 0:
            period = _repeat_period_match(tail, max_period, min_run, min_repeats)
            if period:
                hit = {
                    "kind": "period",
                    "period": int(period),
                    "matches": int(period),
                    "tail_len": len(tail),
                }
    state["tail"] = tail
    state["run"] = run
    state["last"] = last
    state["seen"] = seen
    state["gtail"] = g_tail
    state["gh"] = g_hash
    state["gj"] = g_gram
    state["glast_novel"] = g_last_novel
    state["gseen"] = g_seen
    state["nlgap"] = nlgap
    return hit


def _repeat_early_stop_log(
    request_id: str, last_pos: int, hit: dict[str, Any], in_takeover: bool, stop_slot: int
) -> None:
    global _REPEAT_EARLY_LOGGED
    if _REPEAT_EARLY_LOGGED >= _REPEAT_EARLY_LOG_LIMIT:
        return
    _REPEAT_EARLY_LOGGED += 1
    logger.warning(
        "[opd-rollout] repeat early-stop: rid=%s response_pos=%s kind=%s period=%s matches=%s "
        "tail_len=%s in_takeover=%s stop_slot=%s (further hits counted in stats)",
        request_id,
        last_pos,
        hit.get("kind"),
        hit.get("period"),
        hit.get("matches"),
        hit.get("tail_len"),
        in_takeover,
        stop_slot,
    )


def _apply_repeat_early_stop(
    output: torch.Tensor,
    internal_stop_output: torch.Tensor,
    emitted: torch.Tensor,
    ext_rids: list[str | None],
    in_takeover_per_req: torch.Tensor,
    vocab_size: int,
    device: torch.device,
    cfg: dict[str, Any],
) -> torch.Tensor | None:
    """End every request whose response has started repeating.

    Mutates ``output`` / ``internal_stop_output`` in place and returns a mask of
    the slots that hold a *repetition* stop token (so the stop-event record can
    tag it, while the request itself rides the existing internal-stop channel).
    Split out of the sampler so it can be tested without a GPU.
    """
    if not cfg.get("enabled"):
        return None
    batch_size = int(output.shape[0])
    repeat_stop_output: torch.Tensor | None = None
    emitted_now = emitted & ~internal_stop_output
    stopping_now = internal_stop_output.any(dim=1)
    for bi in range(batch_size):
        ext = ext_rids[bi] if 0 <= bi < len(ext_rids) else None
        if ext is None:
            continue
        if bool(stopping_now[bi]):
            # This request already ends during this step; its history is dead.
            _repeat_early_stop_reset(ext)
            continue
        pos_list = torch.nonzero(emitted_now[bi], as_tuple=False).flatten().detach().cpu().tolist()
        if not pos_list:
            continue
        step_tokens = [int(output[bi, int(p)].item()) for p in pos_list]
        hit = _repeat_early_stop_observe(ext, step_tokens, cfg)
        if hit is None:
            continue
        stop_id = _get_internal_stop_token_id(vocab_size)
        if stop_id is None:
            _stats_add("maestro_repeat_early_stop_missing_internal_stop_id", 1, device=device)
            continue
        last_pos = int(pos_list[-1])
        stop_slot = last_pos + 1 if last_pos + 1 < output.shape[1] else last_pos
        if stop_slot == last_pos:
            # No spare slot in this spec block: sacrifice the final emitted token,
            # exactly like the max-takeover stop path does.
            _stats_add("maestro_repeat_early_stop_no_extra_slot", 1, device=device)
        output[bi, stop_slot] = int(stop_id)
        internal_stop_output[bi, stop_slot] = True
        if repeat_stop_output is None:
            repeat_stop_output = torch.zeros_like(internal_stop_output)
        repeat_stop_output[bi, stop_slot] = True
        in_takeover_now = bool(in_takeover_per_req[bi])
        _stats_add("maestro_repeat_early_stop_events", 1, device=device)
        _stats_add("maestro_repeat_early_stop_pos_sum", float(last_pos), device=device)
        _stats_add(
            "maestro_repeat_early_stop_run" if hit["kind"] == "run" else "maestro_repeat_early_stop_period",
            1,
            device=device,
        )
        if hit["kind"] == "noeos_gap":
            # Dedicated \n\n-watchdog counter (also rides the generic _period tally
            # above so the existing aggregate is unchanged).
            _stats_add("maestro_repeat_early_stop_noeos_gap", 1, device=device)
        if in_takeover_now:
            _stats_add("maestro_repeat_early_stop_in_takeover", 1, device=device)
        # The request ends here, so release every per-request relay structure (the
        # max-takeover stop path pops the same keys).  Without this a long run
        # would leak one entry per finished request, for every dict.
        _clear_pds_state(ext)
        _clear_takeover_recovery_state(ext)
        _TAKEOVER_TOKENS_REMAINING.pop(ext, None)
        _TAKEOVER_PARAGRAPHS_REMAINING.pop(ext, None)
        _TAKEOVER_COOLDOWN_REMAINING.pop(ext, None)
        _COMPLETED_TAKEOVERS.pop(ext, None)
        _LOW_PDS_BONUS_USED.pop(ext, None)
        _POST_TAKEOVER_TAIL_REMAINING.pop(ext, None)
        _repeat_early_stop_reset(ext)
        _repeat_early_stop_log(ext, last_pos, hit, in_takeover_now, stop_slot)
    return repeat_stop_output


def _apply_post_takeover_tail_cap(
    output: torch.Tensor,
    internal_stop_output: torch.Tensor,
    teacher_output: torch.Tensor,
    ext_rids: list[str | None],
    vocab_size: int,
    device: torch.device,
) -> None:
    """End every rollout that has walked ``tail_max`` student tokens past its last
    takeover.  ``_POST_TAKEOVER_TAIL_REMAINING[ext]`` is seeded with tail_max the
    step a rollout exhausts its takeover quota under resume_student; here we count
    the STUDENT tokens it emits afterwards (teacher tokens excluded so the final
    leg's own takeover tokens don't count) and inject an internal stop once the
    budget is spent.  Same no-EOS / prefix-distill terminal shape as the other
    internal-stop paths, so the tail is simply never decoded past the cap.
    Mutates ``output`` / ``internal_stop_output`` in place.
    """
    if not _POST_TAKEOVER_TAIL_REMAINING:
        return
    batch_size = int(output.shape[0])
    # Recompute emitted from the CURRENT output: the caller's `emitted` predates the
    # takeover-termination loop that drops excess teacher tokens to PLACEHOLDER, so a
    # dropped slot (teacher_output just cleared) would otherwise miscount as student.
    emitted_student = (
        output.ge(0)
        & output.ne(PLACEHOLDER_TOKEN_ID)
        & ~internal_stop_output
        & ~teacher_output
    )
    stopping_now = internal_stop_output.any(dim=1)
    for bi in range(batch_size):
        ext = ext_rids[bi] if 0 <= bi < len(ext_rids) else None
        if ext is None or ext not in _POST_TAKEOVER_TAIL_REMAINING:
            continue
        if bool(stopping_now[bi]):
            # Request already ends this step (repeat stop / max-takeover stop): drop
            # the deadline so it can't leak past a finished request.
            _POST_TAKEOVER_TAIL_REMAINING.pop(ext, None)
            continue
        pos_list = torch.nonzero(emitted_student[bi], as_tuple=False).flatten().detach().cpu().tolist()
        if not pos_list:
            continue
        remaining = int(_POST_TAKEOVER_TAIL_REMAINING.get(ext, 0))
        if len(pos_list) < remaining:
            _POST_TAKEOVER_TAIL_REMAINING[ext] = remaining - len(pos_list)
            continue
        # The budget runs out inside this step: keep the first ``remaining`` student
        # tokens and stop right after the last kept one.  vLLM cuts the response at
        # the first internal-stop slot, so any tokens later in this block are dropped.
        stop_id = _get_internal_stop_token_id(vocab_size)
        if stop_id is None:
            _stats_add("maestro_post_takeover_tail_missing_internal_stop_id", 1, device=device)
            _POST_TAKEOVER_TAIL_REMAINING.pop(ext, None)
            continue
        keep_pos = int(pos_list[remaining - 1]) if remaining >= 1 else int(pos_list[0])
        stop_slot = keep_pos + 1 if keep_pos + 1 < output.shape[1] else keep_pos
        if stop_slot == keep_pos:
            _stats_add("maestro_post_takeover_tail_no_extra_slot", 1, device=device)
        output[bi, stop_slot] = int(stop_id)
        internal_stop_output[bi, stop_slot] = True
        _stats_add("maestro_post_takeover_tail_stop_events", 1, device=device)
        # Request ends here: release every per-request relay structure (same set the
        # max-takeover/repeat stop paths pop).
        _POST_TAKEOVER_TAIL_REMAINING.pop(ext, None)
        _clear_pds_state(ext)
        _clear_takeover_recovery_state(ext)
        _TAKEOVER_TOKENS_REMAINING.pop(ext, None)
        _TAKEOVER_PARAGRAPHS_REMAINING.pop(ext, None)
        _TAKEOVER_COOLDOWN_REMAINING.pop(ext, None)
        _COMPLETED_TAKEOVERS.pop(ext, None)
        _LOW_PDS_BONUS_USED.pop(ext, None)
        _repeat_early_stop_reset(ext)


def _record_trigger_stop_events(
    sampling_metadata: Any,
    output: torch.Tensor,
    internal_stop_output: torch.Tensor,
    student_topk_threshold_by_output: torch.Tensor | None = None,
    repeat_stop_output: torch.Tensor | None = None,
) -> None:
    req_id_strs = getattr(sampling_metadata, "req_ids", None)
    if req_id_strs is None or not internal_stop_output.any():
        return
    batch_size = int(output.shape[0])
    base_positions = _get_response_base_positions(sampling_metadata, batch_size)
    output_cpu = output.detach().cpu()
    trunc_cpu = internal_stop_output.detach().bool().cpu()
    student_topk_threshold_cpu = (
        student_topk_threshold_by_output.detach().cpu()
        if student_topk_threshold_by_output is not None
        else None
    )
    repeat_stop_cpu = (
        repeat_stop_output.detach().bool().cpu() if repeat_stop_output is not None else None
    )
    trace_path = os.environ.get("MAESTRO_RUNTIME_STOP_EVENTS_JSONL")
    file_records: list[dict[str, Any]] = []
    for bi in range(batch_size):
        if bi >= len(req_id_strs):
            continue
        rid_raw = req_id_strs[bi]
        if rid_raw is None:
            continue
        rid = _strip_vllm_suffix(str(rid_raw))
        response_pos = base_positions[bi] if base_positions is not None else 0
        for out_pos in torch.nonzero(trunc_cpu[bi], as_tuple=False).flatten().tolist():
            tok = int(output_cpu[bi, out_pos])
            rec: dict[str, Any] = {
                "request_id": rid,
                "response_pos": int(response_pos + out_pos),
                "spec_pos": int(out_pos),
                "internal_stop_token_id": tok,
                "stop_kind": (
                    "repeat"
                    if repeat_stop_cpu is not None and bool(repeat_stop_cpu[bi, out_pos])
                    else "trigger"
                ),
            }
            if student_topk_threshold_cpu is not None:
                threshold_val = int(student_topk_threshold_cpu[bi, out_pos])
                if threshold_val > 0:
                    rec["student_rank_teacher_argmax_gt"] = threshold_val
            if os.environ.get("MAESTRO_RUNTIME_MASK_IPC_SOCKET"):
                _send_stop_event_ipc(rid, rec)
            file_records.append(rec)

    if trace_path and file_records and _trace_file_writer_enabled():
        try:
            os.makedirs(os.path.dirname(trace_path), exist_ok=True)
            with open(trace_path, "a", encoding="utf-8") as f:
                for record in file_records:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            logger.exception("[opd-rollout] failed to write stop-event JSONL: %s", trace_path)


def _append_trace_events(
    sampling_metadata: Any,
    output: torch.Tensor,
    teacher_output: torch.Tensor,
    req_ids: torch.Tensor,
    pos_in_req: torch.Tensor,
    valid_pos: torch.Tensor,
    draft_token_ids: torch.Tensor,
    target_logits: torch.Tensor,
    draft_probs: torch.Tensor | None,
    trigger_mask: torch.Tensor | None = None,
    takeover_mask_by_output: torch.Tensor | None = None,
    standard_target_by_output: torch.Tensor | None = None,
) -> None:
    if not _trace_enabled():
        return
    req_id_strs = getattr(sampling_metadata, "req_ids", None)
    if req_id_strs is None:
        return
    batch_size = int(output.shape[0])
    base_positions = _get_response_base_positions(sampling_metadata, batch_size)

    with torch.no_grad():
        file_records: list[dict[str, Any]] = []
        emitted = output.ge(0) & output.ne(PLACEHOLDER_TOKEN_ID)
        teacher_cpu = teacher_output.detach().bool().cpu()
        emitted_cpu = emitted.detach().cpu()
        output_cpu = output.detach().cpu()
        req_cpu = req_ids.detach().cpu()
        pos_cpu = pos_in_req.detach().cpu()
        valid_cpu = valid_pos.detach().bool().cpu()

        if _trace_mode() == "mask":
            for bi in range(batch_size):
                rid_raw = req_id_strs[bi] if bi < len(req_id_strs) else None
                if rid_raw is None:
                    continue
                rid = _strip_vllm_suffix(str(rid_raw))
                response_pos = base_positions[bi] if base_positions is not None else 0
                chunk_positions: list[int] = []
                chunk_tokens: list[int] = []
                for out_pos in torch.nonzero(emitted_cpu[bi], as_tuple=False).flatten().tolist():
                    if bool(teacher_cpu[bi, out_pos]):
                        rec = {
                            "response_pos": int(response_pos),
                            "spec_pos": int(out_pos),
                            "emit_token_id": int(output_cpu[bi, out_pos]),
                            "has_logits": False,
                        }
                        _TRACE_EVENTS_BY_REQUEST_ID.setdefault(rid, []).append(rec)
                        chunk_positions.append(int(response_pos))
                        chunk_tokens.append(int(output_cpu[bi, out_pos]))
                    response_pos += 1
                if chunk_positions:
                    file_records.append({
                        "request_id": rid,
                        "positions": chunk_positions,
                        "tokens": chunk_tokens,
                    })
            trace_path = os.environ.get("MAESTRO_RUNTIME_TRACE_JSONL")
            if trace_path and file_records and _trace_file_writer_enabled():
                try:
                    os.makedirs(os.path.dirname(trace_path), exist_ok=True)
                    with open(trace_path, "a", encoding="utf-8") as f:
                        for record in file_records:
                            f.write(json.dumps(record, ensure_ascii=False) + "\n")
                except Exception:
                    logger.exception("[opd-rollout] failed to write SKD trace jsonl: %s", trace_path)
            return

        if draft_probs is None:
            return

        metric_rows = min(int(target_logits.shape[0]), int(draft_probs.shape[0]))
        if metric_rows <= 0:
            return
        target_logits = target_logits[:metric_rows]
        draft_probs = draft_probs[:metric_rows]

        topk = min(max(_int("MAESTRO_RUNTIME_TRACE_TOPK", 128), 1), int(target_logits.shape[-1]))
        target_logprobs = target_logits.float().log_softmax(dim=-1)
        target_top_logp, target_top_ids = torch.topk(target_logprobs, k=topk, dim=-1)
        target_top_probs = target_top_logp.exp()
        target_mass = target_top_probs.sum(dim=-1).clamp_min(1e-30)
        target_top_q = target_top_probs / target_mass[:, None]
        target_top_logq = target_top_q.clamp_min(1e-30).log()
        student_top_probs = draft_probs.gather(1, target_top_ids).clamp_min(1e-30)
        student_top_mass = student_top_probs.sum(dim=-1).clamp_min(1e-30)
        student_top_q = student_top_probs / student_top_mass[:, None]
        fkl_teacher_renorm = (target_top_q * (target_top_logq - student_top_probs.log())).sum(dim=-1)
        student_top_ids = draft_probs.topk(k=topk, dim=-1).indices
        teacher_in_student_topk_trace = target_top_ids[:, :, None].eq(student_top_ids[:, None, :]).any(dim=-1)
        topk_intersection = teacher_in_student_topk_trace.sum(dim=-1)
        topk_overlap = topk_intersection.to(torch.float32) / float(topk)
        topk_jaccard = topk_intersection.to(torch.float32) / (
            (2.0 * float(topk)) - topk_intersection.to(torch.float32)
        ).clamp_min(1.0)
        teacher_mass_coverage = (target_top_q * teacher_in_student_topk_trace.to(torch.float32)).sum(dim=-1)
        local_bhattacharyya = torch.sqrt(
            target_top_q.clamp_min(1e-30) * student_top_q.clamp_min(1e-30)
        ).sum(dim=-1)
        agreement = teacher_mass_coverage * local_bhattacharyya
        pds = (1.0 - agreement).clamp(0.0, 1.0)

        teacher_argmax = target_logprobs.argmax(dim=-1)
        safe_teacher_argmax = teacher_argmax.clamp_min(0).clamp_max(target_logits.shape[-1] - 1)
        teacher_argmax_logp = target_logprobs.gather(1, safe_teacher_argmax[:, None]).squeeze(1)
        student_argmax_prob = draft_probs.gather(1, safe_teacher_argmax[:, None]).squeeze(1).clamp_min(1e-30)
        student_argmax_logp = student_argmax_prob.log()

        safe_draft = draft_token_ids.long().clamp_min(0).clamp_max(target_logits.shape[-1] - 1)
        teacher_draft_logp = target_logprobs.gather(1, safe_draft[:, None]).squeeze(1)
        student_draft_prob = draft_probs.gather(1, safe_draft[:, None]).squeeze(1).clamp_min(1e-30)
        student_draft_logp = student_draft_prob.log()

        trigger_cpu = trigger_mask.detach().bool().cpu() if trigger_mask is not None else None
        takeover_cpu = takeover_mask_by_output.detach().bool().cpu() if takeover_mask_by_output is not None else None
        standard_target_cpu = (
            standard_target_by_output.detach().bool().cpu() if standard_target_by_output is not None else None
        )
        # Offline rewind-replay needs a per-token view of the *whole* response
        # (student tokens included), because the divergence point that a rewind
        # rule would target lives in the student author segment, not in the
        # teacher leg.  Gated so that training-side metrics (which average over
        # the in-memory trace) keep their historical teacher-only definition.
        trace_all_tokens = _flag("MAESTRO_RUNTIME_TRACE_ALL_TOKENS", False)

        flat_by_req_pos: dict[tuple[int, int], int] = {}
        for flat_idx in range(int(req_cpu.numel())):
            if not bool(valid_cpu[flat_idx]):
                continue
            flat_by_req_pos[(int(req_cpu[flat_idx]), int(pos_cpu[flat_idx]))] = flat_idx

        for bi in range(batch_size):
            rid_raw = req_id_strs[bi] if bi < len(req_id_strs) else None
            if rid_raw is None:
                continue
            rid = _strip_vllm_suffix(str(rid_raw))
            response_pos = base_positions[bi] if base_positions is not None else 0
            for out_pos in torch.nonzero(emitted_cpu[bi], as_tuple=False).flatten().tolist():
                is_teacher = bool(teacher_cpu[bi, out_pos])
                if not is_teacher and not trace_all_tokens:
                    response_pos += 1
                    continue
                flat_idx = flat_by_req_pos.get((bi, int(out_pos)))
                rec: dict[str, Any] = {
                    "response_pos": int(response_pos),
                    "spec_pos": int(out_pos),
                    "emit_token_id": int(output_cpu[bi, out_pos]),
                    "has_logits": flat_idx is not None,
                    "is_teacher": bool(is_teacher),
                }
                if flat_idx is not None and flat_idx < metric_rows:
                    rec.update({
                        "draft_token_id": int(draft_token_ids.detach().cpu()[flat_idx]),
                        "teacher_argmax_token_id": int(teacher_argmax.detach().cpu()[flat_idx]),
                        "is_trigger_first_token": bool(trigger_cpu[flat_idx]) if trigger_cpu is not None else False,
                        "is_takeover_continuation": (
                            bool(takeover_cpu[bi, out_pos]) if takeover_cpu is not None else False
                        ),
                        "is_standard_target_emit": (
                            bool(standard_target_cpu[bi, out_pos]) if standard_target_cpu is not None else False
                        ),
                        "teacher_logp_teacher_argmax": float(teacher_argmax_logp.detach().cpu()[flat_idx]),
                        "student_logp_teacher_argmax": float(student_argmax_logp.detach().cpu()[flat_idx]),
                        "student_p_teacher_argmax": float(student_argmax_prob.detach().cpu()[flat_idx]),
                        "teacher_student_logp_gap_argmax": float(
                            (teacher_argmax_logp - student_argmax_logp).detach().cpu()[flat_idx]
                        ),
                        "teacher_logp_draft": float(teacher_draft_logp.detach().cpu()[flat_idx]),
                        "student_logp_draft": float(student_draft_logp.detach().cpu()[flat_idx]),
                        "teacher_topk_mass": float(target_mass.detach().cpu()[flat_idx]),
                        "student_mass_on_teacher_topk": float(student_top_mass.detach().cpu()[flat_idx]),
                        "fkl_topk_teacher_renorm": float(fkl_teacher_renorm.detach().cpu()[flat_idx]),
                        "teacher_student_topk_overlap": float(topk_overlap.detach().cpu()[flat_idx]),
                        "teacher_student_topk_jaccard": float(topk_jaccard.detach().cpu()[flat_idx]),
                        "pds": float(pds.detach().cpu()[flat_idx]),
                        "teacher_mass_coverage": float(teacher_mass_coverage.detach().cpu()[flat_idx]),
                        "local_bhattacharyya": float(local_bhattacharyya.detach().cpu()[flat_idx]),
                        "student_p_teacher_top1": float(student_argmax_prob.detach().cpu()[flat_idx]),
                        "teacher_student_logp_gap_top1": float(
                            (teacher_argmax_logp - student_argmax_logp).detach().cpu()[flat_idx]
                        ),
                    })
                elif flat_idx is not None:
                    rec["has_logits"] = False
                if is_teacher or not trace_all_tokens:
                    _TRACE_EVENTS_BY_REQUEST_ID.setdefault(rid, []).append(rec)
                file_records.append({"request_id": rid, **rec})
                response_pos += 1
        trace_path = os.environ.get("MAESTRO_RUNTIME_TRACE_JSONL")
        if trace_path and file_records and _trace_file_writer_enabled():
            try:
                os.makedirs(os.path.dirname(trace_path), exist_ok=True)
                with open(trace_path, "a", encoding="utf-8") as f:
                    for record in file_records:
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
            except Exception:
                logger.exception("[opd-rollout] failed to write SKD trace jsonl: %s", trace_path)


def _get_response_base_positions(sampling_metadata: Any, batch_size: int) -> list[int] | None:
    """Return response-relative base positions for the current vLLM sample call."""
    base_positions = getattr(sampling_metadata, "verl_opd_response_base_positions", None)
    if base_positions is None:
        return None
    try:
        if len(base_positions) < batch_size:
            return None
        return [int(x) for x in base_positions[:batch_size]]
    except Exception:
        return None


def _mechanical_phase(
    response_pos: int, student_tokens: int, teacher_tokens: int, one_shot: bool = False
) -> tuple[bool, int]:
    """Return whether ``response_pos`` is teacher-owned and tokens to the next phase."""
    fixed_raw = os.environ.get("MAESTRO_MECHANICAL_TAKEOVER_POSITIONS", "").strip()
    if fixed_raw:
        try:
            positions = sorted({int(x.strip()) for x in fixed_raw.split(",") if x.strip()})
        except ValueError as exc:
            raise ValueError("MAESTRO_MECHANICAL_TAKEOVER_POSITIONS must be comma-separated integers") from exc
        if any(x < 0 for x in positions):
            raise ValueError("MAESTRO_MECHANICAL_TAKEOVER_POSITIONS must be non-negative")
        if teacher_tokens <= 0:
            raise ValueError("fixed mechanical schedule requires teacher_tokens > 0")
        for start in positions:
            if start <= response_pos < start + teacher_tokens:
                return True, start + teacher_tokens - response_pos
        return False, 2**31 - 1
    if student_tokens == -1:
        return False, 2**31 - 1
    if student_tokens <= 0 or teacher_tokens <= 0:
        raise ValueError("mechanical schedule requires student_tokens > 0 (or -1 control) and teacher_tokens > 0")
    cycle = student_tokens + teacher_tokens
    if one_shot and response_pos >= cycle:
        return False, 2**31 - 1
    offset = response_pos if one_shot else response_pos % cycle
    if offset < student_tokens:
        return False, student_tokens - offset
    return True, cycle - offset


def _get_pending_mask_rows(sampling_metadata: Any, batch_size: int) -> list[dict[str, list[int] | list[bool] | list[float]]]:
    pending = getattr(sampling_metadata, "verl_opd_pending_mask_rows", None)
    if pending is None or len(pending) < batch_size:
        pending = [
            {
                "mask": [],
                "tokens": [],
                "actions": [],
                "teacher_action_logps": [],
                "topk_overlaps": [],
                "pds": [],
                "teacher_mass_coverage": [],
                "local_bhattacharyya": [],
                "student_p_teacher_top1": [],
                "teacher_student_logp_gap_top1": [],
            }
            for _ in range(batch_size)
        ]
        sampling_metadata.verl_opd_pending_mask_rows = pending
    return pending


def _record_token_source_events(
    sampling_metadata: Any,
    batch_size: int,
    device: torch.device,
    output: torch.Tensor,
    is_teacher_per_req: torch.Tensor | None = None,
    is_teacher_per_token: torch.Tensor | None = None,
    req_ids: torch.Tensor | None = None,
    pos_in_req: torch.Tensor | None = None,
    teacher_output_override: torch.Tensor | None = None,
    action_token_by_output: torch.Tensor | None = None,
    teacher_action_logp_by_output: torch.Tensor | None = None,
    topk_overlap_by_output: torch.Tensor | None = None,
    pds_by_output: torch.Tensor | None = None,
    teacher_mass_coverage_by_output: torch.Tensor | None = None,
    local_bhattacharyya_by_output: torch.Tensor | None = None,
    student_p_teacher_top1_by_output: torch.Tensor | None = None,
    teacher_student_logp_gap_top1_by_output: torch.Tensor | None = None,
    valid_vocab_size: int | None = None,
) -> None:
    """Record every emitted token in a speculative chunk.

    A speculative step may emit several accepted draft tokens, so the event
    stream includes every non-placeholder output position.
    """
    req_id_strs = getattr(sampling_metadata, "req_ids", None)
    if req_id_strs is None:
        return

    emitted = output.ge(0) & output.ne(PLACEHOLDER_TOKEN_ID)
    if not emitted.any():
        return

    if teacher_output_override is not None:
        teacher_output = teacher_output_override.detach().bool() & emitted
    else:
        teacher_output = torch.zeros_like(output, dtype=torch.bool)
    if teacher_output_override is None and is_teacher_per_token is not None and req_ids is not None and pos_in_req is not None:
        valid_teacher = (
            is_teacher_per_token
            & (req_ids >= 0)
            & (req_ids < batch_size)
            & (pos_in_req >= 0)
            & (pos_in_req < output.shape[1])
        )
        if valid_teacher.any():
            teacher_output[req_ids[valid_teacher], pos_in_req[valid_teacher]] = True
    elif teacher_output_override is None and is_teacher_per_req is not None:
        teacher_req = is_teacher_per_req.detach().bool()
        teacher_output = emitted & teacher_req[:, None]

    output_cpu = output.detach().cpu()
    emitted_cpu = emitted.detach().cpu()
    teacher_cpu = teacher_output.detach().cpu()
    action_cpu = action_token_by_output.detach().cpu() if action_token_by_output is not None else None
    action_logp_cpu = (
        teacher_action_logp_by_output.detach().cpu() if teacher_action_logp_by_output is not None else None
    )
    overlap_cpu = topk_overlap_by_output.detach().cpu() if topk_overlap_by_output is not None else None
    pds_cpu = pds_by_output.detach().cpu() if pds_by_output is not None else None
    teacher_mass_coverage_cpu = (
        teacher_mass_coverage_by_output.detach().cpu()
        if teacher_mass_coverage_by_output is not None
        else None
    )
    local_bhattacharyya_cpu = (
        local_bhattacharyya_by_output.detach().cpu()
        if local_bhattacharyya_by_output is not None
        else None
    )
    student_p_teacher_top1_cpu = (
        student_p_teacher_top1_by_output.detach().cpu()
        if student_p_teacher_top1_by_output is not None
        else None
    )
    teacher_student_logp_gap_top1_cpu = (
        teacher_student_logp_gap_top1_by_output.detach().cpu()
        if teacher_student_logp_gap_top1_by_output is not None
        else None
    )
    base_positions = _get_response_base_positions(sampling_metadata, batch_size)
    use_ipc = bool(os.environ.get("MAESTRO_RUNTIME_MASK_IPC_SOCKET"))
    token_trace_path = os.environ.get("MAESTRO_RUNTIME_TRACE_TOKEN_JSONL")
    token_file_records: list[dict[str, Any]] = []
    for i in range(batch_size):
        if i >= len(req_id_strs):
            continue
        rid = req_id_strs[i]
        if rid is None:
            continue
        positions = torch.nonzero(emitted_cpu[i], as_tuple=False).flatten().tolist()
        if not positions:
            continue
        mask_chunk: list[bool] = []
        token_chunk: list[int] = []
        action_chunk: list[int] | None = [] if action_cpu is not None else None
        action_logp_chunk: list[float] | None = [] if action_logp_cpu is not None else None
        overlap_chunk: list[float] | None = [] if overlap_cpu is not None else None
        pds_chunk: list[float] | None = [] if pds_cpu is not None else None
        teacher_mass_coverage_chunk: list[float] | None = [] if teacher_mass_coverage_cpu is not None else None
        local_bhattacharyya_chunk: list[float] | None = [] if local_bhattacharyya_cpu is not None else None
        student_p_teacher_top1_chunk: list[float] | None = [] if student_p_teacher_top1_cpu is not None else None
        teacher_student_logp_gap_top1_chunk: list[float] | None = (
            [] if teacher_student_logp_gap_top1_cpu is not None else None
        )
        position_chunk: list[int] | None = [] if base_positions is not None else None
        for pos in positions:
            tok = int(output_cpu[i, pos])
            if tok == PLACEHOLDER_TOKEN_ID or tok < 0 or (valid_vocab_size is not None and tok >= valid_vocab_size):
                continue
            response_pos = int(base_positions[i]) + len(token_chunk) if base_positions is not None else None
            token_chunk.append(tok)
            mask_chunk.append(bool(teacher_cpu[i, pos]))
            if action_chunk is not None:
                action_chunk.append(int(action_cpu[i, pos]))
            if action_logp_chunk is not None:
                action_logp_chunk.append(float(action_logp_cpu[i, pos]))
            if overlap_chunk is not None:
                overlap_chunk.append(float(overlap_cpu[i, pos]))
            if pds_chunk is not None:
                pds_chunk.append(float(pds_cpu[i, pos]))
            if teacher_mass_coverage_chunk is not None:
                teacher_mass_coverage_chunk.append(float(teacher_mass_coverage_cpu[i, pos]))
            if local_bhattacharyya_chunk is not None:
                local_bhattacharyya_chunk.append(float(local_bhattacharyya_cpu[i, pos]))
            if student_p_teacher_top1_chunk is not None:
                student_p_teacher_top1_chunk.append(float(student_p_teacher_top1_cpu[i, pos]))
            if teacher_student_logp_gap_top1_chunk is not None:
                teacher_student_logp_gap_top1_chunk.append(float(teacher_student_logp_gap_top1_cpu[i, pos]))
            if position_chunk is not None:
                position_chunk.append(int(response_pos))
        if not token_chunk:
            continue
        external_rid = _strip_vllm_suffix(rid)
        if token_trace_path:
            for local_pos, tok in enumerate(token_chunk):
                response_pos = (
                    int(position_chunk[local_pos])
                    if position_chunk is not None
                    else int(local_pos)
                )
                rec: dict[str, Any] = {
                    "request_id": external_rid,
                    "response_pos": response_pos,
                    "spec_pos": response_pos,
                    "emit_token_id": int(tok),
                    "is_teacher": bool(mask_chunk[local_pos]),
                }
                if overlap_chunk is not None:
                    rec["topk_overlap"] = float(overlap_chunk[local_pos])
                if pds_chunk is not None:
                    rec["pds"] = float(pds_chunk[local_pos])
                if teacher_mass_coverage_chunk is not None:
                    rec["teacher_mass_coverage"] = float(teacher_mass_coverage_chunk[local_pos])
                if local_bhattacharyya_chunk is not None:
                    rec["local_bhattacharyya"] = float(local_bhattacharyya_chunk[local_pos])
                if student_p_teacher_top1_chunk is not None:
                    rec["student_p_teacher_top1"] = float(student_p_teacher_top1_chunk[local_pos])
                if teacher_student_logp_gap_top1_chunk is not None:
                    rec["teacher_student_logp_gap_top1"] = float(
                        teacher_student_logp_gap_top1_chunk[local_pos]
                    )
                if action_chunk is not None:
                    rec["draft_action_token_id"] = int(action_chunk[local_pos])
                if action_logp_chunk is not None:
                    rec["teacher_action_logp"] = float(action_logp_chunk[local_pos])
                token_file_records.append(rec)
        if use_ipc:
            pending_rows = _get_pending_mask_rows(sampling_metadata, batch_size)
            pending_rows[i]["mask"].extend(mask_chunk)
            pending_rows[i]["tokens"].extend(token_chunk)
            if action_chunk is not None:
                pending_rows[i]["actions"].extend(action_chunk)
            if action_logp_chunk is not None:
                pending_rows[i]["teacher_action_logps"].extend(action_logp_chunk)
            if overlap_chunk is not None:
                pending_rows[i]["topk_overlaps"].extend(overlap_chunk)
            if pds_chunk is not None:
                pending_rows[i]["pds"].extend(pds_chunk)
            if teacher_mass_coverage_chunk is not None:
                pending_rows[i]["teacher_mass_coverage"].extend(teacher_mass_coverage_chunk)
            if local_bhattacharyya_chunk is not None:
                pending_rows[i]["local_bhattacharyya"].extend(local_bhattacharyya_chunk)
            if student_p_teacher_top1_chunk is not None:
                pending_rows[i]["student_p_teacher_top1"].extend(student_p_teacher_top1_chunk)
            if teacher_student_logp_gap_top1_chunk is not None:
                pending_rows[i]["teacher_student_logp_gap_top1"].extend(
                    teacher_student_logp_gap_top1_chunk
                )
        else:
            _TEACHER_MASK_BY_REQUEST_ID.setdefault(external_rid, []).extend(mask_chunk)
            _TOKEN_EVENTS_BY_REQUEST_ID.setdefault(external_rid, []).extend(token_chunk)
            if position_chunk is not None:
                _POSITION_EVENTS_BY_REQUEST_ID.setdefault(external_rid, []).extend(position_chunk)
            if overlap_chunk is not None:
                _OVERLAP_EVENTS_BY_REQUEST_ID.setdefault(external_rid, []).extend(overlap_chunk)
    if token_trace_path and token_file_records and _trace_file_writer_enabled():
        try:
            os.makedirs(os.path.dirname(token_trace_path), exist_ok=True)
            with open(token_trace_path, "a", encoding="utf-8") as f:
                for record in token_file_records:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            logger.exception("[opd-rollout] failed to write token trace jsonl: %s", token_trace_path)


def _compute_topk_overlap_by_output(
    output: torch.Tensor,
    req_ids: torch.Tensor,
    pos_in_req: torch.Tensor,
    valid_pos: torch.Tensor,
    draft_probs: torch.Tensor | None,
    target_logits: torch.Tensor,
    batch_size: int,
    topk: int,
) -> torch.Tensor | None:
    """Return response-position aligned top-k overlap ratios for teacher/student logits."""
    if draft_probs is None or target_logits.numel() == 0:
        return None
    with torch.no_grad():
        n_rows = min(int(target_logits.shape[0]), int(draft_probs.shape[0]))
        if n_rows <= 0:
            return None
        target_logits = target_logits[:n_rows]
        draft_probs = draft_probs[:n_rows]
        vocab_size = int(target_logits.shape[-1])
        k = min(max(int(topk), 1), vocab_size, int(draft_probs.shape[-1]))
        if k <= 0:
            return None
        teacher_top_ids = target_logits.float().topk(k=k, dim=-1).indices
        student_top_ids = draft_probs.topk(k=k, dim=-1).indices
        intersection = (
            student_top_ids[:, :, None]
            .eq(teacher_top_ids[:, None, :])
            .any(dim=-1)
            .sum(dim=-1)
            .to(torch.float32)
        )
        overlap = intersection / float(k)
        aligned = torch.full(output.shape, float("nan"), dtype=torch.float32, device=output.device)
        valid_out = (
            valid_pos
            & (req_ids >= 0)
            & (req_ids < batch_size)
            & (pos_in_req >= 0)
            & (pos_in_req < output.shape[1])
        )
        if valid_out.any():
            idx = torch.nonzero(valid_out, as_tuple=False).flatten()
            fill_n = min(int(idx.numel()), int(overlap.shape[0]))
            if fill_n > 0:
                idx = idx[:fill_n]
                aligned[req_ids[idx], pos_in_req[idx]] = overlap[:fill_n]
        return aligned


def _align_flat_metric_by_output(
    output: torch.Tensor,
    req_ids: torch.Tensor,
    pos_in_req: torch.Tensor,
    valid_pos: torch.Tensor,
    values: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    aligned = torch.full(output.shape, float("nan"), dtype=torch.float32, device=output.device)
    valid_out = (
        valid_pos
        & (req_ids >= 0)
        & (req_ids < batch_size)
        & (pos_in_req >= 0)
        & (pos_in_req < output.shape[1])
    )
    if valid_out.any():
        idx = torch.nonzero(valid_out, as_tuple=False).flatten()
        fill_n = min(int(idx.numel()), int(values.shape[0]))
        if fill_n > 0:
            idx = idx[:fill_n]
            aligned[req_ids[idx], pos_in_req[idx]] = values[:fill_n].to(torch.float32)
    return aligned


def _compute_pds_metrics_flat(
    draft_probs: torch.Tensor | None,
    target_logits: torch.Tensor,
    topk: int,
) -> dict[str, torch.Tensor] | None:
    """Compute the paper-defined PDS divergence for flat speculative rows."""
    if draft_probs is None or target_logits.numel() == 0:
        return None
    with torch.no_grad():
        n_rows = min(int(target_logits.shape[0]), int(draft_probs.shape[0]))
        if n_rows <= 0:
            return None
        target_logits = target_logits[:n_rows]
        draft_probs = draft_probs[:n_rows]
        vocab_size = min(int(target_logits.shape[-1]), int(draft_probs.shape[-1]))
        k = min(max(int(topk), 1), vocab_size)
        if k <= 0:
            return None

        target_logits = target_logits[:, :vocab_size].float()
        draft_probs = draft_probs[:, :vocab_size]
        teacher_top_logits, teacher_top_ids = torch.topk(target_logits, k=k, dim=-1)
        teacher_top_logq = teacher_top_logits.log_softmax(dim=-1)
        teacher_top_q = teacher_top_logq.exp()
        teacher_top_mass = torch.ones_like(teacher_top_q[:, 0])

        student_probs_on_teacher_topk = draft_probs.gather(1, teacher_top_ids).clamp_min(1e-30)
        student_mass_on_teacher_topk = student_probs_on_teacher_topk.sum(dim=-1).clamp_min(1e-30)
        student_top_q_on_teacher = student_probs_on_teacher_topk / student_mass_on_teacher_topk[:, None]

        student_top_ids = draft_probs.topk(k=k, dim=-1).indices
        teacher_in_student_topk = teacher_top_ids[:, :, None].eq(student_top_ids[:, None, :]).any(dim=-1)
        intersection = teacher_in_student_topk.sum(dim=-1).to(torch.float32)
        topk_overlap = intersection / float(k)
        topk_jaccard = intersection / ((2.0 * float(k)) - intersection).clamp_min(1.0)

        teacher_mass_coverage = (teacher_top_q * teacher_in_student_topk.to(torch.float32)).sum(dim=-1)
        local_bhattacharyya = torch.sqrt(
            teacher_top_q.clamp_min(1e-30) * student_top_q_on_teacher.clamp_min(1e-30)
        ).sum(dim=-1)
        agreement = teacher_mass_coverage * local_bhattacharyya
        pds = (1.0 - agreement).clamp(0.0, 1.0)

        teacher_top1_ids = teacher_top_ids[:, 0]
        teacher_top1_logp = teacher_top_logq[:, 0]
        student_p_teacher_top1 = student_probs_on_teacher_topk[:, 0].clamp_min(1e-30)
        student_logp_teacher_top1 = student_p_teacher_top1.log()
        teacher_student_logp_gap_top1 = teacher_top1_logp - student_logp_teacher_top1
        fkl_topk_teacher_renorm = (
            teacher_top_q * (teacher_top_logq - student_probs_on_teacher_topk.log())
        ).sum(dim=-1)

        return {
            "pds": pds.to(torch.float32),
            "teacher_mass_coverage": teacher_mass_coverage.to(torch.float32),
            "local_bhattacharyya": local_bhattacharyya.to(torch.float32),
            "student_p_teacher_top1": student_p_teacher_top1.to(torch.float32),
            "teacher_student_logp_gap_top1": teacher_student_logp_gap_top1.to(torch.float32),
            "topk_overlap": topk_overlap.to(torch.float32),
            "topk_jaccard": topk_jaccard.to(torch.float32),
            "teacher_topk_mass": teacher_top_mass.to(torch.float32),
            "student_mass_on_teacher_topk": student_mass_on_teacher_topk.to(torch.float32),
            "fkl_topk_teacher_renorm": fkl_topk_teacher_renorm.to(torch.float32),
            "teacher_top1_id": teacher_top1_ids,
            "teacher_top1_logp": teacher_top1_logp.to(torch.float32),
            "student_logp_teacher_top1": student_logp_teacher_top1.to(torch.float32),
        }


def _compute_pds_metrics_by_output(
    output: torch.Tensor,
    req_ids: torch.Tensor,
    pos_in_req: torch.Tensor,
    valid_pos: torch.Tensor,
    draft_probs: torch.Tensor | None,
    target_logits: torch.Tensor,
    batch_size: int,
    topk: int,
) -> dict[str, torch.Tensor] | None:
    flat = _compute_pds_metrics_flat(draft_probs=draft_probs, target_logits=target_logits, topk=topk)
    if flat is None:
        return None
    aligned: dict[str, torch.Tensor] = {}
    for key in (
        "pds",
        "teacher_mass_coverage",
        "local_bhattacharyya",
        "student_p_teacher_top1",
        "teacher_student_logp_gap_top1",
        "topk_overlap",
    ):
        aligned[key] = _align_flat_metric_by_output(
            output=output,
            req_ids=req_ids,
            pos_in_req=pos_in_req,
            valid_pos=valid_pos,
            values=flat[key],
            batch_size=batch_size,
        )
    return aligned


def _clear_pds_state(request_id: str | None) -> None:
    if request_id is None:
        return
    _PDS_WINDOW_VALUES.pop(request_id, None)
    _PDS_PARAGRAPH_VALUES.pop(request_id, None)
    _PDS_BAD_WINDOW_STREAK.pop(request_id, None)
    _PDS_FALLBACK_STREAK.pop(request_id, None)
    _PDS_LAST_ROLLING_MEAN.pop(request_id, None)


def _clear_takeover_recovery_state(request_id: str | None) -> None:
    if request_id is None:
        return
    _TAKEOVER_RECOVERY_VALUES.pop(request_id, None)
    _TAKEOVER_RECOVERY_SEEN.pop(request_id, None)
    _TAKEOVER_RECOVERY_GOOD_STREAK.pop(request_id, None)
    _TAKEOVER_RECOVERY_READY.pop(request_id, None)


def _observe_takeover_recovery(
    request_id: str,
    pds: float,
    *,
    window_tokens: int,
    window_stride: int,
    threshold: float,
    patience: int,
) -> tuple[bool, float | None]:
    """Update one teacher-leg recovery controller from an emitted draft/teacher pair.

    Returns whether recovery became ready at this token and the PDS window mean
    when a window boundary was observed.  A bad observed window clears a prior
    ready state, so a delayed paragraph boundary cannot hand control back after
    the student has diverged again.
    """
    window = _TAKEOVER_RECOVERY_VALUES.setdefault(request_id, [])
    window.append(pds)
    if len(window) > window_tokens:
        del window[:-window_tokens]
    seen = _TAKEOVER_RECOVERY_SEEN.get(request_id, 0) + 1
    _TAKEOVER_RECOVERY_SEEN[request_id] = seen
    if len(window) < window_tokens or seen % window_stride != 0:
        return False, None

    window_mean = float(sum(window) / len(window))
    if window_mean <= threshold:
        streak = _TAKEOVER_RECOVERY_GOOD_STREAK.get(request_id, 0) + 1
        _TAKEOVER_RECOVERY_GOOD_STREAK[request_id] = streak
        if streak >= patience and not _TAKEOVER_RECOVERY_READY.get(request_id, False):
            _TAKEOVER_RECOVERY_READY[request_id] = True
            return True, window_mean
    else:
        _TAKEOVER_RECOVERY_GOOD_STREAK[request_id] = 0
        _TAKEOVER_RECOVERY_READY[request_id] = False
    return False, window_mean


def _sample_from_logits(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # These defaults mirror the Qwen3 non-thinking recommendation.  Launchers
    # still set them explicitly; keeping safe defaults prevents the speculative
    # draft path from silently sampling with a different distribution than the
    # public rollout request.
    temperature = max(_float("MAESTRO_RUNTIME_DRAFT_TEMPERATURE", 0.7), 0.0)
    top_p = _float("MAESTRO_RUNTIME_DRAFT_TOP_P", 0.8)
    top_k = _int("MAESTRO_RUNTIME_DRAFT_TOP_K", 20)

    logits_f = logits.to(torch.float32)
    if temperature <= 1e-6:
        probs = torch.zeros_like(logits_f)
        token_ids = logits_f.argmax(dim=-1)
        probs.scatter_(1, token_ids[:, None], 1.0)
        return token_ids, probs

    logits_f = logits_f / temperature
    if top_k is not None and top_k > 0 and top_k < logits_f.shape[-1]:
        kth = logits_f.topk(top_k, dim=-1).values[:, -1, None]
        logits_f = logits_f.masked_fill(logits_f < kth, float("-inf"))

    if top_p < 1.0:
        sorted_logits, sorted_idx = logits_f.sort(dim=-1, descending=True)
        sorted_probs = sorted_logits.softmax(dim=-1, dtype=torch.float32)
        remove = sorted_probs.cumsum(dim=-1) > top_p
        remove[:, 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        filtered = torch.full_like(logits_f, float("-inf"))
        logits_f = filtered.scatter(1, sorted_idx, sorted_logits)

    probs = logits_f.softmax(dim=-1, dtype=torch.float32)
    probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0).clamp_min_(0.0)
    row_sum = probs.sum(dim=-1, keepdim=True)
    degenerate = row_sum.squeeze(-1) <= 0
    probs = torch.where(
        degenerate[:, None],
        torch.full_like(probs, 1.0 / max(probs.shape[-1], 1)),
        probs / row_sum.clamp_min(1e-30),
    )

    race = torch.empty_like(probs)
    race.exponential_()
    token_ids = (probs / race).argmax(dim=-1)
    token_ids = torch.where(degenerate, logits.to(torch.float32).argmax(dim=-1), token_ids)
    return token_ids, probs


def _request_layout(
    num_draft_tokens: list[int],
    num_tokens: int,
    max_spec_len: int,
    cu_num_draft_tokens: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = len(num_draft_tokens)
    lengths = torch.tensor(num_draft_tokens, dtype=torch.long, device=device)
    req_idx_all = torch.arange(batch_size, device=device, dtype=torch.long)
    req_ids = torch.repeat_interleave(req_idx_all, lengths)
    starts = cu_num_draft_tokens.to(device=device, dtype=torch.long) - lengths
    pos_in_req = torch.arange(num_tokens, device=device, dtype=torch.long)
    pos_in_req = pos_in_req - torch.repeat_interleave(starts, lengths)
    valid_pos = (pos_in_req >= 0) & (pos_in_req < (max_spec_len + 1))
    return lengths, req_idx_all, req_ids, pos_in_req, valid_pos


# ---- Reflection-token resolution ----

_REFLECTION_TOKEN_IDS_CACHE: dict[str, torch.Tensor] = {}
_REFLECTION_TOKENS_LOGGED = False


def _get_reflection_token_ids(device: torch.device) -> torch.Tensor:
    global _REFLECTION_TOKENS_LOGGED
    env_val = os.environ.get(
        "MAESTRO_REFLECTION_TOKENS",
        "Wait, Wait,wait, wait,But, But,but, but,"
        "Hmm, Hmm, hmm,Actually, Actually,actually, actually,"
        "Hold, Hold,hold, hold,However, However,however, however,"
        "Yet, Yet,yet, yet,Oh, Oh,oh, oh,"
        "Alternatively, Alternatively,No, No,no, no,"
        "Ah, Ah,ah, ah,Oops, Oops,Well, Well",
    )
    cache_key = f"{device}:{env_val}"
    cached = _REFLECTION_TOKEN_IDS_CACHE.get(cache_key)
    if cached is not None:
        return cached
    from transformers import AutoTokenizer
    tok_path = os.environ.get("MAESTRO_RUNTIME_TOKENIZER_PATH") or os.environ.get(
        "MAESTRO_RUNTIME_TARGET_MODEL"
    ) or os.environ.get("STUDENT_MODEL")
    if not tok_path:
        logger.warning("[opd-rollout] no tokenizer path resolvable from env")
        tensor = torch.zeros(0, dtype=torch.long, device=device)
        _REFLECTION_TOKEN_IDS_CACHE[cache_key] = tensor
        return tensor
    tok = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
    markers = [s for s in env_val.split(",") if s]
    ids: list[int] = []
    skipped: list[str] = []
    for m in markers:
        enc = tok.encode(m, add_special_tokens=False)
        if len(enc) == 1:
            tid = int(enc[0])
            if tid not in ids:
                ids.append(tid)
        else:
            skipped.append(m)
    if not _REFLECTION_TOKENS_LOGGED:
        logger.warning(
            "[opd-rollout] trigger token IDs (n=%d): %s; skipped multi-token: %s",
            len(ids), ids, skipped,
        )
        _REFLECTION_TOKENS_LOGGED = True
    tensor = torch.tensor(ids, dtype=torch.long, device=device)
    _REFLECTION_TOKEN_IDS_CACHE[cache_key] = tensor
    return tensor


# ---- Rejection sample modes ----

def _skd_topk_rejection_sample(
    draft_token_ids: torch.Tensor,
    num_draft_tokens: list[int],
    max_spec_len: int,
    cu_num_draft_tokens: torch.Tensor,
    draft_probs: torch.Tensor | None,
    target_logits: torch.Tensor,
    bonus_token_ids: torch.Tensor,
    sampling_metadata: Any,
    **kwargs: Any,
) -> torch.Tensor:
    """Google-style SKD top-k speculative replacement.

    Draft tokens are accepted while they are in the teacher top-k set. At the
    first rejection for each request, emit a teacher top-k sample and mark only
    that emitted token as teacher-owned for loss routing.
    """
    del kwargs, draft_probs, bonus_token_ids
    device = target_logits.device
    batch_size = len(num_draft_tokens)
    num_tokens, vocab_size = target_logits.shape
    output = torch.full(
        (batch_size, max_spec_len + 1),
        PLACEHOLDER_TOKEN_ID,
        dtype=torch.int32,
        device=device,
    )
    if num_tokens == 0:
        return output

    k = min(max(_int("SKD_ACCEPTANCE_TOPK", 25), 1), vocab_size)
    valid_draft = (draft_token_ids >= 0) & (draft_token_ids < vocab_size)
    if k == 1:
        teacher_argmax = target_logits.argmax(dim=-1)
        accept_mask = teacher_argmax.eq(draft_token_ids.long()) & valid_draft
    else:
        topk_ids = target_logits.topk(k, dim=-1).indices
        teacher_argmax = topk_ids[:, 0]
        accept_mask = topk_ids.eq(draft_token_ids.long()[:, None]).any(dim=-1) & valid_draft

    _, _, req_ids, pos_in_req, valid_pos = _request_layout(
        num_draft_tokens, num_tokens, max_spec_len, cu_num_draft_tokens, device
    )
    first_rej_pos = torch.full((batch_size,), max_spec_len, dtype=torch.long, device=device)
    rejected = torch.nonzero((~accept_mask) & valid_pos, as_tuple=False).flatten()
    if rejected.numel() > 0:
        first_rej_pos.scatter_reduce_(
            dim=0,
            index=req_ids[rejected],
            src=pos_in_req[rejected],
            reduce="amin",
            include_self=True,
        )

    first_for_token = first_rej_pos[req_ids]
    emit_accept = accept_mask & (pos_in_req < first_for_token) & valid_pos
    emit_reject = (~accept_mask) & (pos_in_req == first_for_token) & valid_pos

    if emit_accept.any():
        output[req_ids[emit_accept], pos_in_req[emit_accept]] = draft_token_ids[emit_accept].to(torch.int32)

    reject_flat = torch.nonzero(emit_reject, as_tuple=False).flatten()
    if reject_flat.numel() > 0:
        reject_logits = target_logits.index_select(0, reject_flat)
        reject_probs = reject_logits.softmax(dim=-1, dtype=torch.float32)
        reject_probs = torch.nan_to_num(reject_probs, nan=0.0, posinf=0.0, neginf=0.0).clamp_min_(0.0)

        topk_idx = reject_logits.topk(k, dim=-1).indices
        topk_mask = torch.zeros_like(reject_probs)
        topk_mask.scatter_(1, topk_idx, 1.0)
        reject_probs = reject_probs * topk_mask

        reject_draft = draft_token_ids.index_select(0, reject_flat).long()
        valid_reject_draft = (reject_draft >= 0) & (reject_draft < vocab_size)
        if valid_reject_draft.any():
            reject_probs[valid_reject_draft].scatter_(1, reject_draft[valid_reject_draft, None], 0.0)

        row_sum = reject_probs.sum(dim=-1, keepdim=True)
        degenerate = row_sum.squeeze(-1) <= 0
        reject_probs = torch.where(
            degenerate[:, None],
            torch.full_like(reject_probs, 1.0 / max(vocab_size, 1)),
            reject_probs / row_sum.clamp_min(1e-30),
        )
        sampled = torch.multinomial(reject_probs, num_samples=1).squeeze(-1)
        recovered = torch.where(degenerate, teacher_argmax.index_select(0, reject_flat), sampled)
        output[req_ids[reject_flat], pos_in_req[reject_flat]] = recovered.to(torch.int32)

    teacher_output = torch.zeros_like(output, dtype=torch.bool)
    if reject_flat.numel() > 0:
        teacher_output[req_ids[reject_flat], pos_in_req[reject_flat]] = True

    attempted = valid_pos & valid_draft
    _stats_add("calls", 1, device=device)
    _stats_add("requests", batch_size, device=device)
    _stats_add("drafted_tokens", attempted)
    _stats_add("emitted_tokens", emit_accept.sum() + reject_flat.numel(), device=device)
    _stats_add("student_tokens", emit_accept)
    _stats_add("teacher_tokens", reject_flat.numel(), device=device)
    _stats_add("teacher_requests", reject_flat.numel(), device=device)
    _stats_add("accepted_draft_tokens", emit_accept)
    _stats_add("rejected_draft_tokens", reject_flat.numel(), device=device)

    _record_token_source_events(
        sampling_metadata,
        batch_size,
        device,
        output,
        teacher_output_override=teacher_output,
        valid_vocab_size=vocab_size,
    )
    _append_trace_events(
        sampling_metadata,
        output,
        teacher_output,
        req_ids,
        pos_in_req,
        valid_pos,
        draft_token_ids,
        target_logits,
        None,
        trigger_mask=emit_reject,
        takeover_mask_by_output=teacher_output,
        standard_target_by_output=teacher_output,
    )
    return output


def _get_paragraph_boundary_token_ids(device: torch.device) -> torch.Tensor:
    cache_key = f"{device}"
    cached = _PARAGRAPH_BOUNDARY_IDS_CACHE.get(cache_key)
    if cached is not None:
        return cached
    from transformers import AutoTokenizer
    tok_path = os.environ.get("MAESTRO_RUNTIME_TOKENIZER_PATH") or os.environ.get(
        "MAESTRO_RUNTIME_TARGET_MODEL") or os.environ.get("STUDENT_MODEL")
    ids: list[int] = []
    if tok_path:
        try:
            tok = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
            vocab_size = getattr(tok, "vocab_size", None) or len(tok)
            for tid in range(vocab_size):
                try:
                    s = tok.decode([tid], skip_special_tokens=False)
                except Exception:
                    continue
                if "\n\n" in s:
                    ids.append(tid)
        except Exception:
            logger.exception("[opd-rollout] para_break: failed to scan vocab")
    logger.warning("[opd-rollout] para_break token IDs (n=%d): %s", len(ids), ids[:30])
    global _PARAGRAPH_BOUNDARY_ID_SET
    if _PARAGRAPH_BOUNDARY_ID_SET is None:
        _PARAGRAPH_BOUNDARY_ID_SET = frozenset(int(x) for x in ids)
    tensor = torch.tensor(ids, dtype=torch.long, device=device)
    _PARAGRAPH_BOUNDARY_IDS_CACHE[cache_key] = tensor
    return tensor


def _standard_rejection_sample_output(
    draft_token_ids: torch.Tensor,
    num_draft_tokens: list[int],
    max_spec_len: int,
    cu_num_draft_tokens: torch.Tensor,
    draft_probs: torch.Tensor | None,
    target_logits: torch.Tensor,
    bonus_token_ids: torch.Tensor,
    sampling_metadata: Any,
    **kwargs: Any,
) -> torch.Tensor:
    try:
        import vllm.v1.sample.rejection_sampler as rs_mod
    except Exception:
        logger.exception("[opd-rollout] failed to import original vLLM rejection sampler")
        raise
    original = getattr(rs_mod, "_verl_opd_original_rejection_sample", None)
    if original is None:
        raise RuntimeError("[opd-rollout] original vLLM rejection_sample is not available")
    return original(
        draft_token_ids,
        num_draft_tokens,
        max_spec_len,
        cu_num_draft_tokens,
        draft_probs,
        target_logits,
        bonus_token_ids,
        sampling_metadata,
        **kwargs,
    )


def _infer_standard_target_emits(
    output: torch.Tensor,
    draft_token_ids: torch.Tensor,
    num_draft_tokens: list[int],
    max_spec_len: int,
    cu_num_draft_tokens: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Best-effort mask for recovered/bonus target tokens in standard spec."""
    num_tokens = draft_token_ids.shape[0]
    emitted = output.ge(0) & output.ne(PLACEHOLDER_TOKEN_ID)
    if num_tokens == 0:
        return emitted
    _, _, req_ids, pos_in_req, valid_pos = _request_layout(
        num_draft_tokens, num_tokens, max_spec_len, cu_num_draft_tokens, device
    )
    draft_by_pos = torch.full_like(output, PLACEHOLDER_TOKEN_ID)
    valid = valid_pos & (draft_token_ids >= 0)
    if valid.any():
        draft_by_pos[req_ids[valid], pos_in_req[valid]] = draft_token_ids[valid].to(torch.int32)
    lengths = torch.tensor(num_draft_tokens, dtype=torch.long, device=device)
    positions = torch.arange(output.shape[1], dtype=torch.long, device=device)[None, :]
    draft_positions = positions < lengths[:, None]
    bonus_positions = positions == lengths[:, None]
    return emitted & ((draft_positions & output.ne(draft_by_pos)) | bonus_positions)


def _maestro_rejection_sample(
    draft_token_ids: torch.Tensor,
    num_draft_tokens: list[int],
    max_spec_len: int,
    cu_num_draft_tokens: torch.Tensor,
    draft_probs: torch.Tensor | None,
    target_logits: torch.Tensor,
    bonus_token_ids: torch.Tensor,
    sampling_metadata: Any,
    **kwargs: Any,
) -> torch.Tensor:
    device = target_logits.device
    batch_size = len(num_draft_tokens)
    num_tokens, vocab_size = target_logits.shape

    output = torch.full(
        (batch_size, max_spec_len + 1), PLACEHOLDER_TOKEN_ID, dtype=torch.int32, device=device,
    )
    if num_tokens == 0:
        return output

    rollout_mode = os.environ.get("MAESTRO_RUNTIME_ROLLOUT_MODE", "relay").strip().lower()
    if rollout_mode not in {"relay", "trigger_stop"}:
        raise RuntimeError(
            f"Relay sampler requires MAESTRO_RUNTIME_ROLLOUT_MODE=relay or trigger_stop, got {rollout_mode!r}"
        )
    takeover_enabled = rollout_mode == "relay"
    max_takeover_tokens = int(os.environ.get("MAESTRO_MAX_TAKEOVER_TOKENS", "256"))
    paragraphs_per_takeover = int(os.environ.get("MAESTRO_PARAGRAPHS_PER_TAKEOVER", "3"))
    max_takeovers = _int("MAESTRO_MAX_TAKEOVERS", 2 if takeover_enabled else 0)
    # Optional legacy bonus: after MAX_TAKEOVERS is spent, allow extra legs for
    # a very low-divergence boundary (PDS < cutoff).
    # Default 0 = feature OFF (no behavior change for existing arms).
    low_pds_bonus = max(_int("MAESTRO_LOW_PDS_BONUS_TAKEOVER", 0), 0)
    low_pds_bonus_cutoff = _float("MAESTRO_LOW_PDS_BONUS_CUTOFF", 0.83)
    adaptive_exit_enabled = False
    exit_on_max_takeovers = "stop"
    trigger_mode = os.environ.get(
        "MAESTRO_TRIGGER_MODE", "pds_only" if _flag("MAESTRO_PDS_ENABLE", False) else "reflection_only"
    ).strip().lower()
    mechanical_mode = trigger_mode == "mechanical_only"
    mechanical_teacher_tokens = _int("MAESTRO_MECHANICAL_TEACHER_TOKENS", 256)
    mechanical_student_tokens = _int("MAESTRO_MECHANICAL_STUDENT_INTERVAL_TOKENS", -1)
    mechanical_one_shot = _flag("MAESTRO_MECHANICAL_ONE_SHOT", False)
    if mechanical_mode:
        try:
            _mechanical_phase(0, mechanical_student_tokens, mechanical_teacher_tokens, mechanical_one_shot)
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc
        if _flag("MAESTRO_PDS_ENABLE", False):
            raise RuntimeError("mechanical_only requires MAESTRO_PDS_ENABLE=0")
    takeover_cooldown_tokens = 0
    # Post-takeover tail cap (2026-09-21): student tokens allowed after the LAST
    # takeover under resume_student.  >0 enables the generation-time hard stop; <=0
    # keeps the legacy unbounded resume.  Only meaningful with resume_student.
    post_takeover_tail_max = _int("MAESTRO_POST_TAKEOVER_TAIL_MAX_TOKENS", -1)
    if takeover_enabled and max_takeovers <= 0:
        raise RuntimeError(
            "MAESTRO_MAX_TAKEOVERS must be positive when takeover is enabled"
        )
    if max_takeover_tokens <= 0:
        raise RuntimeError("MAESTRO_MAX_TAKEOVER_TOKENS must be positive")
    if paragraphs_per_takeover < 0:
        raise RuntimeError("MAESTRO_PARAGRAPHS_PER_TAKEOVER must be non-negative")
    if exit_on_max_takeovers not in {"stop", "resume_student"}:
        raise RuntimeError(
            "MAESTRO_MAX_TAKEOVERS_BEHAVIOR must be stop or resume_student, "
            f"got {exit_on_max_takeovers!r}"
        )

    reflection_token_ids = _get_reflection_token_ids(device)
    _, _, req_ids, pos_in_req, valid_pos = _request_layout(
        num_draft_tokens, num_tokens, max_spec_len, cu_num_draft_tokens, device
    )
    valid_draft = (draft_token_ids >= 0) & (draft_token_ids < vocab_size)
    if reflection_token_ids.numel() == 0 and not mechanical_mode:
        emit = valid_pos & valid_draft
        if emit.any():
            output[req_ids[emit], pos_in_req[emit]] = draft_token_ids[emit].to(torch.int32)
        _record_token_source_events(
            sampling_metadata, batch_size, device, output,
            is_teacher_per_token=torch.zeros_like(emit), req_ids=req_ids, pos_in_req=pos_in_req,
            valid_vocab_size=vocab_size,
        )
        return output

    req_id_strs = getattr(sampling_metadata, "req_ids", None)
    in_takeover_per_req = torch.zeros(batch_size, dtype=torch.bool, device=device)
    ext_rids: list[str | None] = [None] * batch_size
    if req_id_strs is not None:
        for bi in range(min(batch_size, len(req_id_strs))):
            rid = req_id_strs[bi]
            if rid is None:
                continue
            ext = _strip_vllm_suffix(rid)
            ext_rids[bi] = ext
            if takeover_enabled:
                # Continuation legs of a repeat recovery must inherit the legs the
                # rollout already spent, otherwise the paragraph channel gets a
                # fresh quota every round (see _seed_spent_takeovers).
                _seed_spent_takeovers(ext)
            if takeover_enabled and _TAKEOVER_TOKENS_REMAINING.get(ext, 0) > 0:
                in_takeover_per_req[bi] = True

    teacher_output = torch.zeros_like(output, dtype=torch.bool)
    standard_target_output = torch.zeros_like(output, dtype=torch.bool)
    emitted_takeover = torch.zeros_like(output, dtype=torch.bool)
    mechanical_teacher_per_req = torch.zeros(batch_size, dtype=torch.bool, device=device)
    mechanical_phase_remaining = torch.zeros(batch_size, dtype=torch.long, device=device)
    base_positions = None
    if mechanical_mode:
        base_positions = _get_response_base_positions(sampling_metadata, batch_size)
        if req_id_strs is not None and base_positions is None:
            raise RuntimeError("mechanical_only requires absolute response base positions")
        if base_positions is not None:
            for bi, response_pos in enumerate(base_positions):
                is_teacher, remaining = _mechanical_phase(
                    response_pos, mechanical_student_tokens, mechanical_teacher_tokens, mechanical_one_shot
                )
                mechanical_teacher_per_req[bi] = is_teacher
                mechanical_phase_remaining[bi] = remaining
            in_takeover_per_req = mechanical_teacher_per_req
            # Student chunks may cross the S->T boundary. Clip them so the next
            # sampler call starts exactly at the first teacher-owned position.
            lengths = torch.tensor(num_draft_tokens, dtype=torch.long, device=device)
            for bi in torch.nonzero(~mechanical_teacher_per_req, as_tuple=False).flatten().detach().cpu().tolist():
                keep = int(mechanical_phase_remaining[bi].item())
                if keep < int(lengths[bi].item()):
                    valid_pos &= ~((req_ids == bi) & (pos_in_req >= keep))

    if in_takeover_per_req.any():
        standard_output = _standard_rejection_sample_output(
            draft_token_ids,
            num_draft_tokens,
            max_spec_len,
            cu_num_draft_tokens,
            draft_probs,
            target_logits,
            bonus_token_ids,
            sampling_metadata,
            **kwargs,
        )
        # A bonus token is generated beyond the verified speculative prefix.
        # MAESTRO drops it so every takeover token has an unambiguous source
        # label and the next decoding step resumes from the verified prefix.
        lengths = torch.tensor(num_draft_tokens, dtype=torch.long, device=device)
        valid_bonus_pos = in_takeover_per_req & (lengths >= 0) & (lengths < standard_output.shape[1])
        if valid_bonus_pos.any():
            bonus_rows = torch.nonzero(valid_bonus_pos, as_tuple=False).flatten()
            bonus_pos = lengths[bonus_rows]
            bonus_present = (
                standard_output[bonus_rows, bonus_pos].ge(0)
                & standard_output[bonus_rows, bonus_pos].ne(PLACEHOLDER_TOKEN_ID)
            )
            if bonus_present.any():
                standard_output[bonus_rows[bonus_present], bonus_pos[bonus_present]] = PLACEHOLDER_TOKEN_ID
                _stats_add("maestro_takeover_bonus_dropped", int(bonus_present.sum().item()), device=device)
        standard_emitted = standard_output.ge(0) & standard_output.ne(PLACEHOLDER_TOKEN_ID)
        standard_target_output = _infer_standard_target_emits(
            standard_output, draft_token_ids, num_draft_tokens, max_spec_len, cu_num_draft_tokens, device
        )
        if mechanical_mode:
            for bi in torch.nonzero(mechanical_teacher_per_req, as_tuple=False).flatten().detach().cpu().tolist():
                pos = torch.nonzero(standard_emitted[bi], as_tuple=False).flatten()
                remaining = int(mechanical_phase_remaining[bi].item())
                if pos.numel() > remaining:
                    standard_output[bi, pos[remaining:]] = PLACEHOLDER_TOKEN_ID
            standard_emitted = standard_output.ge(0) & standard_output.ne(PLACEHOLDER_TOKEN_ID)
            standard_target_output &= standard_emitted
        output[in_takeover_per_req] = standard_output[in_takeover_per_req]
        emitted_takeover = standard_emitted & in_takeover_per_req[:, None]
        # The MAESTRO loss treats the whole takeover segment as teacher
        # controlled. Standard speculative decoding is only the accelerator.
        teacher_output |= emitted_takeover

    in_takeover_per_token = in_takeover_per_req[req_ids]
    teacher_argmax = target_logits.argmax(dim=-1)
    teacher_is_reflection = torch.isin(teacher_argmax, reflection_token_ids)
    raw_trigger_candidate = (
        teacher_is_reflection & teacher_argmax.ne(draft_token_ids.long())
        & valid_pos & valid_draft & ~in_takeover_per_token
    )
    trigger_topk = _int("MAESTRO_TRIGGER_TOPK", -1)
    if trigger_topk <= 0 and not mechanical_mode:
        raise RuntimeError("MAESTRO_TRIGGER_TOPK must be positive")
    if draft_probs is None:
        # vLLM's profile-run invokes the rejection sampler with synthetic
        # metadata (no req_ids) and deliberately omits draft probabilities.
        # Real patched requests always carry req_ids and must not take this path.
        if req_id_strs is None:
            return _standard_rejection_sample_output(
                draft_token_ids,
                num_draft_tokens,
                max_spec_len,
                cu_num_draft_tokens,
                draft_probs,
                target_logits,
                bonus_token_ids,
                sampling_metadata,
                **kwargs,
            )
        raise RuntimeError("MAESTRO trigger detection requires draft_probs")

    gather_ids = teacher_argmax.clamp_min(0).clamp_max(vocab_size - 1)[:, None]
    draft_probs_aligned = draft_probs[:gather_ids.shape[0]]
    k = min(trigger_topk, draft_probs_aligned.shape[-1])
    student_topk_ids = draft_probs_aligned.topk(k=k, dim=-1).indices
    teacher_in_student_topk = student_topk_ids.eq(gather_ids).any(dim=-1)
    trace_topk = int(os.environ.get("MAESTRO_RUNTIME_TRACE_TOPK", "128"))
    pds_enabled = _flag("MAESTRO_PDS_ENABLE", False)
    pds_aggregation = os.environ.get("MAESTRO_PDS_AGGREGATION", "rolling").strip().lower()
    if pds_aggregation not in {"rolling", "paragraph_mean", "paragraph_weighted", "weighted", "hybrid", "gray_fallback"}:
        raise RuntimeError(
            "MAESTRO_PDS_AGGREGATION must be rolling, paragraph_mean, paragraph_weighted, hybrid, or gray_fallback; "
            f"got {pds_aggregation!r}"
        )
    paragraph_boundary_ids_for_pds = (
        _get_paragraph_boundary_token_ids(device)
        if pds_aggregation in {"paragraph_mean", "paragraph_weighted", "weighted", "hybrid", "gray_fallback"}
        else None
    )
    pds_topk = _int("MAESTRO_PDS_TOPK", 16)
    pds_metrics_flat = (
        _compute_pds_metrics_flat(
            draft_probs=draft_probs_aligned,
            target_logits=target_logits,
            topk=pds_topk,
        )
        if pds_enabled or adaptive_exit_enabled or _flag("MAESTRO_PDS_TRACE", False)
        else None
    )
    pds_metrics_by_output = None
    if pds_metrics_flat is not None:
        pds_metrics_by_output = {
            key: _align_flat_metric_by_output(
                output=output,
                req_ids=req_ids,
                pos_in_req=pos_in_req,
                valid_pos=valid_pos,
                values=pds_metrics_flat[key],
                batch_size=batch_size,
            )
            for key in (
                "pds",
                "teacher_mass_coverage",
                "local_bhattacharyya",
                "student_p_teacher_top1",
                "teacher_student_logp_gap_top1",
                "topk_overlap",
            )
        }
    topk_overlap_by_output = (
        pds_metrics_by_output["topk_overlap"]
        if pds_metrics_by_output is not None and trace_topk == pds_topk
        else _compute_topk_overlap_by_output(
            output=output,
            req_ids=req_ids,
            pos_in_req=pos_in_req,
            valid_pos=valid_pos,
            draft_probs=draft_probs_aligned,
            target_logits=target_logits,
            batch_size=batch_size,
            topk=trace_topk,
        )
    )

    pds_trigger_candidate = torch.zeros_like(raw_trigger_candidate)
    pds_trigger_mean_by_ext: dict[str, float] = {}
    # Maps a request to the first emitted position at which its teacher-leg
    # recovery controller is ready. A value of zero means it was already ready
    # at the start of this speculative call.
    recovery_ready_from_pos: dict[int, int] = {}
    if pds_enabled and pds_metrics_flat is not None and req_id_strs is not None:
        pds_values = pds_metrics_flat["pds"]
        n_metric_rows = min(int(pds_values.shape[0]), int(num_tokens))
        if n_metric_rows > 0:
            pds_base_mask = valid_pos & valid_draft & ~in_takeover_per_token
            finite_mask = torch.isfinite(pds_values[:n_metric_rows])
            observed_mask = pds_base_mask[:n_metric_rows] & finite_mask
            _stats_add("maestro_pds_observed_tokens", observed_mask)
            if observed_mask.any():
                _stats_add("maestro_pds_score_sum", pds_values[:n_metric_rows][observed_mask].sum(), device=device)
                _stats_add("maestro_pds_score_count", observed_mask.sum(), device=device)

            threshold = _float("MAESTRO_PDS_THRESHOLD", 0.45)
            window_tokens = max(_int("MAESTRO_PDS_WINDOW_TOKENS", 100), 1)
            window_stride = max(_int("MAESTRO_PDS_WINDOW_STRIDE", window_tokens), 1)
            patience = max(_int("MAESTRO_PDS_PATIENCE", 2), 1)
            warmup_tokens = max(_int("MAESTRO_PDS_WARMUP_TOKENS", 256), 0)
            # Position guard (2026-09-19): no PDS takeover may start once a response has
            # already generated more than this many tokens.  <=0 disables.  Mirrors the
            # fault-channel guard in verl agent_loop (MAESTRO_TAKEOVER_MAX_POSITION).
            takeover_max_position = _int("MAESTRO_TAKEOVER_MAX_POSITION", -1)
            # 12k floor (2026-09-19): a request that reaches this position having NEVER
            # been taken over is force-taken-over at the FIRST \n\n boundary at/after it
            # (the boundary "nearest 12k"), so the teacher never fully retires and a
            # pure-student rollout does not run unassisted to the 16k cap.  The forced
            # leg must land BEFORE takeover_max_position (12k) so the "no takeover past
            # 12k" upper guard still holds.  <=0 disables.
            takeover_floor_start = _int("MAESTRO_TAKEOVER_FLOOR_START", -1)
            max_tracked = max(_int("MAESTRO_PDS_MAX_TRACKED_REQUESTS", 16384), batch_size)
            base_positions = _get_response_base_positions(sampling_metadata, batch_size)

            observed_cpu = observed_mask.detach().cpu()
            req_cpu = req_ids.detach().cpu()
            pos_cpu = pos_in_req.detach().cpu()
            pds_cpu = pds_values[:n_metric_rows].detach().cpu()
            triggered_exts: set[str] = set()
            for flat_idx in range(n_metric_rows):
                if not bool(observed_cpu[flat_idx]):
                    continue
                bi = int(req_cpu[flat_idx])
                if bi < 0 or bi >= len(ext_rids):
                    continue
                ext = ext_rids[bi]
                if ext is None:
                    continue
                if ext not in _PDS_WINDOW_VALUES and len(_PDS_WINDOW_VALUES) >= max_tracked:
                    drop_n = max(1, len(_PDS_WINDOW_VALUES) - max_tracked + 1)
                    for old_ext in list(_PDS_WINDOW_VALUES.keys())[:drop_n]:
                        _clear_pds_state(old_ext)

                pds_val = float(pds_cpu[flat_idx])
                if not math.isfinite(pds_val):
                    continue
                window = _PDS_WINDOW_VALUES.setdefault(ext, [])
                window.append(pds_val)
                if pds_aggregation in {"rolling", "gray_fallback"} and len(window) > window_tokens:
                    del window[:-window_tokens]
                paragraph_window = None
                if pds_aggregation == "gray_fallback":
                    paragraph_window = _PDS_PARAGRAPH_VALUES.setdefault(ext, [])
                    paragraph_window.append(pds_val)

                response_pos = int(pos_cpu[flat_idx])
                if base_positions is not None and 0 <= bi < len(base_positions):
                    response_pos += int(base_positions[bi])
                if response_pos < warmup_tokens:
                    continue
                is_boundary = bool(
                    paragraph_boundary_ids_for_pds is not None
                    and paragraph_boundary_ids_for_pds.numel() > 0
                    and int(draft_token_ids[flat_idx].item())
                    in set(paragraph_boundary_ids_for_pds.detach().cpu().tolist())
                )
                pds_gate_ok = True
                gray_fallback_hit = False
                if pds_aggregation in {"paragraph_mean", "paragraph_weighted", "weighted"}:
                    min_segment = max(_int("MAESTRO_PDS_PARAGRAPH_MIN_TOKENS", 16), 1)
                    if len(window) < min_segment or not is_boundary:
                        continue
                    pds_window_mean = _pds_segment_score(window, pds_aggregation)
                    if _flag("MAESTRO_PDS_REQUIRE_LAST16", False):
                        margin = _float("MAESTRO_PDS_LAST16_MARGIN", 0.02)
                        tail_mean = _pds_segment_score(window[-16:], "rolling")
                        pds_gate_ok = tail_mean > (threshold - margin)
                    window.clear()
                elif pds_aggregation == "hybrid" and is_boundary and len(window) >= max(_int("MAESTRO_PDS_PARAGRAPH_MIN_TOKENS", 16), 1):
                    pds_window_mean = _pds_segment_score(window, "paragraph_weighted")
                    if _flag("MAESTRO_PDS_REQUIRE_LAST16", False):
                        margin = _float("MAESTRO_PDS_LAST16_MARGIN", 0.02)
                        tail_mean = _pds_segment_score(window[-16:], "rolling")
                        pds_gate_ok = tail_mean > (threshold - margin)
                    window.clear()
                elif pds_aggregation == "gray_fallback":
                    rolling_due = (
                        len(window) >= window_tokens
                        and ((response_pos + 1 - warmup_tokens) % window_stride) == 0
                    )
                    if rolling_due:
                        rolling_mean_now = float(sum(window) / len(window))
                        _PDS_LAST_ROLLING_MEAN[ext] = rolling_mean_now
                        direct_streak = (
                            _PDS_BAD_WINDOW_STREAK.get(ext, 0) + 1
                            if rolling_mean_now >= threshold else 0
                        )
                        _PDS_BAD_WINDOW_STREAK[ext] = direct_streak
                    else:
                        rolling_mean_now = float("nan")
                        direct_streak = _PDS_BAD_WINDOW_STREAK.get(ext, 0)
                    rolling_mean = _PDS_LAST_ROLLING_MEAN.get(ext, float("nan"))
                    min_segment = max(_int("MAESTRO_PDS_PARAGRAPH_MIN_TOKENS", 16), 1)
                    paragraph_ready = bool(is_boundary and paragraph_window is not None and len(paragraph_window) >= min_segment)
                    paragraph_score = (
                        _pds_segment_score(paragraph_window, "paragraph_weighted")
                        if paragraph_ready else float("nan")
                    )
                    tail_mean = (
                        _pds_segment_score(paragraph_window[-16:], "rolling")
                        if paragraph_ready else float("nan")
                    )
                    if paragraph_window is not None and is_boundary:
                        paragraph_window.clear()
                    gray_upper = _float("MAESTRO_PDS_GRAY_UPPER", 0.94)
                    paragraph_threshold = _float("MAESTRO_PDS_PARAGRAPH_THRESHOLD", 0.88)
                    min_gap = _float("MAESTRO_PDS_PARAGRAPH_MIN_GAP", 0.05)
                    last16_threshold = _float("MAESTRO_PDS_PARAGRAPH_LAST16_THRESHOLD", 0.91)
                    gray_fallback_hit = bool(
                        paragraph_ready
                        and math.isfinite(rolling_mean)
                        and gray_upper <= rolling_mean < threshold
                        and paragraph_score > paragraph_threshold
                        and (paragraph_score - rolling_mean) > min_gap
                        and tail_mean > last16_threshold
                    )
                    fallback_patience = max(_int("MAESTRO_PDS_FALLBACK_PATIENCE", 2), 1)
                    if paragraph_ready:
                        fallback_streak = (
                            _PDS_FALLBACK_STREAK.get(ext, 0) + 1
                            if gray_fallback_hit else 0
                        )
                        _PDS_FALLBACK_STREAK[ext] = fallback_streak
                    else:
                        fallback_streak = _PDS_FALLBACK_STREAK.get(ext, 0)
                    direct_trigger = bool(rolling_due and direct_streak >= patience)
                    fallback_trigger = bool(paragraph_ready and fallback_streak >= fallback_patience)
                    if not direct_trigger and not fallback_trigger:
                        continue
                    pds_window_mean = paragraph_score if fallback_trigger and not direct_trigger else rolling_mean_now
                    if ext not in triggered_exts:
                        pds_trigger_candidate[flat_idx] = True
                        triggered_exts.add(ext)
                        pds_trigger_mean_by_ext[ext] = pds_window_mean
                        _stats_add("maestro_pds_window_mean_at_trigger_sum", pds_window_mean, device=device)
                        _stats_add("maestro_pds_window_mean_at_trigger_count", 1, device=device)
                    continue
                else:
                    if len(window) < window_tokens:
                        continue
                    if ((response_pos + 1 - warmup_tokens) % window_stride) != 0:
                        continue
                    pds_window_mean = float(sum(window) / len(window))
                if not pds_gate_ok:
                    streak = 0
                    _PDS_BAD_WINDOW_STREAK[ext] = 0
                    continue
                # 12k floor: a boundary at/after takeover_floor_start on a request that
                # has never been taken over -> force this one leg immediately (ignore the
                # threshold/patience streak), provided it is still <= the 12k upper guard.
                floor_force = (
                    takeover_floor_start > 0
                    and is_boundary
                    and response_pos >= takeover_floor_start
                    and (takeover_max_position <= 0 or response_pos <= takeover_max_position)
                    and _COMPLETED_TAKEOVERS.get(ext, 0) == 0
                    and ext not in triggered_exts
                )
                if floor_force:
                    pds_trigger_candidate[flat_idx] = True
                    triggered_exts.add(ext)
                    pds_trigger_mean_by_ext[ext] = pds_window_mean
                    _stats_add("maestro_pds_trigger_floor_forced", 1, device=device)
                    _stats_add("maestro_pds_window_mean_at_trigger_sum", pds_window_mean, device=device)
                    _stats_add("maestro_pds_window_mean_at_trigger_count", 1, device=device)
                    continue
                if pds_window_mean >= threshold or gray_fallback_hit:
                    streak = _PDS_BAD_WINDOW_STREAK.get(ext, 0) + 1
                else:
                    streak = 0
                _PDS_BAD_WINDOW_STREAK[ext] = streak
                if streak >= patience and ext not in triggered_exts:
                    if takeover_max_position > 0 and response_pos > takeover_max_position:
                        # Would-be trigger past the position guard: block it and count
                        # how often the guard fires (the "禁止接管触发多了" signal).
                        _stats_add("maestro_pds_trigger_blocked_maxpos", 1, device=device)
                        continue
                    pds_trigger_candidate[flat_idx] = True
                    triggered_exts.add(ext)
                    pds_trigger_mean_by_ext[ext] = pds_window_mean
                    _stats_add("maestro_pds_window_mean_at_trigger_sum", pds_window_mean, device=device)
                    _stats_add("maestro_pds_window_mean_at_trigger_count", 1, device=device)
            _stats_add("maestro_pds_trigger_candidates", pds_trigger_candidate)
    if adaptive_exit_enabled and pds_metrics_flat is not None and req_id_strs is not None:
        pds_values = pds_metrics_flat["pds"]
        n_metric_rows = min(int(pds_values.shape[0]), int(num_tokens))
        if n_metric_rows > 0:
            exit_window_stride = max(
            )
            flat_req_ids = req_ids[:n_metric_rows]
            flat_positions = pos_in_req[:n_metric_rows]
            layout_valid = (
                valid_pos[:n_metric_rows]
                & (flat_req_ids >= 0)
                & (flat_req_ids < batch_size)
                & (flat_positions >= 0)
                & (flat_positions < output.shape[1])
            )
            emitted_teacher_flat = torch.zeros(n_metric_rows, dtype=torch.bool, device=device)
            if layout_valid.any():
                emitted_teacher_flat[layout_valid] = emitted_takeover[
                    flat_req_ids[layout_valid], flat_positions[layout_valid]
                ]
            recovery_mask = layout_valid & in_takeover_per_token[:n_metric_rows] & emitted_teacher_flat
            recovery_mask &= torch.isfinite(pds_values[:n_metric_rows])
            _stats_add("maestro_exit_pds_observed_tokens", recovery_mask)
            req_cpu = flat_req_ids.detach().cpu()
            pos_cpu = flat_positions.detach().cpu()
            pds_cpu = pds_values[:n_metric_rows].detach().cpu()
            recovery_cpu = recovery_mask.detach().cpu()
            for flat_idx in range(n_metric_rows):
                if not bool(recovery_cpu[flat_idx]):
                    continue
                bi = int(req_cpu[flat_idx])
                if bi < 0 or bi >= len(ext_rids):
                    continue
                ext = ext_rids[bi]
                if ext is None:
                    continue
                became_ready, window_mean = _observe_takeover_recovery(
                    ext,
                    float(pds_cpu[flat_idx]),
                    window_tokens=exit_window_tokens,
                    window_stride=exit_window_stride,
                    threshold=exit_threshold,
                    patience=exit_patience,
                )
                if window_mean is not None:
                    _stats_add("maestro_exit_pds_window_mean_sum", window_mean, device=device)
                    _stats_add("maestro_exit_pds_window_mean_count", 1, device=device)
                    _stats_add(
                        "maestro_exit_pds_good_windows",
                        int(window_mean <= exit_threshold),
                        device=device,
                    )
                if became_ready:
                    recovery_ready_from_pos[bi] = int(pos_cpu[flat_idx])
                    _stats_add("maestro_exit_recovery_ready", 1, device=device)
                elif _TAKEOVER_RECOVERY_READY.get(ext, False):
                    recovery_ready_from_pos.setdefault(bi, 0)
                else:
                    recovery_ready_from_pos.pop(bi, None)
    reflection_stop_candidate = raw_trigger_candidate & ~teacher_in_student_topk
    if trigger_mode in {"pds", "pds_only"}:
        stop_candidate = pds_trigger_candidate
    elif trigger_mode in {"reflection", "reflection_only"}:
        stop_candidate = reflection_stop_candidate
    elif trigger_mode in {"or", "combined", "reflection_or_pds", "pds_or_reflection"}:
        stop_candidate = reflection_stop_candidate | pds_trigger_candidate
    elif trigger_mode == "mechanical_only":
        stop_candidate = torch.zeros_like(reflection_stop_candidate)
    else:
        raise RuntimeError(
            "MAESTRO_TRIGGER_MODE must be one of pds_only, reflection_only, "
            f"reflection_or_pds, or mechanical_only; got {trigger_mode!r}"
        )
    trigger_topk_threshold = torch.where(
        stop_candidate,
        torch.full((num_tokens,), int(trigger_topk), dtype=torch.int32, device=device),
        torch.zeros((num_tokens,), dtype=torch.int32, device=device),
    )
    _stats_add("maestro_reflection_candidates", raw_trigger_candidate)
    _stats_add("maestro_divergence_triggers", reflection_stop_candidate)
    _stats_add("maestro_combined_divergence_triggers", stop_candidate)
    _stats_add("maestro_active_reflection_triggers", stop_candidate & reflection_stop_candidate)
    _stats_add("maestro_active_pds_triggers", stop_candidate & pds_trigger_candidate)
    _stats_add("maestro_pds_only_triggers", pds_trigger_candidate & ~reflection_stop_candidate)
    _stats_add("maestro_pds_reflection_overlap_triggers", pds_trigger_candidate & reflection_stop_candidate)
    _stats_add("maestro_teacher_argmax_in_student_topk", raw_trigger_candidate & teacher_in_student_topk)

    maestro_trigger_candidate = torch.zeros_like(stop_candidate)
    if takeover_enabled and stop_candidate.any():
        maestro_trigger_candidate = stop_candidate.clone()
        if adaptive_exit_enabled or exit_on_max_takeovers == "resume_student":
            suppressed = torch.zeros_like(maestro_trigger_candidate)
            for flat_idx in torch.nonzero(maestro_trigger_candidate, as_tuple=False).flatten().detach().cpu().tolist():
                bi = int(req_ids[flat_idx].detach().cpu().item())
                ext = ext_rids[bi] if 0 <= bi < len(ext_rids) else None
                if ext is None:
                    continue
                max_takeovers_req = _TAKEOVER_MAX_TAKEOVERS.get(ext, max_takeovers)
                exhausted = _COMPLETED_TAKEOVERS.get(ext, 0) >= max_takeovers_req
                cooling_down = _TAKEOVER_COOLDOWN_REMAINING.get(ext, 0) > 0
                # low-PDS bonus: quota spent but this boundary is deeply low-confidence
                # -> grant one extra leg (once per rollout), lifting the effective quota.
                # Judged on the SAME paragraph_weighted aggregated score used for entry
                # (pds_trigger_mean_by_ext), not the raw per-token PDS.
                if (
                    exhausted
                    and not cooling_down
                    and low_pds_bonus > 0
                    and _LOW_PDS_BONUS_USED.get(ext, 0) < low_pds_bonus
                    and ext in pds_trigger_mean_by_ext
                ):
                    pds_here = float(pds_trigger_mean_by_ext[ext])
                    if math.isfinite(pds_here) and pds_here < low_pds_bonus_cutoff:
                        _LOW_PDS_BONUS_USED[ext] = _LOW_PDS_BONUS_USED.get(ext, 0) + 1
                        _TAKEOVER_MAX_TAKEOVERS[ext] = max_takeovers_req + 1
                        _stats_add("maestro_low_pds_bonus_takeovers", 1, device=device)
                        exhausted = False
                if exhausted or cooling_down:
                    suppressed[flat_idx] = True
            if suppressed.any():
                _stats_add("maestro_takeover_trigger_suppressed_exhausted", suppressed)
                maestro_trigger_candidate &= ~suppressed
        stop_candidate = torch.zeros_like(stop_candidate)
        _stats_add("maestro_takeover_triggers", maestro_trigger_candidate)

    trigger_candidate = (
        maestro_trigger_candidate
        if takeover_enabled
        else torch.zeros_like(raw_trigger_candidate)
    )

    mechanical_start = torch.zeros_like(trigger_candidate)
    if mechanical_mode and takeover_enabled and (
        mechanical_student_tokens >= 0
        or os.environ.get("MAESTRO_MECHANICAL_TAKEOVER_POSITIONS", "").strip()
    ):
        assert base_positions is not None
        for flat_idx in range(num_tokens):
            if not bool((valid_pos[flat_idx] & valid_draft[flat_idx] & ~in_takeover_per_token[flat_idx]).item()):
                continue
            bi = int(req_ids[flat_idx].item())
            response_pos = int(base_positions[bi]) + int(pos_in_req[flat_idx].item())
            is_teacher, _ = _mechanical_phase(
                response_pos, mechanical_student_tokens, mechanical_teacher_tokens, mechanical_one_shot
            )
            if is_teacher:
                mechanical_start[flat_idx] = True
        if mechanical_start.any():
            first_by_req: set[int] = set()
            for flat_idx in torch.nonzero(mechanical_start, as_tuple=False).flatten().detach().cpu().tolist():
                bi = int(req_ids[flat_idx].item())
                if bi in first_by_req:
                    mechanical_start[flat_idx] = False
                else:
                    first_by_req.add(bi)
            trigger_candidate = mechanical_start
        else:
            trigger_candidate = mechanical_start

    allowed_trigger = trigger_candidate
    allowed_stop = stop_candidate
    INF_POS = max_spec_len + 2
    first_event_pos = torch.full((batch_size,), INF_POS, dtype=torch.long, device=device)
    event_candidate = trigger_candidate | allowed_stop
    if event_candidate.any():
        masked_pos = torch.where(event_candidate, pos_in_req, torch.full_like(pos_in_req, INF_POS))
        first_event_pos.scatter_reduce_(0, req_ids, masked_pos, reduce="amin", include_self=True)

    first_event_for_token = first_event_pos[req_ids]
    emit_student = valid_pos & valid_draft & ~in_takeover_per_token & (pos_in_req < first_event_for_token)
    emit_trigger = allowed_trigger & (pos_in_req == first_event_for_token)
    emit_stop = allowed_stop & (pos_in_req == first_event_for_token)

    trigger_pds_by_ext = dict(pds_trigger_mean_by_ext)
    if emit_trigger.any() and pds_metrics_flat is not None:
        pds_cpu = pds_metrics_flat["pds"].detach().cpu()
        for idx in torch.nonzero(emit_trigger, as_tuple=False).flatten().detach().cpu().tolist():
            bi = int(req_ids[idx].detach().cpu().item())
            ext = ext_rids[bi] if 0 <= bi < len(ext_rids) else None
            if ext is not None and ext not in trigger_pds_by_ext and idx < len(pds_cpu):
                value = float(pds_cpu[idx])
                if math.isfinite(value):
                    trigger_pds_by_ext[ext] = value

    if emit_student.any():
        output[req_ids[emit_student], pos_in_req[emit_student]] = draft_token_ids[emit_student].to(torch.int32)
    if emit_trigger.any():
        output[req_ids[emit_trigger], pos_in_req[emit_trigger]] = teacher_argmax[emit_trigger].to(torch.int32)
        teacher_output[req_ids[emit_trigger], pos_in_req[emit_trigger]] = True
        for bi in req_ids[emit_trigger].detach().cpu().tolist():
            ext = ext_rids[int(bi)] if 0 <= int(bi) < len(ext_rids) else None
            if mechanical_mode:
                if ext is not None:
                    _TAKEOVER_PARAGRAPH_LIMIT[ext] = 2**31 - 1
                    # The trigger token is already teacher-owned; continuation owns T-1.
                    _TAKEOVER_TOKEN_LIMIT[ext] = mechanical_teacher_tokens - 1
                    _TAKEOVER_MAX_TAKEOVERS[ext] = 2**31 - 1
                continue
            if ext is not None:
                budget_paragraphs, budget_tokens, budget_max_takeovers = _dynamic_takeover_budget(
                    trigger_pds_by_ext.get(ext), paragraphs_per_takeover, max_takeover_tokens, max_takeovers
                )
                _TAKEOVER_PARAGRAPH_LIMIT[ext] = budget_paragraphs
                _TAKEOVER_TOKEN_LIMIT[ext] = budget_tokens
                _TAKEOVER_MAX_TAKEOVERS[ext] = budget_max_takeovers
                _stats_add("maestro_selected_paragraph_budget_sum", budget_paragraphs, device=device)
                _stats_add("maestro_selected_token_budget_sum", budget_tokens, device=device)
                _stats_add("maestro_selected_max_takeovers_sum", budget_max_takeovers, device=device)
                if budget_max_takeovers > max_takeovers:
                    _stats_add("maestro_dynamic_hard_budget_triggers", 1, device=device)
            _clear_pds_state(ext)
            _clear_takeover_recovery_state(ext)
            if ext is not None:
                _TAKEOVER_COOLDOWN_REMAINING.pop(ext, None)
    internal_stop_output = torch.zeros_like(output, dtype=torch.bool)
    stop_topk_threshold_by_output = None
    if emit_stop.any():
        stop_id = _get_internal_stop_token_id(vocab_size)
        if stop_id is None:
            _stats_add("trigger_stop_missing_internal_stop_id", emit_stop)
            emit_stop = torch.zeros_like(emit_stop)
        else:
            stop_indices = torch.nonzero(emit_stop, as_tuple=False).flatten()
            event_req_ids = req_ids[stop_indices]
            stopped_token_pos = pos_in_req[stop_indices]
            # Keep the divergent student action in the training response, then
            # emit one private stop token that rollout strips before loss.
            output[event_req_ids, stopped_token_pos] = draft_token_ids[stop_indices].to(torch.int32)
            internal_stop_pos = stopped_token_pos + 1
            can_place_stop = internal_stop_pos < output.shape[1]
            if can_place_stop.any():
                stop_req_ids = event_req_ids[can_place_stop]
                stop_pos = internal_stop_pos[can_place_stop]
                output[stop_req_ids, stop_pos] = int(stop_id)
                internal_stop_output[stop_req_ids, stop_pos] = True
            if (~can_place_stop).any():
                # This should not happen because output has max_spec_len + 1
                # slots, but fall back to the old behavior to force a stop.
                bad_req_ids = event_req_ids[~can_place_stop]
                bad_pos = stopped_token_pos[~can_place_stop]
                output[bad_req_ids, bad_pos] = int(stop_id)
                internal_stop_output[bad_req_ids, bad_pos] = True
                _stats_add("trigger_stop_no_extra_slot", int((~can_place_stop).sum().item()), device=device)

    if internal_stop_output.any():
        stop_topk_threshold_by_output = torch.zeros(output.shape, dtype=torch.int32, device=device)
        stop_indices = torch.nonzero(emit_stop, as_tuple=False).flatten()
        stop_req_ids = req_ids[stop_indices]
        internal_stop_pos = pos_in_req[stop_indices] + 1
        can_place_stop = internal_stop_pos < output.shape[1]
        if can_place_stop.any():
            stop_topk_threshold_by_output[
                stop_req_ids[can_place_stop], internal_stop_pos[can_place_stop]
            ] = trigger_topk_threshold[stop_indices][can_place_stop]
        if (~can_place_stop).any():
            bad_pos = pos_in_req[stop_indices][~can_place_stop]
            bad_req_ids = stop_req_ids[~can_place_stop]
            stop_topk_threshold_by_output[bad_req_ids, bad_pos] = (
                trigger_topk_threshold[stop_indices][~can_place_stop]
            )

    paragraph_boundary_ids = (
        torch.empty(0, dtype=torch.long, device=device)
        if mechanical_mode
        else _get_paragraph_boundary_token_ids(device)
    )
    new_trigger_per_req = torch.zeros(batch_size, dtype=torch.bool, device=device)
    trigger_is_boundary_per_req = torch.zeros(batch_size, dtype=torch.bool, device=device)
    if emit_trigger.any():
        new_trigger_per_req[req_ids[emit_trigger]] = True
        if paragraph_boundary_ids.numel() > 0:
            trigger_hit_para = emit_trigger & torch.isin(teacher_argmax, paragraph_boundary_ids)
            if trigger_hit_para.any():
                trigger_is_boundary_per_req[req_ids[trigger_hit_para]] = True

    takeover_terminated = torch.zeros(batch_size, dtype=torch.bool, device=device)
    adaptive_exit_terminated = torch.zeros(batch_size, dtype=torch.bool, device=device)
    token_cap_exit_terminated = torch.zeros(batch_size, dtype=torch.bool, device=device)
    paragraph_cap_exit_terminated = torch.zeros(batch_size, dtype=torch.bool, device=device)
    stopped_after_takeover = torch.zeros(batch_size, dtype=torch.bool, device=device)
    internal_stop_id = (
        _get_internal_stop_token_id(vocab_size)
        if max_takeovers > 0
        else None
    )
    if paragraphs_per_takeover <= 0 and emit_trigger.any() and max_takeovers > 0:
        # Trigger-only ablation: keep the teacher trigger token in the loss, but
        # do not continue with a teacher takeover segment. Count the trigger as
        # a completed takeover so stop1/stop2/... semantics remain comparable.
        for idx in torch.nonzero(emit_trigger, as_tuple=False).flatten().detach().cpu().tolist():
            bi = int(req_ids[idx].detach().cpu().item())
            ext = ext_rids[bi] if 0 <= bi < len(ext_rids) else None
            if ext is None:
                continue
            end_count = _COMPLETED_TAKEOVERS.get(ext, 0) + 1
            max_takeovers_req = _TAKEOVER_MAX_TAKEOVERS.get(ext, max_takeovers)
            if end_count >= max_takeovers_req:
                _clear_pds_state(ext)
                _clear_takeover_recovery_state(ext)
                if exit_on_max_takeovers == "resume_student":
                    _COMPLETED_TAKEOVERS[ext] = end_count
                    if takeover_cooldown_tokens:
                        _TAKEOVER_COOLDOWN_REMAINING[ext] = takeover_cooldown_tokens
                    _stats_add("maestro_resumed_after_max_takeovers", 1, device=device)
                    if post_takeover_tail_max > 0:
                        _POST_TAKEOVER_TAIL_REMAINING[ext] = post_takeover_tail_max
                else:
                    _COMPLETED_TAKEOVERS.pop(ext, None)
                    if internal_stop_id is None:
                        _stats_add("maestro_stop_missing_internal_stop_id", 1, device=device)
                        continue
                    stop_pos = int(pos_in_req[idx].detach().cpu().item()) + 1
                    if stop_pos >= output.shape[1]:
                        stop_pos = int(pos_in_req[idx].detach().cpu().item())
                        _stats_add("maestro_stop_no_extra_slot", 1, device=device)
                    output[bi, stop_pos] = int(internal_stop_id)
                    teacher_output[bi, stop_pos] = False
                    emitted_takeover[bi, stop_pos] = False
                    standard_target_output[bi, stop_pos] = False
                    internal_stop_output[bi, stop_pos] = True
                    stopped_after_takeover[bi] = True
            else:
                _COMPLETED_TAKEOVERS[ext] = end_count

    emitted = output.ge(0) & output.ne(PLACEHOLDER_TOKEN_ID)
    if paragraph_boundary_ids.numel() > 0:
        takeover_boundary_hits = emitted_takeover & torch.isin(output.long(), paragraph_boundary_ids)
    else:
        takeover_boundary_hits = torch.zeros_like(emitted_takeover)
    if emitted_takeover.any():
        for bi in torch.nonzero(in_takeover_per_req, as_tuple=False).flatten().detach().cpu().tolist():
            ext = ext_rids[bi]
            if ext is None:
                continue
            exit_min_takeover_tokens = _exit_min_takeover_tokens(ext, exit_min_takeover_default)
            pos = torch.nonzero(emitted_takeover[bi], as_tuple=False).flatten()
            if pos.numel() == 0:
                continue
            cap_cur = _TAKEOVER_TOKENS_REMAINING.get(ext, 0)
            paragraphs_cur = _TAKEOVER_PARAGRAPHS_REMAINING.get(ext, 0)
            if cap_cur <= 0 or (paragraphs_cur <= 0 and not mechanical_mode):
                continue

            keep_until_pos: int | None = None
            exit_reason: str | None = None
            para_seen = 0
            pos_cpu = pos.detach().cpu().tolist()
            para_cpu = takeover_boundary_hits[bi, pos].detach().cpu().tolist()
            ready_from = recovery_ready_from_pos.get(bi)
            if ready_from is None and adaptive_exit_enabled and _TAKEOVER_RECOVERY_READY.get(ext, False):
                ready_from = 0
            prior_takeover_tokens = max(max_takeover_tokens - cap_cur, 0)
            for emitted_idx, (out_pos, is_para) in enumerate(zip(pos_cpu, para_cpu), start=1):
                if bool(is_para):
                    para_seen += 1
                recovery_exit = (
                    adaptive_exit_enabled
                    and ready_from is not None
                    and int(out_pos) >= ready_from
                    and prior_takeover_tokens + emitted_idx >= exit_min_takeover_tokens
                    and (bool(is_para) or not exit_require_boundary)
                )
                if recovery_exit:
                    keep_until_pos = int(out_pos)
                    exit_reason = "recovery"
                    break
                if emitted_idx >= cap_cur:
                    keep_until_pos = int(out_pos)
                    exit_reason = "token_cap"
                    break
                if not mechanical_mode and paragraphs_cur - para_seen <= 0:
                    keep_until_pos = int(out_pos)
                    exit_reason = "paragraph_cap"
                    break

            if keep_until_pos is None:
                continue
            drop_pos = pos[pos > keep_until_pos]
            if drop_pos.numel() > 0:
                output[bi, drop_pos] = PLACEHOLDER_TOKEN_ID
                teacher_output[bi, drop_pos] = False
                emitted_takeover[bi, drop_pos] = False
                takeover_boundary_hits[bi, drop_pos] = False
                standard_target_output[bi, drop_pos] = False
            takeover_terminated[bi] = True
            if exit_reason == "recovery":
                adaptive_exit_terminated[bi] = True
                _stats_add("maestro_exit_teacher_tokens_sum", prior_takeover_tokens + emitted_idx, device=device)
                _stats_add("maestro_exit_teacher_tokens_count", 1, device=device)
                _stats_add("maestro_exit_min_takeover_tokens_sum", exit_min_takeover_tokens, device=device)
                _stats_add("maestro_exit_min_takeover_tokens_count", 1, device=device)
            elif exit_reason == "token_cap":
                token_cap_exit_terminated[bi] = True
            elif exit_reason == "paragraph_cap":
                paragraph_cap_exit_terminated[bi] = True
            _clear_takeover_recovery_state(ext)
            if mechanical_mode:
                end_count = _COMPLETED_TAKEOVERS.get(ext, 0) + 1
                _COMPLETED_TAKEOVERS[ext] = end_count
                _stats_add("maestro_mechanical_legs_completed", 1, device=device)
                if max_takeovers > 0 and end_count >= _TAKEOVER_MAX_TAKEOVERS.get(ext, max_takeovers):
                    if exit_on_max_takeovers == "resume_student" and post_takeover_tail_max > 0:
                        _POST_TAKEOVER_TAIL_REMAINING[ext] = post_takeover_tail_max
                continue
            if adaptive_exit_enabled and takeover_cooldown_tokens:
                _TAKEOVER_COOLDOWN_REMAINING[ext] = takeover_cooldown_tokens
            if max_takeovers > 0:
                max_takeovers_req = _TAKEOVER_MAX_TAKEOVERS.get(ext, max_takeovers)
                end_count = _COMPLETED_TAKEOVERS.get(ext, 0) + 1
                if end_count >= max_takeovers_req:
                    _clear_pds_state(ext)
                    if exit_on_max_takeovers == "resume_student":
                        _COMPLETED_TAKEOVERS[ext] = end_count
                        _stats_add("maestro_resumed_after_max_takeovers", 1, device=device)
                        if post_takeover_tail_max > 0:
                            _POST_TAKEOVER_TAIL_REMAINING[ext] = post_takeover_tail_max
                    else:
                        _COMPLETED_TAKEOVERS.pop(ext, None)
                        if internal_stop_id is None:
                            _stats_add("maestro_stop_missing_internal_stop_id", 1, device=device)
                        else:
                            stop_pos = int(keep_until_pos) + 1
                            if stop_pos >= output.shape[1]:
                                # The fast path normally has an extra non-loss slot.
                                # If a future vLLM shape lacks it, force termination
                                # by sacrificing the final emitted takeover token.
                                stop_pos = int(keep_until_pos)
                                _stats_add("maestro_stop_no_extra_slot", 1, device=device)
                            output[bi, stop_pos] = int(internal_stop_id)
                            teacher_output[bi, stop_pos] = False
                            emitted_takeover[bi, stop_pos] = False
                            takeover_boundary_hits[bi, stop_pos] = False
                            standard_target_output[bi, stop_pos] = False
                            internal_stop_output[bi, stop_pos] = True
                            stopped_after_takeover[bi] = True
                else:
                    _COMPLETED_TAKEOVERS[ext] = end_count

    new_trigger_list = new_trigger_per_req.detach().cpu().tolist()
    trigger_is_boundary_list = trigger_is_boundary_per_req.detach().cpu().tolist()
    takeover_list = in_takeover_per_req.detach().cpu().tolist()
    takeover_terminated_list = takeover_terminated.detach().cpu().tolist()
    takeover_emit_counts = emitted_takeover.sum(dim=1).detach().cpu().tolist()
    takeover_boundary_counts = takeover_boundary_hits.sum(dim=1).detach().cpu().tolist()
    for bi in range(batch_size):
        ext = ext_rids[bi]
        if ext is None:
            continue
        if new_trigger_list[bi]:
            paragraphs_limit = _TAKEOVER_PARAGRAPH_LIMIT.get(ext, paragraphs_per_takeover)
            token_limit = _TAKEOVER_TOKEN_LIMIT.get(ext, max_takeover_tokens)
            paragraphs_left = (
                paragraphs_limit
                if mechanical_mode
                else paragraphs_limit - (1 if trigger_is_boundary_list[bi] else 0)
            )
            if paragraphs_left <= 0:
                _TAKEOVER_TOKENS_REMAINING.pop(ext, None)
                _TAKEOVER_PARAGRAPHS_REMAINING.pop(ext, None)
            else:
                _TAKEOVER_TOKENS_REMAINING[ext] = token_limit
                _TAKEOVER_PARAGRAPHS_REMAINING[ext] = paragraphs_left
        elif takeover_list[bi] and takeover_emit_counts[bi] > 0:
            cap_cur = _TAKEOVER_TOKENS_REMAINING.get(ext, 0)
            paragraphs_cur = _TAKEOVER_PARAGRAPHS_REMAINING.get(ext, 0)
            if not mechanical_mode:
                paragraphs_cur -= int(takeover_boundary_counts[bi])
            cap_cur -= int(takeover_emit_counts[bi])
            if takeover_terminated_list[bi] or paragraphs_cur <= 0 or cap_cur <= 0:
                _TAKEOVER_TOKENS_REMAINING.pop(ext, None)
                _TAKEOVER_PARAGRAPHS_REMAINING.pop(ext, None)
            else:
                _TAKEOVER_TOKENS_REMAINING[ext] = cap_cur
                _TAKEOVER_PARAGRAPHS_REMAINING[ext] = paragraphs_cur

    # ---- repetition early stop ------------------------------------------------
    # A confident loop never fires the PDS trigger (student and teacher agree,
    # so PDS stays ~1), which is why the student used to decode the whole 16384
    # budget before anything reacted.  End the request here instead, at the loop
    # onset, and let the agent loop rewind a few dozen tokens.  The rewind,
    # teacher leg and takeover ledger are untouched: this only stops generation.
    repeat_early_cfg = _repeat_early_stop_config()
    repeat_stop_output: torch.Tensor | None = None
    if repeat_early_cfg["enabled"]:
        repeat_stop_output = _apply_repeat_early_stop(
            output,
            internal_stop_output,
            emitted,
            ext_rids,
            in_takeover_per_req,
            vocab_size,
            device,
            repeat_early_cfg,
        )

    # ---- post-takeover tail cap ----------------------------------------------
    # Runs AFTER the repeat watchdog so a repeat/max-takeover stop this step wins:
    # such a request has an internal stop already, so the cap drops its deadline
    # instead of double-stopping.  Counts only the student tokens emitted after the
    # last takeover; injects the hard stop once tail_max is spent.
    if post_takeover_tail_max > 0:
        _apply_post_takeover_tail_cap(
            output,
            internal_stop_output,
            teacher_output,
            ext_rids,
            vocab_size,
            device,
        )

    _stats_add("calls", 1, device=device)
    _stats_add("requests", batch_size, device=device)
    _stats_add("drafted_tokens", valid_pos & (draft_token_ids >= 0))
    final_emitted = emitted & ~internal_stop_output
    _stats_add("emitted_tokens", final_emitted)
    _stats_add("student_tokens", final_emitted & ~teacher_output)
    _stats_add("teacher_tokens", teacher_output)
    if mechanical_mode:
        mechanical_trigger_count = int(emit_trigger.sum().item())
        if mechanical_trigger_count:
            _stats_add("maestro_mechanical_legs_started", mechanical_trigger_count, device=device)
        _stats_add("maestro_mechanical_student_tokens", final_emitted & ~teacher_output)
        _stats_add("maestro_mechanical_teacher_tokens", teacher_output)
    _stats_add("maestro_new_triggers", emit_trigger)
    _stats_add("trigger_stop_events", emit_stop)
    _stats_add("maestro_takeover_tokens", emitted_takeover)
    _stats_add("maestro_standard_target_tokens", standard_target_output & in_takeover_per_req[:, None])
    _stats_add("maestro_paragraph_boundaries", takeover_boundary_hits)
    _stats_add("maestro_takeovers_completed", takeover_terminated)
    _stats_add("maestro_adaptive_recovery_exits", adaptive_exit_terminated)
    _stats_add("maestro_token_cap_exits", token_cap_exit_terminated)
    _stats_add("maestro_paragraph_cap_exits", paragraph_cap_exit_terminated)
    _stats_add("maestro_stopped_after_max_takeovers", stopped_after_takeover)
    student_emitted = final_emitted & ~teacher_output
    for bi in range(batch_size):
        ext = ext_rids[bi]
        if ext is None:
            continue
        cooldown = _TAKEOVER_COOLDOWN_REMAINING.get(ext, 0)
        if cooldown <= 0:
            continue
        remaining = cooldown - int(student_emitted[bi].sum().item())
        if remaining > 0:
            _TAKEOVER_COOLDOWN_REMAINING[ext] = remaining
        else:
            _TAKEOVER_COOLDOWN_REMAINING.pop(ext, None)
    action_token_by_output = None
    teacher_action_logp_by_output = None
    if _flag("MAESTRO_EXPORT_STUDENT_ACTION", False):
        action_token_by_output = output.clone()
        teacher_action_logp_by_output = torch.full(output.shape, float("nan"), dtype=torch.float32, device=device)
        valid_action = (
            valid_pos
            & valid_draft
            & (req_ids >= 0)
            & (req_ids < batch_size)
            & (pos_in_req >= 0)
            & (pos_in_req < output.shape[1])
        )
        if valid_action.any():
            flat_req = req_ids[valid_action]
            flat_pos = pos_in_req[valid_action]
            teacher_flat = teacher_output[flat_req, flat_pos]
            if teacher_flat.any():
                action_indices = torch.nonzero(valid_action, as_tuple=False).flatten()[teacher_flat]
                action_req = req_ids[action_indices]
                action_pos = pos_in_req[action_indices]
                safe_draft = draft_token_ids[action_indices].long().clamp_min(0).clamp_max(vocab_size - 1)
                teacher_draft_logp = target_logits[action_indices].float().log_softmax(dim=-1).gather(
                    1, safe_draft[:, None]
                ).squeeze(1)
                action_token_by_output[action_req, action_pos] = draft_token_ids[action_indices].to(output.dtype)
                teacher_action_logp_by_output[action_req, action_pos] = teacher_draft_logp.to(torch.float32)
    _record_token_source_events(
        sampling_metadata, batch_size, device, output,
        teacher_output_override=teacher_output,
        action_token_by_output=action_token_by_output,
        teacher_action_logp_by_output=teacher_action_logp_by_output,
        topk_overlap_by_output=topk_overlap_by_output,
        pds_by_output=None if pds_metrics_by_output is None else pds_metrics_by_output["pds"],
        teacher_mass_coverage_by_output=(
            None if pds_metrics_by_output is None else pds_metrics_by_output["teacher_mass_coverage"]
        ),
        local_bhattacharyya_by_output=(
            None if pds_metrics_by_output is None else pds_metrics_by_output["local_bhattacharyya"]
        ),
        student_p_teacher_top1_by_output=(
            None if pds_metrics_by_output is None else pds_metrics_by_output["student_p_teacher_top1"]
        ),
        teacher_student_logp_gap_top1_by_output=(
            None if pds_metrics_by_output is None else pds_metrics_by_output["teacher_student_logp_gap_top1"]
        ),
        valid_vocab_size=vocab_size,
    )
    _record_trigger_stop_events(
        sampling_metadata,
        output,
        internal_stop_output,
        student_topk_threshold_by_output=stop_topk_threshold_by_output,
        repeat_stop_output=repeat_stop_output,
    )
    _append_trace_events(
        sampling_metadata,
        output,
        teacher_output,
        req_ids,
        pos_in_req,
        valid_pos,
        draft_token_ids,
        target_logits,
        draft_probs,
        trigger_mask=emit_trigger,
        takeover_mask_by_output=emitted_takeover,
        standard_target_by_output=standard_target_output,
    )
    return output


def _patch_vllm_vocab_check() -> None:
    try:
        from vllm.config.speculative import SpeculativeConfig
    except Exception:
        logger.exception("[opd-rollout] failed to import SpeculativeConfig for vocab patch")
        return
    if getattr(SpeculativeConfig, "_verl_opd_vocab_patch", False):
        return

    def _warn_only(self) -> None:
        target = self.target_model_config.get_vocab_size() if self.target_model_config else None
        draft = self.draft_model_config.get_vocab_size() if self.draft_model_config else None
        if target != draft:
            logger.warning(
                "[opd-rollout] vocab_size mismatch tolerated: target=%s draft=%s",
                target, draft,
            )

    SpeculativeConfig.verify_equal_vocab_size_if_draft_model = _warn_only
    SpeculativeConfig._verl_opd_vocab_patch = True


def _patch_vllm_draft_sampling() -> None:
    try:
        from vllm.v1.spec_decode import llm_base_proposer as proposer_mod
        from vllm.v1.worker import gpu_model_runner as runner_mod
    except Exception:
        logger.exception("[opd-rollout] failed to import vLLM proposer/runner for draft patch")
        return

    cls = proposer_mod.SpecDecodeBaseProposer
    if not getattr(cls, "_verl_opd_draft_patch", False):
        original_greedy_sample = cls._greedy_sample
        original_propose = cls.propose

        def _patched_greedy_sample(self, hidden_states: torch.Tensor) -> torch.Tensor:
            if not _flag("MAESTRO_RUNTIME_DRAFT_SAMPLING", True):
                return original_greedy_sample(self, hidden_states)
            try:
                token_ids, probs = _sample_from_logits(self.model.compute_logits(hidden_states))
                if getattr(self, "_verl_opd_collect_draft_probs", False):
                    self._verl_opd_draft_prob_chunks.append(probs.detach())
                return token_ids
            except Exception:
                logger.exception("[opd-rollout] draft sampling failed; falling back to greedy")
                return original_greedy_sample(self, hidden_states)

        def _patched_propose(self, *args, **kwargs):
            collect = _flag("MAESTRO_RUNTIME_COLLECT_DRAFT_PROBS", True)
            if collect:
                self._verl_opd_draft_prob_chunks = []
                self._verl_opd_collect_draft_probs = True
            try:
                draft_token_ids = original_propose(self, *args, **kwargs)
                if collect and getattr(self, "_verl_opd_draft_prob_chunks", None):
                    chunks = self._verl_opd_draft_prob_chunks
                    probs = torch.stack(chunks, dim=1).reshape(-1, chunks[0].shape[-1])
                    self._verl_opd_draft_probs = probs
                else:
                    self._verl_opd_draft_probs = None
                return draft_token_ids
            finally:
                if collect:
                    self._verl_opd_collect_draft_probs = False

        cls._greedy_sample = _patched_greedy_sample
        cls.propose = _patched_propose
        cls._verl_opd_draft_patch = True

    runner_cls = runner_mod.GPUModelRunner
    if not getattr(runner_cls, "_verl_opd_sample_patch", False):
        original_sample = runner_cls._sample

        def _patched_sample(self, logits, spec_decode_metadata):
            if spec_decode_metadata is None:
                # vLLM emits the first response token before speculative
                # decoding starts. Defer its mask event to bookkeeping, where
                # the accepted output tokens and absolute positions are known.
                self._verl_opd_non_spec_sample = True
                self._verl_opd_pending_mask_rows = None
                return original_sample(self, logits, spec_decode_metadata)

            self._verl_opd_non_spec_sample = False
            sampling_metadata = self.input_batch.sampling_metadata
            sampling_metadata.req_ids = list(self.input_batch.req_ids)
            try:
                num_reqs = len(sampling_metadata.req_ids)
                num_tokens_no_spec = self.input_batch.num_tokens_no_spec
                num_prompt_tokens = self.input_batch.num_prompt_tokens
                sampling_metadata.verl_opd_response_base_positions = [
                    max(0, int(num_tokens_no_spec[i]) - int(num_prompt_tokens[i]))
                    for i in range(num_reqs)
                ]
            except Exception:
                raise RuntimeError("failed to compute absolute response positions for OPD mask IPC")
            self.input_batch.update_async_output_token_ids()
            if self.use_async_scheduling and self._draft_token_req_ids is not None:
                draft_token_ids_cpu, _ = self._get_draft_token_ids_cpu()
                self.input_batch.update_async_spec_token_ids(draft_token_ids_cpu)

            draft_probs = None
            if _flag("MAESTRO_RUNTIME_COLLECT_DRAFT_PROBS", True) and hasattr(self, "drafter"):
                draft_probs = getattr(self.drafter, "_verl_opd_draft_probs", None)
            sampler_output = self.rejection_sampler(spec_decode_metadata, draft_probs, logits, sampling_metadata)
            self._verl_opd_pending_mask_rows = getattr(
                sampling_metadata, "verl_opd_pending_mask_rows", None
            )
            sampling_metadata.verl_opd_pending_mask_rows = None
            return sampler_output

        runner_cls._sample = _patched_sample
        runner_cls._verl_opd_sample_patch = True

    if not getattr(runner_cls, "_verl_opd_bookkeeping_patch", False):
        original_bookkeeping_sync = runner_cls._bookkeeping_sync

        def _patched_bookkeeping_sync(self, *args, **kwargs):
            result = original_bookkeeping_sync(self, *args, **kwargs)
            pending_rows = getattr(self, "_verl_opd_pending_mask_rows", None)
            non_spec_sample = getattr(self, "_verl_opd_non_spec_sample", False)
            self._verl_opd_pending_mask_rows = None
            self._verl_opd_non_spec_sample = False
            token_trace_path = os.environ.get("MAESTRO_RUNTIME_TRACE_TOKEN_JSONL")
            token_file_records: list[dict[str, Any]] = []
            if not os.environ.get("MAESTRO_RUNTIME_MASK_IPC_SOCKET"):
                if non_spec_sample and token_trace_path:
                    try:
                        valid_sampled_token_ids = result[2]
                        req_ids_output_copy = result[4]
                        for req_idx, sampled_ids in enumerate(valid_sampled_token_ids):
                            if req_idx >= len(req_ids_output_copy):
                                continue
                            if not sampled_ids:
                                continue
                            rid = req_ids_output_copy[req_idx]
                            if rid is None:
                                continue
                            sampled = [int(x) for x in sampled_ids]
                            req_state = self.requests.get(rid)
                            if req_state is None:
                                continue
                            end_pos = len(req_state.output_token_ids)
                            start_pos = end_pos - len(sampled)
                            for local_pos, tok in enumerate(sampled):
                                token_file_records.append({
                                    "request_id": _strip_vllm_suffix(rid),
                                    "response_pos": int(start_pos + local_pos),
                                    "spec_pos": int(start_pos + local_pos),
                                    "emit_token_id": int(tok),
                                    "is_teacher": False,
                                    "topk_overlap": float("nan"),
                                })
                        if token_trace_path and token_file_records and _trace_file_writer_enabled():
                            os.makedirs(os.path.dirname(token_trace_path), exist_ok=True)
                            with open(token_trace_path, "a", encoding="utf-8") as f:
                                for record in token_file_records:
                                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    except Exception:
                        logger.exception("[opd-rollout] failed to write non-spec token trace")
                return result
            try:
                valid_sampled_token_ids = result[2]
                req_ids_output_copy = result[4]
                for req_idx, sampled_ids in enumerate(valid_sampled_token_ids):
                    if req_idx >= len(req_ids_output_copy):
                        continue
                    if not sampled_ids:
                        continue
                    rid = req_ids_output_copy[req_idx]
                    if rid is None:
                        continue
                    sampled = [int(x) for x in sampled_ids]
                    n = len(sampled)
                    if non_spec_sample:
                        row_tokens = sampled
                        row_mask = [False] * n
                        row_actions: list[int] = []
                        row_action_logps: list[float] = []
                        row_overlaps: list[float] = [float("nan")] * n
                        if _flag("MAESTRO_PDS_ENABLE", False) or _flag("MAESTRO_PDS_TRACE", False):
                            row_pds: list[float] = [float("nan")] * n
                            row_teacher_mass_coverage: list[float] = [float("nan")] * n
                            row_local_bhattacharyya: list[float] = [float("nan")] * n
                            row_student_p_teacher_top1: list[float] = [float("nan")] * n
                            row_teacher_student_logp_gap_top1: list[float] = [float("nan")] * n
                        else:
                            row_pds = []
                            row_teacher_mass_coverage = []
                            row_local_bhattacharyya = []
                            row_student_p_teacher_top1 = []
                            row_teacher_student_logp_gap_top1 = []
                    else:
                        if pending_rows is None or req_idx >= len(pending_rows):
                            raise RuntimeError(
                                "[opd-rollout] speculative output has no pending mask row: "
                                f"rid={rid} req_idx={req_idx} sampled_len={n}"
                            )
                        row = pending_rows[req_idx]
                        row_tokens = [int(x) for x in row.get("tokens", [])]
                        row_mask = [bool(x) for x in row.get("mask", [])]
                        row_actions = [int(x) for x in row.get("actions", [])]
                        row_action_logps = [float(x) for x in row.get("teacher_action_logps", [])]
                        row_overlaps = [float(x) for x in row.get("topk_overlaps", [])]
                        row_pds = [float(x) for x in row.get("pds", [])]
                        row_teacher_mass_coverage = [float(x) for x in row.get("teacher_mass_coverage", [])]
                        row_local_bhattacharyya = [float(x) for x in row.get("local_bhattacharyya", [])]
                        row_student_p_teacher_top1 = [float(x) for x in row.get("student_p_teacher_top1", [])]
                        row_teacher_student_logp_gap_top1 = [
                            float(x) for x in row.get("teacher_student_logp_gap_top1", [])
                        ]
                        if row_tokens[:n] != sampled:
                            raise RuntimeError(
                                "[opd-rollout] pending mask tokens do not match vLLM bookkeeping output: "
                                f"rid={rid} row_head={row_tokens[:min(n, 8)]} "
                                f"sampled_head={sampled[:min(n, 8)]} row_len={len(row_tokens)} sampled_len={n}"
                            )
                    req_state = self.requests.get(rid)
                    if req_state is None:
                        continue
                    end_pos = len(req_state.output_token_ids)
                    start_pos = end_pos - n
                    positions = list(range(start_pos, end_pos))
                    actions = row_actions[:n] if len(row_actions) >= n else None
                    action_logps = row_action_logps[:n] if len(row_action_logps) >= n else None
                    overlaps = row_overlaps[:n] if len(row_overlaps) >= n else None
                    pds_values = row_pds[:n] if len(row_pds) >= n else None
                    teacher_mass_coverage = (
                        row_teacher_mass_coverage[:n] if len(row_teacher_mass_coverage) >= n else None
                    )
                    local_bhattacharyya = (
                        row_local_bhattacharyya[:n] if len(row_local_bhattacharyya) >= n else None
                    )
                    student_p_teacher_top1 = (
                        row_student_p_teacher_top1[:n] if len(row_student_p_teacher_top1) >= n else None
                    )
                    teacher_student_logp_gap_top1 = (
                        row_teacher_student_logp_gap_top1[:n]
                        if len(row_teacher_student_logp_gap_top1) >= n
                        else None
                    )
                    _send_mask_chunk_ipc(
                        _strip_vllm_suffix(rid),
                        row_mask[:n],
                        sampled,
                        positions,
                        actions,
                        action_logps,
                        overlaps,
                        pds_values,
                        teacher_mass_coverage,
                        local_bhattacharyya,
                        student_p_teacher_top1,
                        teacher_student_logp_gap_top1,
                    )
                    if token_trace_path:
                        for local_pos, tok in enumerate(sampled):
                            rec = {
                                "request_id": _strip_vllm_suffix(rid),
                                "response_pos": int(start_pos + local_pos),
                                "spec_pos": int(start_pos + local_pos),
                                "emit_token_id": int(tok),
                                "is_teacher": bool(row_mask[local_pos]) if local_pos < len(row_mask) else False,
                                "topk_overlap": (
                                    float(row_overlaps[local_pos])
                                    if local_pos < len(row_overlaps)
                                    else float("nan")
                                ),
                            }
                            if local_pos < len(row_pds):
                                rec["pds"] = float(row_pds[local_pos])
                            if local_pos < len(row_teacher_mass_coverage):
                                rec["teacher_mass_coverage"] = float(row_teacher_mass_coverage[local_pos])
                            if local_pos < len(row_local_bhattacharyya):
                                rec["local_bhattacharyya"] = float(row_local_bhattacharyya[local_pos])
                            if local_pos < len(row_student_p_teacher_top1):
                                rec["student_p_teacher_top1"] = float(row_student_p_teacher_top1[local_pos])
                            if local_pos < len(row_teacher_student_logp_gap_top1):
                                rec["teacher_student_logp_gap_top1"] = float(
                                    row_teacher_student_logp_gap_top1[local_pos]
                                )
                            token_file_records.append(rec)
            except Exception:
                logger.exception("[opd-rollout] failed to flush deferred mask IPC after vLLM bookkeeping")
                raise
            if token_trace_path and token_file_records and _trace_file_writer_enabled():
                try:
                    os.makedirs(os.path.dirname(token_trace_path), exist_ok=True)
                    with open(token_trace_path, "a", encoding="utf-8") as f:
                        for record in token_file_records:
                            f.write(json.dumps(record, ensure_ascii=False) + "\n")
                except Exception:
                    logger.exception("[opd-rollout] failed to write token trace jsonl: %s", token_trace_path)
            return result

        runner_cls._bookkeeping_sync = _patched_bookkeeping_sync
        runner_cls._verl_opd_bookkeeping_patch = True


def _patch_vllm_rejection_sample() -> None:
    try:
        import vllm.v1.sample.rejection_sampler as rs_mod
    except Exception:
        logger.exception("[opd-rollout] failed to import vLLM rejection sampler")
        return

    if getattr(rs_mod, "_verl_opd_rejection_patch", False):
        return

    mode_raw = os.environ.get("MAESTRO_RUNTIME_ROLLOUT_MODE", "relay")
    mode = mode_raw.strip().lower()
    fn_map = {
        "skd": _skd_topk_rejection_sample,
        "trigger_stop": _maestro_rejection_sample,
        "relay": _maestro_rejection_sample,
    }
    if mode not in fn_map:
        raise RuntimeError(
            f"Unknown MAESTRO_RUNTIME_ROLLOUT_MODE={mode_raw!r} (normalized: {mode!r}); "
            f"supported modes: {sorted(fn_map)}"
        )
    rs_mod._verl_opd_original_rejection_sample = rs_mod.rejection_sample
    rs_mod.rejection_sample = fn_map[mode]
    rs_mod._verl_opd_rejection_patch = True
    logger.warning("[opd-rollout] patched vLLM rejection_sample mode=%s", mode)


def _port_lock_dir() -> str:
    """Environment is read lazily so a launcher can still redirect the lock dir."""
    return os.environ.get("MAESTRO_RUNTIME_VLLM_PORT_LOCK_DIR", "/tmp/.verl_opd_port_locks")


def _proc_starttime(pid: int) -> str | None:
    """Field 22 of /proc/<pid>/stat, used to tell a live pid from a reused one."""
    try:
        with open(f"/proc/{pid}/stat", "r") as handle:
            data = handle.read()
    except OSError:
        return None
    try:
        return data.rsplit(")", 1)[1].split()[19]
    except (IndexError, AttributeError):
        return None


def _port_claim(port: int) -> bool:
    """Atomically claim `port` for this process.

    Two processes can both pass a bind probe before either binds for real (the
    probe socket is closed again), and two PIDs 700 apart share the same
    PID-derived band.  An O_CREAT|O_EXCL lock file closes both races.  Any
    failure degrades to the plain bind probe instead of breaking startup.
    """
    try:
        os.makedirs(_port_lock_dir(), exist_ok=True)
    except OSError:
        return True
    path = os.path.join(_port_lock_dir(), f"{port}.lock")
    pid = os.getpid()
    stamp = _proc_starttime(pid) or "0"
    for _ in range(2):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            owner_pid, owner_start = -1, None
            try:
                with open(path, "r") as handle:
                    parts = handle.read().split()
                owner_pid = int(parts[0])
                owner_start = parts[1] if len(parts) > 1 else None
            except (OSError, ValueError, IndexError):
                owner_pid, owner_start = -1, None
            if owner_pid > 0 and owner_start is not None:
                current = _proc_starttime(owner_pid)
                if current == owner_start:
                    return False  # live holder
                if current is None and os.path.exists(f"/proc/{owner_pid}"):
                    # /proc/<pid>/stat unreadable but the pid exists: assume the
                    # holder is alive and keep the lock (never steal on a hunch).
                    return False
            try:
                os.unlink(path)  # stale (owner gone or pid reused)
            except OSError:
                return False
            continue
        except OSError:
            return True
        try:
            os.write(fd, f"{pid} {stamp}\n".encode())
        except OSError:
            pass
        finally:
            os.close(fd)
        return True
    return False


def _port_lock_sweep() -> None:
    """Drop lock files whose owner is gone; called once when the patch installs."""
    try:
        names = os.listdir(_port_lock_dir())
    except OSError:
        return
    for name in names:
        if not name.endswith(".lock"):
            continue
        path = os.path.join(_port_lock_dir(), name)
        try:
            with open(path, "r") as handle:
                parts = handle.read().split()
            pid = int(parts[0])
            start = parts[1] if len(parts) > 1 else None
        except (OSError, ValueError, IndexError):
            continue
        current = _proc_starttime(pid)
        stale = current != start
        if stale and current is None and os.path.exists(f"/proc/{pid}"):
            stale = False  # owner exists, /proc just unreadable
        if stale:
            try:
                os.unlink(path)
            except OSError:
                pass


def _patch_vllm_port_probe() -> None:
    """Avoid PaddleJob's cross-process collision in vLLM's port-0 probe."""
    if os.environ.get("MAESTRO_RUNTIME_PLATFORM_PORT_ISOLATION") != "1":
        return

    import importlib

    network_utils = importlib.import_module("vllm.utils.network_utils")
    if getattr(network_utils, "_verl_opd_port_patch", False):
        return

    original_get_open_port = network_utils.get_open_port

    def get_open_port() -> int:
        global _VLLM_PORT_COUNTER
        # Keep each server process in a PID-derived band, walk the 16-port
        # windows in a rotated random order, and take an exclusive lock on the
        # winner so a concurrent process cannot pick the same port.
        base = int(os.environ.get("MAESTRO_RUNTIME_VLLM_PORT_BASE", "48000"))
        band = (os.getpid() % 700) * 16
        windows = list(range(16))
        rotation = _VLLM_PORT_COUNTER % 16
        windows = windows[rotation:] + windows[:rotation]
        random.shuffle(windows)
        for window in windows:
            offsets = list(range(16))
            random.shuffle(offsets)
            for offset in offsets:
                candidate = base + band + window * 16 + offset
                _VLLM_PORT_COUNTER += 1
                try:
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                        sock.bind(("", candidate))
                except OSError:
                    continue
                if _port_claim(candidate):
                    return candidate
        return original_get_open_port()

    network_utils.get_open_port = get_open_port
    for module_name in (
        "vllm.v1.executor.multiproc_executor",
        "vllm.v1.executor.ray_executor",
        "vllm.v1.executor.ray_executor_v2",
        "vllm.v1.executor.uniproc_executor",
        "vllm.v1.utils",
    ):
        try:
            module = importlib.import_module(module_name)
            if hasattr(module, "get_open_port"):
                module.get_open_port = get_open_port
        except Exception:
            logger.debug("could not patch vLLM port alias in %s", module_name, exc_info=True)
    network_utils._verl_opd_port_patch = True
    _port_lock_sweep()
    logger.warning(
        "[opd-rollout] patched vLLM port allocation for PaddleJob (base=%s band=%s lock_dir=%s)",
        os.environ.get("MAESTRO_RUNTIME_VLLM_PORT_BASE", "48000"),
        (os.getpid() % 700) * 16,
        _port_lock_dir(),
    )



def _effective_config_banner() -> str:
    """One-line dump of every knob that changes rollout behaviour.

    Printed once when the patch installs so that an automated gate never has to
    trust a config file or a shell export: the process that will actually read
    these values states them itself.
    """
    def g(k, d=""):
        return os.environ.get(k, d)

    keys = [
        "MAESTRO_RUNTIME_ROLLOUT_MODE", "MAESTRO_TRIGGER_MODE", "MAESTRO_TRIGGER_TOPK",
        "MAESTRO_PDS_ENABLE", "MAESTRO_PDS_AGGREGATION", "MAESTRO_PDS_TOPK",
        "MAESTRO_PDS_THRESHOLD", "MAESTRO_PDS_WINDOW_TOKENS", "MAESTRO_PDS_WINDOW_STRIDE",
        "MAESTRO_PDS_PATIENCE", "MAESTRO_PDS_WARMUP_TOKENS", "MAESTRO_PDS_PARAGRAPH_MIN_TOKENS",
        "MAESTRO_PDS_PREFIX_ALPHA", "MAESTRO_PDS_PREFIX_TAU",
        "MAESTRO_PDS_REQUIRE_LAST16", "MAESTRO_PDS_LAST16_MARGIN",
        "MAESTRO_PDS_TRIGGER_THRESHOLD", "MAESTRO_PDS_DIFFICULTY_STEP",
        "MAESTRO_PDS_BASE_PARAGRAPHS", "MAESTRO_PDS_MAX_PARAGRAPHS",
        "MAESTRO_MAX_TAKEOVERS",
        "MAESTRO_PARAGRAPHS_PER_TAKEOVER", "MAESTRO_MAX_TAKEOVER_TOKENS",
        "MAESTRO_MAX_TAKEOVERS_BEHAVIOR", "MAESTRO_REPEAT_RECOVERY",
        "MAESTRO_RUNTIME_NUM_SPECULATIVE_TOKENS",
    ]
    return " ".join(f"{k.replace('MAESTRO_', '').replace('MAESTRO_RUNTIME_', '')}={g(k)!r}" for k in keys)


def apply_port_patch() -> None:
    """Apply only the platform port-allocation patch.

    Teacher vLLM workers do not need the Relay speculative-decoding patch, but
    they still spawn vLLM worker processes that call get_open_port().  Keep this
    entry point separate so sitecustomize can patch those spawned workers even
    when MAESTRO_RUNTIME_VLLM_PATCH_DEFER=1.
    """
    _patch_vllm_port_probe()


def apply_patches() -> None:
    global _PATCHED
    if _PATCHED:
        return
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    _patch_vllm_vocab_check()
    _patch_vllm_draft_sampling()
    _patch_vllm_rejection_sample()
    _patch_vllm_port_probe()
    _PATCHED = True
    logger.warning("[opd-rollout] vLLM speculative-decoding patches active")
    logger.warning("[opd-effective] %s", _effective_config_banner())
