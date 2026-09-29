# Over-budget repetition recovery for MAESTRO rollout.
#
# Why this lives in the agent loop and not in the vLLM sampler
# -----------------------------------------------------------
# The relay controller is patched into vLLM's rejection-sampling step, so it only
# decides which tokens of the *current* speculative block are emitted.  Tokens that
# were accepted in earlier decode steps are already appended to the vLLM sequence
# (and to its KV cache); from inside the sampler there is no way to delete them.  A
# real rewind -- drop the repeated suffix and let the teacher continue from the
# repeat onset -- therefore has to happen one level up, in the agent loop, where the
# whole response is a plain python list and a continuation can be issued as a fresh
# generate() request with prompt_ids = prompt + cleaned_prefix.
#
# What this design gives for free:
#   * the teacher leg is its own request, so it can carry its own sampling params
#     (including a repetition penalty applied only after the teacher itself loops);
#   * deleted tokens never reach the trainer, so no stale rollout log-probs survive;
#   * maestro_teacher_mask is built token by token instead of through the IPC channel.
import os
from typing import Any, Optional


def _flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int) -> int:
    try:
        return int(float(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return int(default)


def _float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


def recovery_enabled() -> bool:
    return _flag("MAESTRO_REPEAT_RECOVERY", False)


# The per-token mask an A3 run needs in the loss.  One constant is shared by the
# rollout (which writes it), the trainer (which turns it into a batch tensor) and
# the distillation loss (which reads it), so the three can never disagree about
# the key name -- a class of bug that is invisible until the loss silently
# ignores the mask.
GATE_MASK_KEY = "maestro_repeat_gate_mask"


def gate_enabled() -> bool:
    """True when the advantage gate (A3) is switched on for this run."""
    return _flag("MAESTRO_ADV_GATE_ENABLE", False)


def gate_config() -> dict[str, Any]:
    """Rollout-side half of the advantage gate.

    The gate itself lives in the distillation loss (trainer side), because that is
    where the k1 advantage is built.  The rollout only has to say *which tokens*
    are the repetition run-up / repetition run; the loss decides how much of
    their positive advantage to keep.

    ``window`` marks the tokens immediately before every rewind point: those are
    the tokens that trained the student to loop, and they survive the rewind (the
    loop itself is deleted), so they are exactly the residual poison gradient.
    ``run_tokens`` covers the other case -- a loop that could not be rewritten
    (round bound hit, teacher unavailable), where the loop tokens themselves stay
    in the sequence and must not be reinforced either.
    """
    return {
        "enabled": gate_enabled(),
        "window": _int("MAESTRO_ADV_GATE_WINDOW", 64),
        "run_tokens": _int("MAESTRO_ADV_GATE_RUN_TOKENS", 64),
        "mask_key": os.environ.get("MAESTRO_ADV_GATE_MASK_KEY", GATE_MASK_KEY),
    }


def build_gate_mask(
    ranges: list[tuple[int, int]],
    length: int,
    *,
    keep: list[int] | None = None,
) -> list[int]:
    """Materialise ``ranges`` into a 0/1 mask of exactly ``length`` entries.

    ``keep`` (optional, length ``length``) restricts the marked positions to
    tokens that stay in the training sequence -- e.g. teacher takeover tokens are
    supervised by a different branch and must not be silently rewritten by the
    gate.
    """
    mask = [0] * max(0, int(length))
    n = len(mask)
    for start, end in ranges:
        s = max(0, min(n, int(start)))
        e = max(s, min(n, int(end)))
        for i in range(s, e):
            mask[i] = 1
    if keep is not None and len(keep) == n:
        for i in range(n):
            if not keep[i]:
                mask[i] = 0
    return mask


def config() -> dict[str, Any]:
    return {
        # "normal budget" is a *monitoring* dimension only: the 2026-09-17 guide
        # revision says a confirmed repetition must trigger recovery regardless of
        # length, so this value never gates the trigger.  It is reported next to
        # the recovery so we can tell "long healthy reasoning" from "loop".
        "min_tokens": _int("MAESTRO_REPEAT_MIN_TOKENS", 2048),
        "min_run": _int("MAESTRO_REPEAT_MIN_RUN", 24),
        "ngram": _int("MAESTRO_REPEAT_NGRAM", 4),
        "ngram_min_count": _int("MAESTRO_REPEAT_NGRAM_MIN_COUNT", 10),
        "max_takeovers": _int("MAESTRO_MAX_TAKEOVERS", 2),
        "paragraphs": _int("MAESTRO_PARAGRAPHS_PER_TAKEOVER", 3),
        "teacher_tokens": _int("MAESTRO_MAX_TAKEOVER_TOKENS", 256),
        "teacher_rp": _float("MAESTRO_TEACHER_REPEAT_RP", 1.2),
        # First teacher attempt is deliberately unpenalised (same distribution as the
        # MAESTRO teacher leg); overridable so the value is never a hidden constant.
        "teacher_rp_first": _float("MAESTRO_TEACHER_RP", 1.0),
        "teacher_repeat_min_run": _int("MAESTRO_TEACHER_REPEAT_MIN_RUN", 12),
        "cooldown": _int("MAESTRO_REPEAT_COOLDOWN_TOKENS", 256),
        # The repetition channel is a safety channel: it may take over even when the
        # run has already spent max_takeovers (and even after the controller asked to
        # stop), and it keeps using the same takeover ledger (reported as
        # pre_recovery_teacher_legs + repeat_recovery_teacher_legs).  It never gates the
        # trigger on the count.  The only bound is the number of rewind rounds inside a
        # single response, so a pathological sequence cannot stall a rollout forever;
        # <= 0 means "no round bound".  Hitting it is reported, never silent.
        # Repetition remains independent of the normal paragraph/PDS quota, but a
        # pathological rollout must not monopolize the whole batch.  At this cap
        # the agent loop removes the newly detected repeating suffix and ends the
        # rollout; it does not launch another teacher leg and does not train on the
        # bad suffix.
        "max_rounds": _int("MAESTRO_REPEAT_RECOVERY_MAX_ROUNDS", 12),
        # Fault-takeover quota (2026-09-18 redesign).  A "fault" is any repeat / noeos
        # runaway.  When >= 0 it caps the recovery rounds to at most this many teacher
        # legs per rollout (min with ``max_rounds``): the first fault spends the quota
        # on ONE teacher leg, and a second fault after the quota is gone falls straight
        # to the safety-cut (truncate at the paragraph boundary, no teacher, no EOS,
        # distill on the prefix).  ``-1`` (default) keeps the legacy ``max_rounds``
        # behaviour so runs that never set the env var are unchanged.
        "fault_quota": _int("MAESTRO_FAULT_TAKEOVER_QUOTA", -1),
        # When on, every fault rewind onset is snapped back to the token right after
        # the previous "\n\n" paragraph break (see snap_onset_to_boundary): the whole
        # partial paragraph that contains the loop is dropped so both the teacher leg
        # and the terminal truncation start from a clean boundary.  Off by default so
        # an announced run's semantics are unchanged unless the launcher opts in.
        "rewind_to_boundary": _flag("MAESTRO_FAULT_REWIND_TO_BOUNDARY", False),
        # Position guard (2026-09-19): once a response has already generated this many
        # tokens, NO new takeover (TSA or fault) may start.  A fault whose rewind onset
        # lands at/after this position is not rescued by a teacher leg -- it is
        # truncated at the \n\n boundary with no teacher and no EOS (prefix distill),
        # exactly like the tsa_quota_exhausted path.  The sampler applies the same cap
        # to the PDS channel (see MAESTRO_TAKEOVER_MAX_POSITION in speculative_decode).
        # <= 0 (default) disables the guard, so unset runs are unchanged.
        "max_takeover_position": _int("MAESTRO_TAKEOVER_MAX_POSITION", -1),
        "temperature": _float("MAESTRO_ROLLOUT_TEMPERATURE", 1.0),
        "top_p": _float("MAESTRO_ROLLOUT_TOP_P", 0.95),
        "max_response": _int("MAESTRO_MAX_RESPONSE_LENGTH", 16384),
        # Sampler-side early-stop rule (see maestro/patches/vllm/speculative_decode.py
        # ``_repeat_period_match``).  The rewind detector must use the SAME rule:
        # otherwise the sampler ends a request on a short tandem cycle, the agent
        # loop finds no "stable" repetition of its own, and the already truncated
        # sequence is handed straight back without any takeover.
        "tandem_min_run": _int("MAESTRO_REPEAT_TANDEM_MIN_RUN", 24),
        "tandem_max_period": _int("MAESTRO_REPEAT_EARLY_MAX_PERIOD", 32),
        "tandem_min_repeats": _int("MAESTRO_REPEAT_EARLY_MIN_PERIOD_REPEATS", 3),
        # How far back from the reported sampler stop position an onset is looked
        # for.  The sampler's matched span is <= tandem_max_period * max(repeats)
        # (96 tokens for the defaults), plus the stride slack.
        "tandem_back_window": _int("MAESTRO_REPEAT_TANDEM_BACK_WINDOW", 192),
        # Generalised repetition: any period, no tandem bound.  Catches
        # sentence/phrase echo that the tandem rule (max_period=32) and the n-gram
        # rule (stride must equal ngram) both miss -- see find_general_repeat_start.
        # Off by default so an unrelated worker restart cannot change the semantics
        # of an already-announced run; enable explicitly per run.
        "general": _flag("MAESTRO_REPEAT_GENERAL", False),
        "general_ngram": _int("MAESTRO_REPEAT_GENERAL_NGRAM", 16),
        "general_min_span": _int("MAESTRO_REPEAT_GENERAL_MIN_SPAN", 192),
        # \n\n-watchdog (noeos_gap): a pattern-agnostic runaway guard.  A response
        # that keeps emitting tokens without ever closing a paragraph ("\n\n") is
        # almost never productive, and the strict/general/tandem detectors miss the
        # single-word, multi-word and digit loops that never settle into a stable
        # n-gram.  When the tail since the last paragraph break covers
        # >= noeos_gap_max tokens, the rewind onset is placed right after that break
        # (rewind to the previous "\n\n"); the teacher leg then tries to rescue and,
        # on max_rounds exhaustion, the still-open tail is safety-cut from the loss.
        # Off by default so an unrelated worker restart cannot change an announced
        # run's semantics; enable explicitly per run.
        "noeos_gap": _flag("MAESTRO_NOEOS_GAP", False),
        "noeos_gap_max": _int("MAESTRO_NOEOS_GAP_MAX_TOKENS", 3072),
        # Post-takeover tail cap (2026-09-20): after the LAST teacher leg ends, the
        # student may continue at most this many tokens; a rollout that wanders past
        # last_teacher_end + this cap is safety-cut at the first \n\n boundary at/after
        # the cap (no teacher, no EOS, distill on prefix -- same channel as fault cut).
        # Only applies to rollouts that WERE taken over (pure-student rollouts are bounded
        # by max_response_length instead).  <=0 disables.
        "post_takeover_tail_max": _int("MAESTRO_POST_TAKEOVER_TAIL_MAX_TOKENS", -1),
    }


def fit_logprobs(values, n: int) -> tuple[list[float], int]:
    """Pad/trim a segment of log-probs to exactly ``n`` entries.

    Returns ``(values, n_zero_filled)``.  The recovery path can only reconstruct
    rollout log-probs for the segments it generated itself; anything the teacher
    server produced without emitting log-probs is filled with 0.0 and counted so
    the distortion is visible in the diagnostics instead of being silent.
    """
    if n <= 0:
        return [], 0
    vals = [float(x) for x in (values or [])]
    if len(vals) >= n:
        return vals[:n], 0
    return vals + [0.0] * (n - len(vals)), n - len(vals)


def count_teacher_legs(mask) -> int:
    """Number of contiguous teacher segments (legs) in an absolute-position mask.

    Used to tell whether the relay controller had already spent its takeover budget
    (each leg consumes one) before the repetition channel stepped in.
    """
    legs = 0
    prev = 0
    for m in mask or []:
        cur = 1 if m else 0
        if cur and not prev:
            legs += 1
        prev = cur
    return legs


def find_repeat_start(
    token_ids: list[int],
    *,
    min_run: int,
    ngram: int,
    ngram_min_count: int,
) -> Optional[dict[str, Any]]:
    """Earliest onset of a *stable* repetition pattern, or None.

    Two independent signals, earliest onset wins:

    * a run of one token repeated ``min_run`` times -- captures whitespace / single
      token loops such as ``' \\\\' x 282``.  The onset is the *first* token of the run,
      which is what has to be removed for the teacher to see a clean prefix.
    * one ``ngram``-gram repeated ``ngram_min_count`` times inside a sliding window --
      captures short cycles like ``beef $2$ beef $2$``.  The onset is the first
      occurrence of that n-gram.

    Single numbers, brackets and '=' signs are not patterns on their own; requiring a
    full run/cycle keeps normal math expressions out of the detector.
    """
    n = len(token_ids)
    best: Optional[dict[str, Any]] = None

    def _consider(onset: int, kind: str, detail: str) -> None:
        nonlocal best
        if onset < 0:
            return
        if best is None or onset < best["repeat_start"]:
            best = {"repeat_start": int(onset), "kind": kind, "detail": detail}

    if min_run > 1:
        run_start = 0
        for i in range(1, n + 1):
            if i == n or token_ids[i] != token_ids[run_start]:
                run_len = i - run_start
                if run_len >= min_run:
                    _consider(run_start, "run", f"token={token_ids[run_start]} run={run_len}")
                run_start = i

    if ngram > 0 and ngram_min_count > 1 and n >= ngram * ngram_min_count:
        # Only count *consecutive, non-overlapping* occurrences of the same n-gram.
        # A plain "how often did this n-gram appear" counter is wrong here: inside a
        # long run of one token every window (7,7,7,7) matches, so the naive onset
        # back-computes to index ~0 instead of the actual run start.
        streak: dict[tuple[int, ...], tuple[int, int]] = {}  # key -> (count, first_index)
        for i in range(0, n - ngram + 1):
            key = tuple(token_ids[i : i + ngram])
            prev = streak.get(key)
            if prev is not None and i == prev[1] + (prev[0] - 1) * ngram + ngram:
                count, first = prev[0] + 1, prev[1]
            else:
                count, first = 1, i
            streak[key] = (count, first)
            if count >= ngram_min_count:
                _consider(first, "ngram", f"count={count}")

    return best


def _tandem_span(period: int, min_run: int, min_repeats: int) -> int:
    """Tokens covered by ``period``-sized blocks repeated often enough."""
    repeats = max(int(min_repeats), -(-int(min_run) // int(period)))
    return int(period) * repeats


def find_tandem_repeat_start(
    token_ids: list[int],
    *,
    end: Optional[int] = None,
    max_period: int = 32,
    min_repeats: int = 3,
    min_run: int = 24,
    back_window: int = 192,
) -> Optional[dict[str, Any]]:
    """Onset of the *sampler's* repetition rule, or ``None``.

    Mirrors ``_repeat_period_match`` in ``maestro/patches/vllm/speculative_decode.py``:
    a block of ``k <= max_period`` tokens repeated at least ``min_repeats`` times
    and covering at least ``min_run`` tokens.  The sampler uses this rule to end a
    request early, so the rewind has to use it too -- ``find_repeat_start`` above
    is deliberately much stricter (a single-token run of ``min_run``, or one
    4-gram repeated ``ngram_min_count`` times), and on 2026-09-17 that mismatch
    meant the sampler stopped 100% of A1 rollouts at ~150 tokens while the
    recovery recognised a loop in only 3% of them, so 97% of trajectories were
    truncated with no takeover at all.

    ``end`` anchors the search: the pattern was matched on the tail the sampler
    had already emitted, so only onsets in ``[end - back_window, end)`` are
    considered.  ``None`` means "the whole sequence ends here".
    """
    n = len(token_ids)
    end = n if end is None else max(0, min(int(end), n))
    if n == 0 or max_period < 1:
        return None
    span_max = max(_tandem_span(k, min_run, min_repeats) for k in range(1, max_period + 1))
    lo = max(0, end - max(int(back_window), span_max + 1))
    limit = min(n, end + 1)
    for start in range(lo, end):
        for k in range(1, max_period + 1):
            span = _tandem_span(k, min_run, min_repeats)
            if start + span > limit:
                continue
            # Cheap probe first: the block boundary has to repeat.  The sampler
            # prunes the same way before comparing whole blocks.
            if token_ids[start + k - 1] != token_ids[start + 2 * k - 1]:
                continue
            block = token_ids[start : start + k]
            if all(
                token_ids[start + r * k : start + (r + 1) * k] == block
                for r in range(2, span // k)
            ):
                return {
                    "repeat_start": int(start),
                    "kind": "tandem",
                    "detail": f"period={k} span={span}",
                }
    return None


def _ngram_flags(token_ids, n: int):
    """Rolling-hash duplicate flags for every sliding ``n``-gram, O(len(ids)).

    ``flags[i]`` is 1 when the ``n``-gram starting at ``i`` had already occurred
    earlier in the sequence.  A single 64-bit rolling hash keeps this O(1) per
    token instead of O(n).
    """
    MASK = (1 << 64) - 1
    BASE = 0x100000001B3
    # the term that leaves the window sits at BASE**n after the shift below
    pow_base = pow(BASE, n, 1 << 64)
    seen = set()
    flags = []
    h = 0
    ids = token_ids
    for i in range(len(ids)):
        h = (h * BASE + (int(ids[i]) & 0xFFFFFFFF) + 1) & MASK
        if i >= n:
            h = (h - ((int(ids[i - n]) & 0xFFFFFFFF) + 1) * pow_base) & MASK
        if i >= n - 1:
            if h in seen:
                flags.append(1)
            else:
                seen.add(h)
                flags.append(0)
    # flags[j] describes the n-gram starting at j
    return flags


def find_general_repeat_start(
    token_ids: list[int],
    *,
    ngram: int = 16,
    min_span: int = 192,
) -> Optional[dict[str, Any]]:
    """Onset of *generalised* repetition -- any period, no tandem bound.

    Motivation (2026-09-17, real 16384-maxed rollouts): the strict rules miss
    sentence-level echo.  Probe row 11466 repeats a ~283-token block verbatim --
    far outside ``max_period=32`` -- so the tandem rule only matched at token
    16192 of 16384, and the n-gram channel cannot fire at all (it requires a
    stride of exactly ``ngram``).  A rollout that spent 98% of its tokens
    echoing one sentence was therefore never detected.

    Rule (deliberately parameter-light): walk the response with a sliding
    ``ngram``-token window; let ``last_novel`` be the position of the last window
    that had *never* occurred before.  Everything after it is by construction
    already-seen material -- an echo -- so report a hit when that echoed region
    covers at least ``min_span`` tokens, with the onset placed exactly at
    ``last_novel + 1`` (i.e. at the start of the echo, which is also the clean
    prefix the rewind has to keep).

    This subsumes the other two channels -- a whitespace run and a short numeric
    cycle are both "everything after the onset is already seen" -- while adding
    no period bound.  It stays quiet on healthy reasoning, because a genuine
    derivation keeps producing 16-grams that never occurred before, which resets
    the counter.
    """
    ids = list(token_ids)
    n = max(int(ngram), 1)
    if len(ids) <= n:
        return None
    flags = _ngram_flags(ids, n)
    if not flags:
        return None
    last_novel = None
    for j in range(len(flags) - 1, -1, -1):
        if flags[j] == 0:
            last_novel = j
            break
    if last_novel is None:
        # every window in the whole response had already occurred: the response is
        # pure echo from the very first token
        onset = 0
    else:
        onset = last_novel + 1
    span = len(ids) - onset
    if span < int(min_span):
        return None
    return {
        "repeat_start": int(onset),
        "kind": "general",
        "detail": f"echo_span={span} ngram={n} last_novel={last_novel}",
    }


_BOUNDARY_IDS_CACHE: dict[int, frozenset] = {}


def paragraph_boundary_ids(tokenizer) -> frozenset:
    """Token ids whose decoded text contains a paragraph break (``"\\n\\n"``).

    Qwen3 encodes ``"\\n\\n"`` as a *single* token (271), plus many punctuated
    variants (``".\\n\\n"``, ``";\\n\\n"`` ...), so "tokens since the last
    paragraph break" is measured against this whole set rather than a
    consecutive-newline scan (which would only match ``"\\n\\n\\n\\n"``).  This is
    the same definition the sampler uses in
    ``maestro/patches/vllm/speculative_decode.py::_get_paragraph_boundary_token_ids``,
    so the sampler-side early stop and the agent-loop rewind agree on where a
    paragraph ends.  Cached per tokenizer instance -- the vocab walk is O(vocab)
    and must run only once.
    """
    key = id(tokenizer)
    cached = _BOUNDARY_IDS_CACHE.get(key)
    if cached is not None:
        return cached
    ids: set[int] = set()
    try:
        vocab_size = getattr(tokenizer, "vocab_size", None) or len(tokenizer)
        for tid in range(int(vocab_size)):
            try:
                s = tokenizer.decode([tid], skip_special_tokens=False)
            except Exception:
                continue
            if "\n\n" in s:
                ids.add(int(tid))
    except Exception:
        ids = set()
    result = frozenset(ids)
    _BOUNDARY_IDS_CACHE[key] = result
    return result


def find_no_boundary_tail_start(
    token_ids: list[int],
    boundary_ids,
    *,
    max_gap: int,
) -> Optional[dict[str, Any]]:
    """Onset when the tail has run ``max_gap`` tokens with no paragraph break.

    Pattern-agnostic runaway signal (see ``config()`` for the motivation): a
    response that keeps emitting without ever closing a paragraph is almost never
    productive.  The onset is placed at the token right after the last paragraph
    break -- the clean prefix to keep -- so the teacher leg regenerates from
    there, and on budget exhaustion safety-cut drops the still-open tail from the
    loss.  When no paragraph break exists at all the onset is 0 (the whole
    response is one runaway); the caller aborts that as ``empty_prefix`` because
    there is nothing to rescue from, and the sampler-side early stop has already
    bounded such a response's length.
    """
    if max_gap <= 0 or not boundary_ids:
        return None
    bset = (
        boundary_ids
        if isinstance(boundary_ids, (set, frozenset))
        else set(int(x) for x in boundary_ids)
    )
    n = len(token_ids)
    last_break_end = 0
    for p in range(n - 1, -1, -1):
        if token_ids[p] in bset:
            last_break_end = p + 1
            break
    gap = n - last_break_end
    if gap < int(max_gap):
        return None
    return {
        "repeat_start": int(last_break_end),
        "kind": "noeos_gap",
        "detail": f"noeos_gap={gap} last_break_end={last_break_end}",
    }


def snap_onset_to_boundary(token_ids: list[int], boundary_ids, onset: int) -> int:
    """Move a rewind onset back to just after the previous paragraph break.

    On a fault the whole partial paragraph that contains the loop / runaway is
    suspect, so the clean prefix to keep ends at the last ``"\\n\\n"`` boundary at
    or before ``onset``.  Returns the position right after the nearest boundary
    token strictly before ``onset``; if no boundary precedes ``onset`` the original
    ``onset`` is returned unchanged (never delete the whole response just because
    the first paragraph never closed).  Idempotent on a noeos onset, which already
    sits right after a boundary.
    """
    onset = int(onset)
    if onset <= 0 or not boundary_ids:
        return onset
    bset = (
        boundary_ids
        if isinstance(boundary_ids, (set, frozenset))
        else set(int(x) for x in boundary_ids)
    )
    for p in range(min(onset, len(token_ids)) - 1, -1, -1):
        if token_ids[p] in bset:
            return p + 1
    return onset


def locate_rewind(
    token_ids: list[int],
    *,
    cfg: dict[str, Any],
    stop_kind: Optional[str] = None,
    stop_pos: Optional[int] = None,
    boundary_ids=None,
) -> Optional[dict[str, Any]]:
    """Pick the rewind onset for one recovery round.

    Three detectors, earliest onset wins:

    * ``find_repeat_start`` -- the strict "stable repetition" rule;
    * ``find_general_repeat_start`` -- period-free echo detector (opt-in);
    * ``find_tandem_repeat_start`` -- the rule the sampler actually used to stop.

    ``stop_kind == "repeat"`` (the sampler said "this response is repeating")
    *forces* a rewind.  Only ``stop_pos`` is known in that case, so the onset is
    taken ``min_run`` tokens before it, which at least removes the tail of the
    loop; the next round re-detects on the rebuilt sequence.  What must never
    happen again is handing the truncated sequence back untouched.
    """
    hits = []
    strict = find_repeat_start(
        token_ids,
        min_run=int(cfg["min_run"]),
        ngram=int(cfg["ngram"]),
        ngram_min_count=int(cfg["ngram_min_count"]),
    )
    if strict is not None:
        hits.append(strict)
    if cfg.get("general"):
        general = find_general_repeat_start(
            token_ids,
            ngram=int(cfg.get("general_ngram", 16)),
            min_span=int(cfg.get("general_min_span", 192)),
        )
        if general is not None:
            hits.append(general)
    if cfg.get("noeos_gap") and boundary_ids:
        # \n\n-watchdog: independent of stop_kind (it catches runaways the sampler
        # never flagged), so it always runs when enabled.  Earliest onset still
        # wins below, so a co-occurring echo onset can only make the rewind more
        # conservative, never less.
        noeos = find_no_boundary_tail_start(
            token_ids,
            boundary_ids,
            max_gap=int(cfg.get("noeos_gap_max", 3072)),
        )
        if noeos is not None:
            hits.append(noeos)
    forced = str(stop_kind or "").strip().lower() == "repeat"
    # The sampler's looser rule only runs when the sampler itself flagged the
    # repetition: outside that case this function has to behave exactly as before,
    # so a run with the early stop disabled is not rewound on short tandems the
    # detector was never asked about.
    if forced:
        tandem = find_tandem_repeat_start(
            token_ids,
            end=stop_pos,
            max_period=int(cfg.get("tandem_max_period", 32)),
            min_repeats=int(cfg.get("tandem_min_repeats", 3)),
            min_run=int(cfg.get("tandem_min_run", cfg["min_run"])),
            back_window=int(cfg.get("tandem_back_window", 192)),
        )
        if tandem is not None:
            hits.append(tandem)
    result: Optional[dict[str, Any]] = None
    if hits:
        best = dict(min(hits, key=lambda h: int(h["repeat_start"])))
        best["forced"] = bool(forced)
        best["source"] = "sampler_stop" if forced else "detector"
        result = best
    elif forced:
        pos = len(token_ids) if stop_pos in (None, "") else int(stop_pos)
        pos = max(0, min(pos, len(token_ids)))
        onset = max(0, pos - int(cfg["min_run"]))
        result = {
            "repeat_start": int(onset),
            "kind": "stop_pos",
            "detail": f"sampler stop at {pos}",
            "forced": True,
            "source": "sampler_stop",
        }
    if result is None:
        return None
    # Unconditionally roll the onset back to the previous "\n\n" (2026-09-18): the
    # teacher leg (and, on quota exhaustion, the terminal truncation) then start from
    # a clean paragraph boundary rather than mid-paragraph.
    if cfg.get("rewind_to_boundary") and boundary_ids:
        snapped = snap_onset_to_boundary(token_ids, boundary_ids, int(result["repeat_start"]))
        if snapped != int(result["repeat_start"]):
            result["detail"] = f'{result["detail"]} boundary_snap={snapped}(from {result["repeat_start"]})'
            result["repeat_start"] = int(snapped)
            result["boundary_snapped"] = True
    return result


def truncate_at_paragraphs(
    token_ids: list[int],
    newline_ids: tuple[int, ...],
    max_paragraphs: int,
) -> list[int]:
    """Cut a teacher leg after ``max_paragraphs`` double-newline breaks.

    ``max_paragraphs <= 0`` means "no paragraph limit" (token cap still applies).
    """
    if max_paragraphs <= 0 or not newline_ids:
        return list(token_ids)
    nl = set(int(x) for x in newline_ids)
    paragraphs = 0
    i = 1
    while i < len(token_ids):
        if token_ids[i - 1] in nl and token_ids[i] in nl:
            paragraphs += 1
            if paragraphs >= max_paragraphs:
                return list(token_ids[: i + 1])
            i += 1
        i += 1
    return list(token_ids)
