# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import logging
import os
import sys
import time
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.experimental.agent_loop import repeat_recovery as rr
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from verl.workers.rollout.replica import TokenOutput

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# One-shot flag: the worker prints the *effective* recovery config the first time the
# path is used, which is the only reliable proof that these env vars survived Ray's
# runtime_env whitelist (the same trap that once made repetition_penalty inert).
_RECOVERY_ANNOUNCED = False


def _announce_breadcrumb(recovery_cfg: dict[str, Any], gate_cfg: dict[str, Any]) -> None:
    """Append the worker-side effective config to a JSONL file.

    The one-shot log line below is the historical proof that these env vars survived
    Ray's runtime_env whitelist, but it has turned out to be unreliable *from inside a
    training worker*: on 2026-09-17 the 1.7B arm produced non-empty ``recovery/*``
    metrics (which can only come from this function) while
    ``[opd-repeat-recovery] agent_loop effective cfg=`` never appeared in training.log,
    in the Ray worker ``.err`` files, or anywhere else -- and the launcher's step-1
    gate therefore killed a healthy run.  A file written by the worker itself cannot be
    swallowed by log plumbing, so the same payload is dumped here as well.
    """
    path = os.environ.get("MAESTRO_ANNOUNCE_LOG", "/tmp/opd_worker_announce.jsonl")
    try:
        record = {
            "ts": time.time(),
            "pid": os.getpid(),
            "ppid": os.getppid(),
            "cwd": os.getcwd(),
            "recovery": recovery_cfg,
            "gate": gate_cfg,
        }
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, sort_keys=True, default=str) + "\n")
    except Exception:  # never let an evidence file break a rollout
        logger.exception("[opd-announce] could not write worker breadcrumb")


def _adv_gate_marks(final_ids, final_mask, gate_ranges, gate, cfg, stats):
    """Per-token 0/1 marks for the A3 advantage gate, or None when it is off.

    Two sources, both anchored on the repetition detector:

    * ``gate_ranges`` -- the ``window`` tokens that immediately precede every
      rewind point.  The loop itself is deleted by the recovery, but this run-up
      is what taught the student to loop and it stays in the training sequence, so
      it carries the residual "reinforce the degenerate token" gradient.
    * a loop still present in the returned sequence (round bound hit, teacher
      unavailable) -- marked from its onset, because those tokens are exactly the
      ones the current update would strengthen.

    Teacher-owned positions are never marked: the teacher leg is the clean
    rewrite and is supervised by its own branch.
    """
    if not gate["enabled"]:
        return None
    marks = rr.build_gate_mask(gate_ranges, len(final_ids))
    if not gate_ranges:
        hit = rr.find_repeat_start(
            list(final_ids),
            min_run=int(cfg["min_run"]),
            ngram=int(cfg["ngram"]),
            ngram_min_count=int(cfg["ngram_min_count"]),
        )
        if hit is not None:
            onset = int(hit["repeat_start"])
            marks = rr.build_gate_mask(
                [(onset, onset + int(gate["run_tokens"]))], len(final_ids)
            )
            stats["adv_gate_marked_surviving_loop"] = 1
    # A3a-full: the strict/run_tokens window above only covers a fixed 64-token
    # slice, which is ~nothing against a surviving loop whose onset lands at token
    # 40-900 and runs to ~16k -- masking it is the same "delete 25 tokens of a
    # 16000-token loop" no-op that let the collapse survive.  When the period-free
    # detector is on, union the WHOLE echo region [onset, end] into the mask so the
    # still-present loop's positive k1 advantage is actually zeroed instead of only
    # sampled.  Gated on cfg["general"], so a run without it is byte-identical to
    # the legacy gate.  This fires on both call sites: the no-rewrite hand-back and
    # the "rewound but the student tail looped again after hand-off" residual.
    if cfg.get("general"):
        gen = rr.find_general_repeat_start(
            list(final_ids),
            ngram=int(cfg.get("general_ngram", 16)),
            min_span=int(cfg.get("general_min_span", 192)),
        )
        if gen is not None:
            g_onset = max(0, min(len(marks), int(gen["repeat_start"])))
            for i in range(g_onset, len(marks)):
                marks[i] = 1
            stats["adv_gate_marked_general_full"] = 1
            stats["adv_gate_general_onset"] = g_onset
            stats["adv_gate_general_span"] = len(marks) - g_onset
    if final_mask is not None and len(final_mask) == len(marks):
        marks = [m if not t else 0 for m, t in zip(marks, final_mask)]
    stats["adv_gate_enabled"] = 1
    stats["adv_gate_marked_tokens"] = int(sum(marks))
    stats["adv_gate_ranges"] = len(gate_ranges)
    return marks


@register("single_turn_agent")
class SingleTurnAgentLoop(AgentLoopBase):
    """Naive agent loop that only do single turn chat completion."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length
        # Teacher client is injected by AgentLoopWorker (single-teacher path).
        self.teacher_client = kwargs.get("teacher_client") or {}
        self.teacher_key = None
        try:
            self.teacher_key = self.config.distillation.teacher_key
        except Exception:
            self.teacher_key = None

    # ---- over-budget repetition recovery -------------------------------------
    #
    # Sequence surgery has to happen here rather than inside the vLLM sampler:
    # the sampler can only cut the block it is currently emitting, while a real
    # rewind has to drop tokens that were already accepted steps ago.  See
    # repeat_recovery.py for the full rationale.
    def _teacher_handle(self):
        if not self.teacher_client:
            return None
        if len(self.teacher_client) == 1:
            return next(iter(self.teacher_client.values()))
        if self.teacher_key and self.teacher_key in self.teacher_client:
            return self.teacher_client[self.teacher_key]
        return None

    async def _generate(self, prompt_ids, sampling_params, tag):
        return await self.server_manager.generate(
            request_id=f"relay{tag}-{uuid4().hex}",
            prompt_ids=prompt_ids,
            sampling_params=sampling_params,
        )

    async def _teacher_leg(self, prompt_ids, cfg, stats, rollout_sampling=None):
        client = self._teacher_handle()
        if client is None:
            stats["teacher_repeat_skipped_no_client"] = 1
            return [], []
        # The teacher leg runs as its own request: take the sampling settings from the
        # live rollout request (not from a second, possibly stale source), and only fall
        # back to the env-provided defaults when the request does not carry them.
        rollout_sampling = rollout_sampling or {}
        base = {
            "max_tokens": int(cfg["teacher_tokens"]),
            "temperature": float(rollout_sampling.get("temperature", cfg["temperature"])),
            "top_p": float(rollout_sampling.get("top_p", cfg["top_p"])),
            "top_k": int(rollout_sampling.get("top_k", 20)),
            "min_p": float(rollout_sampling.get("min_p", 0.0)),
            "presence_penalty": float(rollout_sampling.get("presence_penalty", 0.0)),
        }
        out = await client.generate(
            request_id=f"relayteacher-{uuid4().hex}",
            prompt_ids=prompt_ids,
            sampling_params=dict(base, repetition_penalty=float(cfg["teacher_rp_first"])),
        )
        toks = list(out.token_ids)
        logps = list(out.log_probs) if out.log_probs is not None else []
        stats["teacher_tokens"] = len(toks)
        # If the teacher itself loops, retry once with a repetition penalty.  This
        # only affects the teacher leg because it is a separate request; the global
        # rollout rp (which the student shares with the spec-decode target) stays
        # untouched.
        hit = rr.find_repeat_start(
            toks,
            min_run=int(cfg["teacher_repeat_min_run"]),
            ngram=int(cfg["ngram"]),
            ngram_min_count=int(cfg["ngram_min_count"]),
        )
        if hit is not None:
            stats["teacher_repeat_detected"] = 1
            stats["teacher_repeat_start"] = int(hit["repeat_start"])
            stats["teacher_repetition_penalty_applied"] = float(cfg["teacher_rp"])
            retry = await client.generate(
                request_id=f"relayteacherrp-{uuid4().hex}",
                prompt_ids=prompt_ids,
                sampling_params=dict(base, repetition_penalty=float(cfg["teacher_rp"])),
            )
            retry_toks = list(retry.token_ids)
            retry_logps = list(retry.log_probs) if retry.log_probs is not None else []
            retry_hit = rr.find_repeat_start(
                retry_toks,
                min_run=int(cfg["teacher_repeat_min_run"]),
                ngram=int(cfg["ngram"]),
                ngram_min_count=int(cfg["ngram_min_count"]),
            )
            if retry_hit is None:
                stats["teacher_repeat_recovered"] = 1
                stats["teacher_repeat_exit_reason"] = "rp_retry_clean"
                toks = retry_toks
                logps = retry_logps
            else:
                stats["teacher_repeat_exit_reason"] = "rp_retry_still_looping"
                # keep the shorter of the two failures rather than a longer loop
                if len(retry_toks) < len(toks):
                    toks = retry_toks
                    logps = retry_logps
        if not toks:
            stats["teacher_repeat_exit_reason"] = stats.get("teacher_repeat_exit_reason", "empty")
            return [], []
        newline_ids = self._newline_ids()
        toks = rr.truncate_at_paragraphs(toks, newline_ids, int(cfg["paragraphs"]))
        logps = logps[: len(toks)]
        stats["teacher_tokens_after_paragraph_cut"] = len(toks)
        return toks, logps

    def _newline_ids(self):
        try:
            ids = self.tokenizer.encode("\n\n", add_special_tokens=False)
            return tuple(int(x) for x in ids)
        except Exception:
            return ()

    async def _repeat_recovery(
        self, token_ids, log_probs, extra_fields, prompt_ids, sampling_params, allow_rewrite=True
    ):
        """Detector + rewind + teacher leg + student continuation.

        Returns ``(token_ids, log_probs, extra_fields, stats)``.  The caller wraps
        this in try/except: it is a recovery path, so a failure has to degrade to
        "keep the original response" rather than kill the training step.

        ``allow_rewrite=False`` runs the detector in read-only mode: the A3 gate
        wants the repetition marks but must not perform sequence surgery.  The
        argument is explicit so that enabling the advantage gate can never switch
        recovery on as a side effect.

        Trigger policy (guide revision 2026-09-17): a confirmed repetition always
        triggers the takeover, whether or not the response has passed the normal
        length budget.  ``min_tokens`` is only reported as ``over_normal_budget``.
        """
        cfg = rr.config()
        gate = rr.gate_config()
        stats: dict[str, Any] = {
            "repeat_recovery_enabled": 1,
            "repeat_recovery_triggered": 0,
            "response_tokens_before_recovery": len(token_ids),
        }
        if not token_ids:
            return token_ids, log_probs, extra_fields, stats

        extra = dict(extra_fields or {})
        # Bookkeeping on the *incoming* mask: how much of the takeover budget the
        # paragraph/PDS controller had already spent before the loop appeared.
        incoming_mask = list(extra.get("maestro_teacher_mask") or [])
        incoming_legs = rr.count_teacher_legs(incoming_mask)
        stats["pre_recovery_teacher_legs"] = incoming_legs
        stats["pre_recovery_trigger_stopped"] = int(bool(extra.get("trigger_stopped")))
        # Distinguishes the two stop channels: "trigger" is the PDS/paragraph
        # controller asking to stop, "repeat" is the sampler-side repetition
        # early stop.  The latter is what keeps the student from decoding a whole
        # 16384-token budget that recovery would then delete (A1: 12878 removed
        # tokens per response, 44 min/step).
        if extra.get("trigger_stop_kind") == "repeat":
            stats["pre_recovery_repeat_stop"] = 1
            stats["pre_recovery_repeat_stop_pos"] = int(extra.get("trigger_stop_pos") or 0)
        if incoming_legs >= int(cfg["max_takeovers"]) or extra.get("trigger_stopped"):
            # The safety channel is explicitly allowed to act after the budget is
            # gone (and after the controller already asked to stop).
            stats["repeat_recovery_beyond_quota"] = 1

        budget = int(sampling_params.get("max_tokens") or cfg["max_response"])
        base_mask = incoming_mask if len(incoming_mask) == len(token_ids) else [0] * len(token_ids)
        seq = list(token_ids)
        seq_logps = list(log_probs) if log_probs is not None else None
        mask = list(base_mask)
        last_onset = -1
        rounds = 0
        max_rounds = int(cfg["max_rounds"])
        # Fault-takeover quota (2026-09-18): when set (>=0) it caps the recovery to
        # this many teacher legs per rollout.  fault_quota=1 => a single fault gets
        # ONE teacher leg, and any recurrence after that falls straight to the
        # safety-cut below (truncate at the paragraph boundary, no teacher, no EOS).
        # spent_legs (below) already carries T=max(0,T-1) into the sampler, so the
        # TSA quota coupling needs no extra change.  fault_quota<0 keeps legacy.
        fault_quota = int(cfg.get("fault_quota", -1))
        if fault_quota >= 0:
            max_rounds = fault_quota if max_rounds <= 0 else min(fault_quota, max_rounds)
            unlimited_rounds = False
        else:
            unlimited_rounds = max_rounds <= 0
        stats["fault_quota"] = fault_quota
        # Position guard (2026-09-19): a fault whose boundary-snapped onset lands at or
        # after this token position is not rescued -- see the truncate branch below.
        max_takeover_position = int(cfg.get("max_takeover_position", -1))
        stats["max_takeover_position"] = max_takeover_position
        # Ranges whose positive k1 advantage is gated (A3).  Recorded in the
        # coordinates of the *final* sequence: every rewind leaves the prefix
        # untouched, so a range recorded at round r stays valid after later rounds
        # rewrite the suffix.
        gate_ranges: list[tuple[int, int]] = []
        # Where the *sampler* stopped this request, and why.  "repeat" means it
        # found a loop and ended the request only to stop wasting decode; the
        # takeover still has to happen here, so the stop information drives the
        # rewind (see rr.locate_rewind).  Round 0 uses the incoming rollout, later
        # rounds use the continuation leg they just ran.
        round_stop_kind = extra.get("trigger_stop_kind")
        round_stop_pos = extra.get("trigger_stop_pos")
        # \n\n-watchdog boundary set (cached per tokenizer): only resolved when the
        # watchdog is enabled, so a plain recovery run pays nothing for it.
        noeos_boundary_ids = (
            rr.paragraph_boundary_ids(self.tokenizer)
            if (cfg.get("noeos_gap") or cfg.get("rewind_to_boundary"))
            else None
        )

        while unlimited_rounds or rounds < max_rounds:
            if not allow_rewrite:
                break
            hit = rr.locate_rewind(
                seq, cfg=cfg, stop_kind=round_stop_kind, stop_pos=round_stop_pos,
                boundary_ids=noeos_boundary_ids,
            )
            if hit is None:
                break

            repeat_start = int(hit["repeat_start"])
            stats.setdefault("first_repeat_start", repeat_start)
            stats["last_repeat_start"] = repeat_start
            stats["last_repeat_kind"] = str(hit["kind"])
            stats["last_repeat_detail"] = str(hit["detail"])
            stats["last_repeat_forced"] = int(bool(hit.get("forced")))
            stats["last_repeat_source"] = str(hit.get("source", "detector"))
            stats["over_normal_budget"] = int(len(seq) > int(cfg["min_tokens"]))
            if repeat_start <= last_onset:
                # Within one sample we only ever move the rewind point forward;
                # otherwise a pathological sequence could ping-pong forever.
                stats["repeat_recovery_stalled"] = "onset_not_advancing"
                break
            cleaned = list(seq[:repeat_start])
            if not cleaned:
                stats["repeat_recovery_aborted"] = "empty_prefix"
                break
            gate_window = int(gate["window"])
            if gate_window > 0:
                gate_ranges.append((max(0, repeat_start - gate_window), repeat_start))

            # Position guard (2026-09-19): if the (boundary-snapped) fault onset is at
            # or past max_takeover_position, no teacher rescue -- truncate at the \n\n
            # boundary, no teacher, no EOS, distill on the prefix.  Same terminal shape
            # as tsa_quota_exhausted; distinct reason for accounting.  <=0 disables.
            if max_takeover_position > 0 and repeat_start >= max_takeover_position:
                cut = max(0, min(repeat_start, len(seq)))
                stats["repeat_recovery_triggered"] = 1
                stats["repeat_recovery_safety_truncated"] = 1
                stats["repeat_recovery_safety_truncate_reason"] = "past_max_position"
                stats["repeat_recovery_safety_kind"] = str(hit["kind"])
                stats["repeat_recovery_safety_cut_pos"] = cut
                stats["repeat_recovery_safety_removed_tokens"] = len(seq) - cut
                stats["repeat_recovery_blocked_max_position"] = int(
                    stats.get("repeat_recovery_blocked_max_position", 0)
                ) + 1
                seq = list(seq[:cut])
                mask = list(mask[:cut])
                if seq_logps is not None:
                    seq_logps = seq_logps[:cut]
                last_onset = repeat_start
                break

            # Fault takeover is gated by the TSA quota (user 2026-09-18 rev2):
            # a fault leg consumes one TSA slot (via spent_legs coupling), so it may
            # only fire while the paragraph/TSA channel still has budget.  Once TSA
            # is exhausted the fault is NOT allowed to take over -- rewind to the
            # \n\n boundary (repeat_start is already boundary-snapped by
            # locate_rewind), truncate there with no teacher and no EOS, and distill
            # on the prefix only.  fault_quota<0 (legacy) skips this gate entirely.
            if fault_quota >= 0:
                tsa_spent = int(stats.get("pre_recovery_teacher_legs", 0)) + rounds
                if tsa_spent >= int(cfg["max_takeovers"]):
                    cut = max(0, min(repeat_start, len(seq)))
                    stats["repeat_recovery_triggered"] = 1
                    stats["repeat_recovery_safety_truncated"] = 1
                    stats["repeat_recovery_safety_truncate_reason"] = "tsa_quota_exhausted"
                    stats["repeat_recovery_safety_kind"] = str(hit["kind"])
                    stats["repeat_recovery_safety_cut_pos"] = cut
                    stats["repeat_recovery_safety_removed_tokens"] = len(seq) - cut
                    seq = list(seq[:cut])
                    mask = list(mask[:cut])
                    if seq_logps is not None:
                        seq_logps = seq_logps[:cut]
                    last_onset = repeat_start
                    break

            teacher_tokens, teacher_logps = await self._teacher_leg(
                prompt_ids + cleaned, cfg, stats, rollout_sampling=sampling_params
            )
            if not teacher_tokens:
                stats["repeat_recovery_aborted"] = "no_teacher_tokens"
                break
            rounds += 1
            stats["repeat_recovery_triggered"] = 1
            stats[f"round{rounds}_repeat_start"] = repeat_start
            stats[f"round{rounds}_remove_tokens"] = len(seq) - repeat_start
            stats[f"round{rounds}_teacher_tokens"] = len(teacher_tokens)

            remaining = budget - len(cleaned) - len(teacher_tokens)
            student_tail: list[int] = []
            student_tail_logps: list[float] = []
            # Teacher legs this rollout has already spent before this continuation
            # leg (paragraph/PDS legs from the incoming mask plus the repeat-recovery
            # legs so far).  The sampler keys its takeover ledger by request id and
            # every continuation leg is a new request, so the count has to travel in
            # the request id or the paragraph channel would get a fresh quota each
            # round.  See _seed_spent_takeovers in maestro/patches/vllm/speculative_decode.py.
            spent_legs = int(stats.get("pre_recovery_teacher_legs", 0)) + rounds
            tail_mask: list[int] | None = None
            if remaining > 8:
                cont = dict(sampling_params)
                cont["max_tokens"] = remaining
                tail_out = None
                try:
                    tail_out = await self._generate(
                        prompt_ids + cleaned + teacher_tokens,
                        cont,
                        f"repeatcontinue{spent_legs}",
                    )
                    student_tail = list(tail_out.token_ids)
                    student_tail_logps = list(tail_out.log_probs) if tail_out.log_probs is not None else []
                except Exception:
                    logger.exception("[repeat-recovery] student continuation failed")
                # The continuation is its own request, so the sampler can early-stop
                # *it* for repetition too.  Carry that stop into the next round;
                # without this the loop exits holding a truncated tail and the
                # rollout is returned with no further takeover (the 2026-09-17 A1
                # failure mode: 100% stopped, 3% taken over, 7.3k -> 0.5k tokens).
                tail_extra = dict(getattr(tail_out, "extra_fields", None) or {}) if tail_out is not None else {}
                # Keep the sampler's own teacher mask for the tail when it lines up.
                # It should be all zeros now that the paragraph quota is inherited and
                # therefore suppressed, but if a leg ever does contain a takeover the
                # tokens must not silently pass as student actions.
                tail_mask_raw = tail_extra.get("maestro_teacher_mask")
                if isinstance(tail_mask_raw, (list, tuple)) and len(tail_mask_raw) == len(student_tail):
                    tail_mask = [int(bool(x)) for x in tail_mask_raw]
                if tail_mask is not None:
                    stats[f"round{rounds}_tail_teacher_legs"] = rr.count_teacher_legs(tail_mask)
                    stats["repeat_recovery_tail_teacher_legs"] = int(
                        stats.get("repeat_recovery_tail_teacher_legs", 0)
                    ) + int(stats[f"round{rounds}_tail_teacher_legs"])
                round_stop_kind = tail_extra.get("trigger_stop_kind")
                # The leg reports its stop in its own response coordinates; the next
                # round searches the rebuilt sequence, so shift by the prefix and
                # teacher leg that were prepended to it.
                leg_stop_pos = tail_extra.get("trigger_stop_pos")
                round_stop_pos = (
                    len(cleaned) + len(teacher_tokens) + int(leg_stop_pos)
                    if leg_stop_pos is not None
                    else None
                )
            else:
                # No room for another student leg: the response ends here however
                # the previous leg finished.  Without this the stale stop info
                # would keep forcing rewinds against a budget that is already
                # spent.
                round_stop_kind = None
                round_stop_pos = None
            stats[f"round{rounds}_tail_tokens"] = len(student_tail)

            rebuilt_logps = None
            if seq_logps is not None:
                head = [float(x) for x in seq_logps[:repeat_start]]
                teacher_fit, teacher_pad = rr.fit_logprobs(teacher_logps, len(teacher_tokens))
                tail_fit, tail_pad = rr.fit_logprobs(student_tail_logps, len(student_tail))
                rebuilt_logps = head + teacher_fit + tail_fit
                stats["logprob_zero_filled_tokens"] = int(stats.get("logprob_zero_filled_tokens", 0) + teacher_pad + tail_pad)

            seq = cleaned + teacher_tokens + student_tail
            mask = (
                list(mask[:repeat_start])
                + [1] * len(teacher_tokens)
                + (tail_mask if tail_mask is not None else [0] * len(student_tail))
            )
            # The rollout tensor contract is exactly ``budget`` response tokens.
            # A vLLM continuation may legitimately return one internal-stop slot
            # beyond the requested generation budget; after many rewinds that used
            # to leak through and produce e.g. 16503 tokens for a 16384-wide batch,
            # crashing torch.cat before step 1.  The stop slot is controller metadata,
            # not a trainable response token, so enforce the public contract here.
            if len(seq) > budget:
                overflow = len(seq) - budget
                stats["response_budget_trimmed_tokens"] = int(
                    stats.get("response_budget_trimmed_tokens", 0)
                ) + overflow
                seq = seq[:budget]
                mask = mask[:budget]
                if rebuilt_logps is not None:
                    rebuilt_logps = rebuilt_logps[:budget]
                round_stop_kind = None
                round_stop_pos = None
            seq_logps = rebuilt_logps
            last_onset = repeat_start

        if not unlimited_rounds and rounds >= max_rounds:
            # Only reported when a loop is still present at the round bound: this is the
            # one place where the "always take over" rule is capped, so it must be
            # visible (recovery/round_limit_hit_ratio) instead of looking like a clean run.
            still_looping = rr.find_repeat_start(
                seq,
                min_run=int(cfg["min_run"]),
                ngram=int(cfg["ngram"]),
                ngram_min_count=int(cfg["ngram_min_count"]),
            )
            if cfg.get("general"):
                gen_hit = rr.find_general_repeat_start(
                    seq,
                    ngram=int(cfg.get("general_ngram", 16)),
                    min_span=int(cfg.get("general_min_span", 192)),
                )
                if gen_hit is not None and (
                    still_looping is None
                    or int(gen_hit["repeat_start"]) < int(still_looping["repeat_start"])
                ):
                    still_looping = gen_hit
            if cfg.get("noeos_gap"):
                # A no-\n\n runaway need not be a stable repetition, so the strict/
                # general detectors above can miss it; without this the round budget
                # ("12") would be spent and the still-open tail handed back into the
                # loss.  Treat a surviving no-boundary tail as a reason to safety-cut,
                # exactly as the user asked: "12 预算用完了直接 response 结束，不算 loss".
                noeos_hit = rr.find_no_boundary_tail_start(
                    seq,
                    rr.paragraph_boundary_ids(self.tokenizer),
                    max_gap=int(cfg.get("noeos_gap_max", 3072)),
                )
                if noeos_hit is not None and (
                    still_looping is None
                    or int(noeos_hit["repeat_start"]) < int(still_looping["repeat_start"])
                ):
                    still_looping = noeos_hit
            # Post-takeover tail cap (moved 2026-09-21 into the vLLM sampler):
            # the student tail after the last takeover is now HARD-STOPPED at
            # last_teacher_end + MAESTRO_POST_TAKEOVER_TAIL_MAX_TOKENS during
            # generation (see _apply_post_takeover_tail_cap in
            # maestro/patches/vllm/speculative_decode.py), so the tail is never decoded
            # past the cap.  Pure hard stop, no \n\n snap (user口径 2026-09-21).
            # No post-hoc truncation here anymore -- the sequence already arrives
            # capped -- which is exactly why the old generate-then-trim was removed.
            if still_looping is not None:
                stats["repeat_recovery_round_limit_hit"] = 1
                stats["repeat_recovery_safety_truncated"] = 1
                stats["repeat_recovery_safety_truncate_reason"] = (
                    "fault_quota_exhausted" if fault_quota >= 0 else "max_rounds"
                )
                stats["repeat_recovery_safety_kind"] = str(still_looping["kind"])
                # Never train on the still-looping student suffix that caused the
                # cap.  Rewind it once more, but do not launch another teacher or
                # student request: keep only the clean prefix that precedes the
                # confirmed loop.  This is a latency guard for pathological cases,
                # not a semantic change to ordinary repeat takeovers.
                safety_cut = int(still_looping["repeat_start"])
                # Fault quota is gone: truncate at the paragraph boundary (same
                # onset rule as the takeover), no teacher, no EOS, distill on prefix.
                if cfg.get("rewind_to_boundary"):
                    safety_cut = rr.snap_onset_to_boundary(
                        seq, rr.paragraph_boundary_ids(self.tokenizer), safety_cut
                    )
                safety_cut = max(0, min(safety_cut, len(seq)))
                stats["repeat_recovery_safety_cut_pos"] = safety_cut
                stats["repeat_recovery_safety_removed_tokens"] = len(seq) - safety_cut
                seq = seq[:safety_cut]
                mask = mask[:safety_cut]
                if seq_logps is not None:
                    seq_logps = seq_logps[:safety_cut]
                # In fault-quota mode the truncation IS the outcome (rewind to \n\n,
                # no teacher, no EOS, distill on prefix).  Mark it triggered so the
                # gate below keeps the truncated prefix instead of discarding it and
                # returning the original untruncated sequence.  Legacy max_rounds
                # (fault_quota < 0) behavior is left unchanged.
                if fault_quota >= 0:
                    stats["repeat_recovery_triggered"] = 1

        if stats.get("repeat_recovery_triggered") != 1:
            # Nothing was rewritten (no loop, empty prefix, missing teacher, or the
            # rewind point refused to advance): hand the original sequence back.
            marks = _adv_gate_marks(token_ids, mask, gate_ranges, gate, cfg, stats)
            if marks is not None:
                extra[gate["mask_key"]] = marks
            return token_ids, log_probs, extra, stats

        # Final belt-and-braces assertion at the boundary consumed by
        # AgentLoopWorker._postprocess.  Never return a row wider than the rollout
        # response tensor, even if a future backend changes stop-token handling.
        new_ids = seq[:budget]
        new_logps = seq_logps[:budget] if seq_logps is not None else None
        mask = mask[:budget]
        stats["repeat_recovery_rounds"] = rounds
        stats["repeat_recovery_teacher_legs"] = rounds
        stats["teacher_legs_total"] = rounds + int(stats.get("pre_recovery_teacher_legs", 0))
        stats["takeover_count_exceeded"] = int(
            stats["teacher_legs_total"] > int(cfg["max_takeovers"])
        )
        stats["response_tokens_after_recovery"] = len(new_ids)
        # Reporting keys (kept stable for the trainer-side counters): the rewind of
        # the first round is what "removed" means here; the whole history is in the
        # per-round keys above.
        stats["repeat_start"] = int(stats.get("first_repeat_start", -1))
        stats["repeat_kind"] = str(stats.get("last_repeat_kind", "unknown"))
        stats["removed_suffix_tokens"] = int(stats.get("response_tokens_before_recovery", 0)) - int(
            stats.get("first_repeat_start", 0)
        )
        stats["teacher_tokens"] = sum(int(stats.get(f"round{r}_teacher_tokens", 0)) for r in range(1, rounds + 1))
        stats["student_tail_tokens"] = sum(int(stats.get(f"round{r}_tail_tokens", 0)) for r in range(1, rounds + 1))
        # Per-token trace fields below describe the *pre-recovery* sequence, so their
        # tail no longer lines up with the rewritten token ids.  They must not be
        # dropped from this sample only: DataProto.concat requires every sample of the
        # batch to carry the same non_tensor keys, and a rollout where only part of the
        # batch loops (the normal case) then dies with
        #   "key maestro_teacher_local_bhattacharyya length 64 is not equal to batch size 128".
        # Instead keep the part that is still valid -- every rewind only ever truncates
        # the suffix, so the prefix before the first repeat onset is untouched -- and
        # zero-fill the rest to the new response length.  0 means "no trace recorded for
        # this token" and the trainer pads/truncates these rows by response length anyway.
        keep_prefix = int(stats.get("first_repeat_start", 0))
        zero_filled = 0
        for stale in (
            "maestro_student_action_ids",
            "maestro_student_action_teacher_logprobs",
            "maestro_student_action_mask",
            "maestro_teacher_topk_overlap",
            "maestro_teacher_tas",
            "maestro_teacher_mass_coverage",
            "maestro_teacher_local_bhattacharyya",
            "maestro_student_p_teacher_top1",
            "maestro_teacher_student_logp_gap_top1",
        ):
            if stale not in extra:
                continue
            old = list(extra[stale] or [])
            cut = max(0, min(len(old), keep_prefix, len(new_ids)))
            extra[stale] = old[:cut] + [0] * (len(new_ids) - cut)
            zero_filled += 1
        stats["trace_fields_prefix_kept"] = keep_prefix
        stats["trace_fields_zero_filled"] = zero_filled
        extra["maestro_teacher_mask"] = mask
        extra["maestro_teacher_positions"] = [i for i, m in enumerate(mask) if m]
        extra["repeat_recovery_applied"] = 1
        stats["repeat_recovery_applied"] = 1
        marks = _adv_gate_marks(new_ids, mask, gate_ranges, gate, cfg, stats)
        if marks is not None:
            extra[gate["mask_key"]] = marks
        return new_ids, new_logps, extra, stats

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])

        # 1. extract multimodal inputs from messages
        multi_modal_data = await self.process_multi_modal_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")
        audios = multi_modal_data.get("audios")
        mm_processor_kwargs = self._get_mm_processor_kwargs(audios)

        # 2. apply chat template and tokenize
        prompt_ids = await self.apply_chat_template(
            messages,
            images=images,
            videos=videos,
            audios=audios,
            mm_processor_kwargs=mm_processor_kwargs,
        )

        # 3. generate sequences
        metrics = {}
        maestro_global_step = kwargs.get("__maestro_global_step", -1)
        request_id = f"relaystep{int(maestro_global_step)}-{uuid4().hex}"
        with simple_timer("generate_sequences", metrics):
            output: TokenOutput = await self.server_manager.generate(
                request_id=request_id,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                image_data=images,
                video_data=videos,
                audio_data=audios,
                mm_processor_kwargs=mm_processor_kwargs,
            )
        if metrics.get("num_preempted") is None:
            metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1
        response_length = int(sampling_params.get("max_tokens") or self.response_length)
        token_ids = list(output.token_ids[:response_length])
        log_probs = list(output.log_probs[:response_length]) if output.log_probs is not None else None
        extra_fields = dict(output.extra_fields or {})

        # Optional safety channel: a confirmed repetition (at any length) is rewound
        # to its onset and handed to the teacher, instead of letting the student keep
        # emitting the loop.  Off by default; enabled per run with
        # MAESTRO_REPEAT_RECOVERY=1.
        if (rr.recovery_enabled() or rr.gate_enabled()) and token_ids:
            global _RECOVERY_ANNOUNCED
            if not _RECOVERY_ANNOUNCED:
                _RECOVERY_ANNOUNCED = True
                _recovery_cfg = rr.config()
                _gate_cfg = rr.gate_config()
                logger.warning("[opd-repeat-recovery] agent_loop effective cfg=%r", _recovery_cfg)
                logger.warning("[opd-adv-gate] agent_loop effective cfg=%r", _gate_cfg)
                # Belt and braces: the logger above has been observed to vanish inside a
                # training worker (see _announce_breadcrumb), so also write straight to
                # stderr and keep a per-run JSONL breadcrumb.
                print(
                    f"[opd-repeat-recovery] agent_loop effective cfg={_recovery_cfg} "
                    f"[opd-adv-gate] agent_loop effective cfg={_gate_cfg}",
                    file=sys.stderr,
                    flush=True,
                )
                _announce_breadcrumb(_recovery_cfg, _gate_cfg)
            try:
                token_ids, log_probs, extra_fields, recovery_stats = await self._repeat_recovery(
                    token_ids,
                    log_probs,
                    extra_fields,
                    prompt_ids,
                    sampling_params,
                    allow_rewrite=rr.recovery_enabled(),
                )
            except Exception:
                logger.exception("[repeat-recovery] recovery failed; keeping original response")
                recovery_stats = {
                    "repeat_recovery_enabled": 1,
                    "repeat_recovery_triggered": 0,
                    "repeat_recovery_error": 1,
                }
            extra_fields["repeat_recovery"] = {
                k: v for k, v in recovery_stats.items() if isinstance(v, (int, float, str))
            }
        elif rr.recovery_enabled():
            # Same concat constraint as above: a run with recovery on must report the
            # block for *every* sample, including an empty response that never reached
            # the detector.
            extra_fields["repeat_recovery"] = {
                "repeat_recovery_enabled": 1,
                "repeat_recovery_triggered": 0,
                "response_tokens_before_recovery": 0,
            }
        # Belt and braces for the whole class of bug that killed A1 step 1: recovery may
        # rewrite the response, but it must never change the *set* of keys this sample
        # contributes to the batch, because DataProto.concat only tolerates identical
        # non_tensor keys.  Anything the rollout produced and recovery dropped gets
        # re-added (zero-filled to the current response length for per-token lists).
        #
        # The mirror image of that bug killed the 1.7B arm at step 4
        # ("Key 'repeat_recovery_applied' is not present in the keys of the first
        # dictionary"): recovery *adds* ``repeat_recovery_applied`` for the samples it
        # rewrote, so a batch where only part of the samples looped -- the normal case on
        # a 1.7B student -- came out with inconsistent keys again.  Keys that only the
        # recovery branch can add therefore have to be materialised for every sample.
        if rr.recovery_enabled():
            extra_fields.setdefault("repeat_recovery_applied", 0)
        for field_name, field_value in (output.extra_fields or {}).items():
            if field_name in extra_fields:
                continue
            if isinstance(field_value, (list, tuple)):
                extra_fields[field_name] = [0] * len(token_ids)
            else:
                extra_fields[field_name] = field_value
            logger.warning(
                "[repeat-recovery] restored trace field %r after rewrite (len=%d)",
                field_name,
                len(token_ids),
            )
        # The gate key has to exist for every sample of a batch: the trainer turns
        # the collected per-sample lists into one tensor.  A sample that never
        # entered the detector (empty response, earlier failure) gets all zeros.
        if rr.gate_enabled():
            gate = rr.gate_config()
            if gate["mask_key"] not in extra_fields:
                extra_fields[gate["mask_key"]] = [0] * len(token_ids)
            else:
                marks = list(extra_fields[gate["mask_key"]] or [])
                if len(marks) != len(token_ids):
                    logger.warning(
                        "[opd-adv-gate] mask length mismatch (%d vs %d tokens); zeroing this sample",
                        len(marks),
                        len(token_ids),
                    )
                    extra_fields[gate["mask_key"]] = [0] * len(token_ids)

        response_mask = [1] * len(token_ids)

        output: AgentLoopOutput = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=token_ids,
            response_mask=response_mask,
            response_logprobs=log_probs,
            routed_experts=(
                output.routed_experts[: len(prompt_ids) + len(token_ids)]
                if output.routed_experts is not None
                else None
            ),
            multi_modal_data=multi_modal_data,
            mm_processor_kwargs=mm_processor_kwargs,
            num_turns=2,
            metrics=metrics,
            extra_fields=extra_fields,
        )

        # keeping the schema consistent with tool_agent_loop
        output.extra_fields.update({"turn_scores": [], "tool_rewards": []})

        return output
