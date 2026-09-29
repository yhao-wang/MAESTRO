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
import os

from ray._private.runtime_env.constants import RAY_JOB_CONFIG_JSON_ENV_VAR

from verl.utils.device import get_device_capability

_major, _ = get_device_capability()
# Opt-in GB200 NCCL WAR: set TLLM_DISABLE_NVLS_MNNVL=1 in the launch shell to disable
# both NCCL_NVLS_ENABLE and NCCL_MNNVL_ENABLE on Blackwell. Required by async-RL
# Megatron on GB200 nodes without IMEX (mbridge all_gather raises NCCL 801).
_gb200_nccl_env = {}
if (_major or 0) >= 10 and os.environ.get("TLLM_DISABLE_NVLS_MNNVL", "0") == "1":
    _gb200_nccl_env = {"NCCL_NVLS_ENABLE": "0", "NCCL_MNNVL_ENABLE": "0"}

PPO_RAY_RUNTIME_ENV = {
    "env_vars": {
        "TOKENIZERS_PARALLELISM": "true",
        "NCCL_DEBUG": "WARN",
        "VLLM_LOGGING_LEVEL": "WARN",
        "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        # TODO: disable compile cache due to cache corruption issue
        # https://github.com/vllm-project/vllm/issues/31199
        "VLLM_DISABLE_COMPILE_CACHE": "1",
        # Needed for multi-processes colocated on same NPU device
        # https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/maintenref/envvar/envref_07_0143.html
        "HCCL_HOST_SOCKET_PORT_RANGE": "auto",
        "HCCL_NPU_SOCKET_PORT_RANGE": "auto",
        "HSA_NO_SCRATCH_RECLAIM": "1",
        **_gb200_nccl_env,
    },
}


def get_ppo_ray_runtime_env():
    """
    A filter function to return the PPO Ray runtime environment.
    To avoid repeat of some environment variables that are already set.
    """
    working_dir = (
        json.loads(os.environ.get(RAY_JOB_CONFIG_JSON_ENV_VAR, "{}")).get("runtime_env", {}).get("working_dir", None)
    )

    runtime_env = {
        "env_vars": PPO_RAY_RUNTIME_ENV["env_vars"].copy(),
        **({"working_dir": None} if working_dir is None else {}),
    }
    # MAESTRO launchers configure speculative decoding through environment
    # variables.  Ray's runtime_env is explicit, so preserve those variables
    # for colocated actor/teacher workers on PaddleJob.
    opd_env_keys = (
        "MAESTRO_RUNTIME_VLLM_PATCH",
        "MAESTRO_RUNTIME_VLLM_PATCH_DEFER",
        "MAESTRO_RUNTIME_ROLLOUT_MODE",
        "MAESTRO_RUNTIME_DRAFT_SAMPLING",
        "MAESTRO_RUNTIME_DRAFT_TEMPERATURE",
        "MAESTRO_RUNTIME_DRAFT_TOP_P",
        "MAESTRO_RUNTIME_DRAFT_TOP_K",
        "MAESTRO_RUNTIME_COLLECT_DRAFT_PROBS",
        "MAESTRO_RUNTIME_SOURCE_METRICS",
        "MAESTRO_RUNTIME_TOKENIZER_PATH",
        "MAESTRO_RUNTIME_SPECULATIVE_ENABLE",
        "MAESTRO_RUNTIME_TARGET_MODEL",
        "MAESTRO_RUNTIME_DRAFT_MODEL",
        "MAESTRO_RUNTIME_NUM_SPECULATIVE_TOKENS",
        "MAESTRO_RUNTIME_DRAFT_TENSOR_PARALLEL_SIZE",
        "MAESTRO_RUNTIME_LOGPROB_FP32",
        "MAESTRO_RUNTIME_LOGPROB_FP32_CHUNK",
        "MAESTRO_RUNTIME_LOGPROB_FLOOR",
        "MAESTRO_RUNTIME_LOGPROB_DIAG",
        "MAESTRO_TRIGGER_TOPK",
        "MAESTRO_TRIGGER_MODE",
        "MAESTRO_MECHANICAL_TEACHER_TOKENS",
        "MAESTRO_MECHANICAL_STUDENT_INTERVAL_TOKENS",
    "MAESTRO_MECHANICAL_TAKEOVER_POSITIONS",
        "MAESTRO_MECHANICAL_ONE_SHOT",
        "MAESTRO_MAX_TAKEOVERS",
        "MAESTRO_PARAGRAPHS_PER_TAKEOVER",
        "MAESTRO_PDS_AGGREGATION",
        "MAESTRO_PDS_PREFIX_ALPHA",
        "MAESTRO_PDS_PREFIX_TAU",
        "MAESTRO_PDS_PARAGRAPH_MIN_TOKENS",
        "MAESTRO_PDS_GRAY_UPPER",
        "MAESTRO_PDS_PARAGRAPH_THRESHOLD",
        "MAESTRO_PDS_PARAGRAPH_MIN_GAP",
        "MAESTRO_PDS_PARAGRAPH_LAST16_THRESHOLD",
        "MAESTRO_PDS_FALLBACK_PATIENCE",
        "MAESTRO_MAX_TAKEOVER_TOKENS",
        "MAESTRO_MAX_TAKEOVERS_BEHAVIOR",
        "MAESTRO_EXPORT_STUDENT_ACTION",
        "MAESTRO_PDS_ENABLE",
        "MAESTRO_PDS_TRACE",
        "MAESTRO_PDS_TOPK",
        "MAESTRO_PDS_THRESHOLD",
        "MAESTRO_PDS_WINDOW_TOKENS",
        "MAESTRO_PDS_WINDOW_STRIDE",
        "MAESTRO_PDS_PATIENCE",
        "MAESTRO_PDS_WARMUP_TOKENS",
        "MAESTRO_PDS_MAX_TRACKED_REQUESTS",
        # Agent-loop repetition recovery (rewind + teacher takeover on a confirmed
        # loop).  These are read by the agent loop *inside* the Ray worker, so they
        # must be whitelisted or Ray prunes them and the feature silently stays off.
        "MAESTRO_REPEAT_RECOVERY",
        # Worker-written evidence file: the one-shot "[opd-repeat-recovery] ... effective
        # cfg=" log line proved unreliable inside training workers, so the worker also
        # appends its effective config here.  Without this key in the whitelist the
        # worker falls back to /tmp and the per-run check cannot find it.
        "MAESTRO_ANNOUNCE_LOG",
        "MAESTRO_REPEAT_MIN_TOKENS",
        "MAESTRO_REPEAT_MIN_RUN",
        "MAESTRO_REPEAT_NGRAM",
        "MAESTRO_REPEAT_NGRAM_MIN_COUNT",
        "MAESTRO_REPEAT_RECOVERY_MAX_ROUNDS",
        # Fault-takeover redesign (2026-09-18): unify repeat/noeos into one fault,
        # cap it at fault_quota teacher legs, and rewind every fault onset to the
        # previous \n\n boundary.  Read by repeat_recovery.config() inside the Ray
        # worker, so both must be whitelisted or the feature silently stays off
        # (fault_quota falls back to -1 = legacy, rewind_to_boundary to off).
        "MAESTRO_FAULT_TAKEOVER_QUOTA",
        "MAESTRO_FAULT_REWIND_TO_BOUNDARY",
        # low-PDS bonus takeover after quota exhaustion (default off / cutoff 0.83)
        "MAESTRO_LOW_PDS_BONUS_TAKEOVER",
        "MAESTRO_LOW_PDS_BONUS_CUTOFF",
        "MAESTRO_TAKEOVER_MAX_POSITION",
        "MAESTRO_TAKEOVER_FLOOR_START",
        "MAESTRO_REPEAT_COOLDOWN_TOKENS",
        # Sampler-side repetition early stop: the vLLM worker ends a request at
        # the onset of a confirmed loop so the agent loop rewinds a few dozen
        # tokens instead of a whole 16384-token budget.  Read inside the vLLM
        # worker process, so it must survive Ray's env pruning too.
        "MAESTRO_REPEAT_EARLY_STOP",
        "MAESTRO_REPEAT_EARLY_MIN_RUN",
        "MAESTRO_REPEAT_EARLY_MAX_PERIOD",
        "MAESTRO_REPEAT_EARLY_MIN_PERIOD_REPEATS",
        "MAESTRO_REPEAT_EARLY_CHECK_STRIDE",
        # Read by the agent loop (rewind site), not only by the sampler: the
        # rewind has to use the same rule the early stop used, otherwise the
        # stop fires and the takeover never happens.
        "MAESTRO_REPEAT_TANDEM_MIN_RUN",
        "MAESTRO_REPEAT_TANDEM_BACK_WINDOW",
        "MAESTRO_REPEAT_GENERAL",
        "MAESTRO_REPEAT_GENERAL_NGRAM",
        "MAESTRO_REPEAT_GENERAL_MIN_SPAN",
        # \n\n-watchdog: fire when the tail runs this many tokens with no paragraph
        # break (pattern-agnostic runaway signal).  Shares the max_rounds recovery
        # budget; sampler-side early stop + agent-loop rewind-to-last-\n\n.
        "MAESTRO_NOEOS_GAP",
        "MAESTRO_NOEOS_GAP_MAX_TOKENS",
        "MAESTRO_POST_TAKEOVER_TAIL_MAX_TOKENS",
        "MAESTRO_TEACHER_RP",
        "MAESTRO_TEACHER_REPEAT_RP",
        "MAESTRO_TEACHER_REPEAT_MIN_RUN",
        "MAESTRO_MAX_RESPONSE_LENGTH",
        "MAESTRO_ROLLOUT_TEMPERATURE",
        "MAESTRO_ROLLOUT_TOP_P",
        # A3 advantage gate: the rollout marks repetition run-up / repetition run
        # tokens (MAESTRO_ADV_GATE_*), the trainer turns the marks into a batch
        # tensor and the distillation loss rewrites the positive k1 advantage on
        # them.  Read inside the rollout worker *and* the actor worker, so a
        # missing entry here silently disables the gate.
        "MAESTRO_ADV_GATE_ENABLE",
        "MAESTRO_ADV_GATE_WINDOW",
        "MAESTRO_ADV_GATE_RUN_TOKENS",
        "MAESTRO_ADV_GATE_POS_SCALE",
        "MAESTRO_ADV_GATE_NEG_SCALE",
        "MAESTRO_ADV_GATE_MASK_KEY",
        "MAESTRO_ADV_GATE_ENTROPY_MAX",
        "MAESTRO_RUNTIME_PLATFORM_PORT_ISOLATION",
        "MAESTRO_RUNTIME_MASTER_PORT_BASE",
        "MAESTRO_RUNTIME_MASTER_PORT_WIDTH",
        "MAESTRO_RUNTIME_VLLM_PORT_BASE",
        "MAESTRO_RUNTIME_TRACE",
        "MAESTRO_RUNTIME_TRACE_MODE",
        "MAESTRO_RUNTIME_TRACE_TOPK",
        "MAESTRO_RUNTIME_TRACE_ALL_TOKENS",
        "MAESTRO_RUNTIME_TRACE_JSONL",
        "MAESTRO_RUNTIME_STOP_EVENTS_JSONL",
        # Worker-side diagnostics.  The FSDP engine reads this inside the Ray
        # worker to emit pre/post-clip grad norms; without the whitelist entry
        # Ray prunes it and the instrumentation silently never fires.
        "MAESTRO_LOG_GRAD_POSTCLIP",
        "VLLM_ENABLE_V1_MULTIPROCESSING",
    )
    runtime_env["env_vars"].update({key: os.environ[key] for key in opd_env_keys if os.environ.get(key) is not None})
    for key in list(runtime_env["env_vars"].keys()):
        if key not in opd_env_keys and os.environ.get(key) is not None:
            runtime_env["env_vars"].pop(key, None)
    return runtime_env
