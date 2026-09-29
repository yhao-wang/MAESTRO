# MAESTRO

Official implementation of **MAESTRO** from:

> **From Dissonance to Orchestration: Teacher Intervention in On-Policy Distillation**

MAESTRO is a teacher-guided on-policy distillation method for mathematical reasoning.
It uses paragraph-level policy disagreement to adapt both the timing and depth of teacher
intervention, improving trajectory quality while controlling off-policy deviation.

## Quick Start

```bash
git clone https://github.com/yhao-wang/MAESTRO.git
cd MAESTRO
python -m pip install -r requirements.txt
python -m pip install -e .
```

Set the model, data, and output paths:

```bash
export TRAIN_DATA=/path/to/train.parquet
export STUDENT_MODEL=/path/to/student-checkpoint
export TEACHER_MODEL=/path/to/teacher-checkpoint
export OUTPUT_DIR=/path/to/output
```

Launch training with the provided 1.7B example:

```bash
bash scripts/train_qwen3_1_7b.sh
```

The shared entrypoint is `scripts/maestro_train_core.sh`. The released launchers use
four student GPUs, four teacher GPUs, global batch size 128, PPO mini-batch size 128,
maximum response length 8192, paragraph-level PDS aggregation, and at most two teacher
interventions per trajectory. Fault-triggered truncation and boundary rewind are enabled
for robustness during rollout.

## Requirements

- Linux with NVIDIA GPUs and a CUDA-compatible PyTorch installation.
- Python 3.10 or newer.
- vLLM, Ray, and the packages listed in `requirements.txt`.
- Local access to the student and teacher checkpoints.
- Training and evaluation data in the formats expected by the data configuration.
- Enough GPU memory for colocated student rollout and teacher inference.

Install versions of PyTorch, CUDA, and vLLM that are mutually compatible with the
hardware environment. The repository does not include model weights, datasets, or
cluster credentials.

## Repository Layout

- `scripts/`: training launchers and the shared training entrypoint.
- `maestro/patches/vllm/`: rollout-time PDS computation and teacher takeover logic.
- `maestro/reward/`: mathematical reasoning reward and grading utilities.
- `maestro/eval/`: benchmark evaluation helpers.
- `verl/`: the distributed training runtime.

## Method

At each token position, MAESTRO computes teacher-mass coverage C and local
Bhattacharyya similarity B as defined in the paper. The policy disagreement score is:

```text
PDS = 1 - C * B
```

Higher PDS indicates greater local disagreement. PDS is aggregated within natural
reasoning paragraphs with an early-token weighting. A paragraph boundary whose score
exceeds the configured threshold triggers teacher intervention, while larger PDS values
receive deeper intervention.

## Validation

Run static checks before submitting a distributed training job:

```bash
bash -n scripts/maestro_train_core.sh
bash -n scripts/*.sh
python -m compileall -q maestro verl scripts
```

## License

See `LICENSE`.
