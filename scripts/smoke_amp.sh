#!/bin/bash
# Run through trainreg: trainreg run -- env CUDA_VISIBLE_DEVICES=<gpu> bash scripts/smoke_amp.sh logs/amp_smoke
set -euo pipefail
cd "$(dirname "$0")/.."
RUN_DIR=${1:?usage: bash scripts/smoke_amp.sh <new_log_dir>}
LEGACY_CKPT=${2:-}
PYTHON=${PYTHON:-.venv/bin/python}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
if [[ -e "$RUN_DIR/args.json" ]]; then
  echo "Use a new log directory: $RUN_DIR already contains args.json" >&2
  exit 2
fi
mkdir -p "$RUN_DIR"
exec > >(tee "$RUN_DIR/smoke_stdout.log") 2>&1

echo '1/5 GPU terminal-pair, depth and compiled reward-gradient checks'
"$PYTHON" tools/test_amp.py --gpu AMPGPUContracts -v

SMOKE_ARGS=(--max_clips 3 --num_envs 32 --buffer_steps 64 --warmup_steps 32
  --wm_batch 32 --pi_batch 32 --amp_disc_batch 32 --wm_batches 1 --wm_only_iters 5
  --critic_epochs 1 --critic_minibatches 1 --no_compile --profile
  --ckpt_every 25 --log_every 5 --seed 0)
echo '2/5 Three-stage, depth-latent AMP training: 25 iterations, 45 reference frames, 32 BPTT steps'
bash scripts/train.sh "$RUN_DIR" "${SMOKE_ARGS[@]}" --iterations 25

echo '3/5 Restore full learning state and continue to iteration 30'
bash scripts/train.sh "$RUN_DIR" "${SMOKE_ARGS[@]}" --resume "$RUN_DIR/last.pt" --iterations 30

echo '4/5 Validation, repeated batched evaluation and checkpoint comparison'
"$PYTHON" -m blinkmimic.evaluate --run_dir "$RUN_DIR" --ckpt "$RUN_DIR/last.pt" \
  --split val --max_clips 3 --output "$RUN_DIR/smoke_val.json"
"$PYTHON" tools/eval_batch.py --run_dir "$RUN_DIR" --split val --max_clips 3 --group 2 --reps 2
METRIC_CHECKPOINTS=("$RUN_DIR/ckpt_000025.pt" "$RUN_DIR/last.pt")
if [[ -n "$LEGACY_CKPT" ]]; then
  METRIC_CHECKPOINTS+=("$LEGACY_CKPT")
fi
"$PYTHON" tools/eval_metrics.py "${METRIC_CHECKPOINTS[@]}" --split val --max_clips 3 --out "$RUN_DIR/smoke_metrics"

echo '5/5 Check training, discriminator, checkpoint and evaluation results'
"$PYTHON" - "$RUN_DIR" <<'PY'
import csv
import json
import math
from pathlib import Path
import sys
import torch

root = Path(sys.argv[1])
ck = torch.load(root / 'last.pt', map_location='cpu', weights_only=False)
assert ck['iteration'] == ck['amp_updates'] == 30
assert ck['args']['goal_frames'] == 45 and ck['args']['horizon'] == 32
assert ck['args']['imitation_mode'] == 'amp' and not ck['args']['pi_ref_skip']
assert ck['motion_prior']['state_dict']['norm.count'].item() > 0
for key in ('pi', 'wm', 'critic'):
    assert all(torch.isfinite(v).all() for v in ck[key].values()), key
assert all(torch.isfinite(v).all() for v in ck['motion_prior']['state_dict'].values())
with (root / 'train_log.csv').open() as handle:
    rows = list(csv.DictReader(handle))
assert rows and all(int(r['nonfinite']) == 0 for r in rows)
for row in rows:
    for key in ('wm_loss', 'pi_loss', 'v_loss', 'pi_gradnorm', 'amp_disc_loss', 'amp_grad_penalty'):
        assert math.isfinite(float(row[key])), (key, row)
assert any(float(r['pi_gradnorm']) > 0 for r in rows)
evaluation = json.loads((root / 'smoke_val.json').read_text())
assert evaluation['imitation_mode'] == 'amp'
assert len(evaluation['per_stage']) == 3
for key in ('carry_task_norm_return', 'mean_amp_style', 'mean_amp_score', 'mean_control_penalty',
            'hand_contact_agreement', 'release_success_rate', 'release_stand_success_rate'):
    assert math.isfinite(evaluation[key]), key
common_keys = ('carry_task_norm_return', 'completion_fraction', 'mean_object_error_m', 'mean_control_penalty',
               'hand_contact_agreement', 'release_success_rate', 'release_stand_success_rate')
comparison = json.loads((root / 'smoke_metrics.json').read_text())
for result in [*comparison.values(), *(json.loads(p.read_text()) for p in root.glob('*_eval_r2.json'))]:
    for key in common_keys:
        assert math.isfinite(result[key]), (key, result)
    if result['imitation_mode'] == 'amp':
        assert math.isfinite(result['mean_amp_style']) and math.isfinite(result['mean_amp_score'])
summary = dict(iterations=ck['iteration'], discriminator_updates=ck['amp_updates'],
               reference_seconds=.9, bptt_seconds=.64, stages=3,
               resume='passed', validation='passed', finite_gradients='passed',
               repeated_batch_evaluation='passed', metric_evaluation='passed',
               mixed_mode_evaluation='passed' if any(r['imitation_mode'] == 'tracking' for r in comparison.values()) else 'not_requested')
(root / 'smoke_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
PY
