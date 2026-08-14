# P1 buffer-rotation profiling

Run the editable P1 list from the repository root:

```bash
scripts/run_tk_p1_rotation.sh
```

Edit `P1_CASES=(...)` in that script to select cases. Rotation counts come from
`~/Repos/ThunderKittens-RotationSize.json` by default; override it with
`ROTATION_CONFIG=/path/to/file.json`. The profiling defaults are a 15-second
post-ready delay, a 0.2-second Nsys window at 100 kHz, GPU-metrics device 0,
and a one-hour workload timeout.

Useful overrides:

```bash
scripts/run_tk_p1_rotation.sh \
  --nsys-duration 0.2 --nsys-frequency 100000 --nsys-gpus 0 \
  --nsys-delay 15 --profile-timeout 3600
```

The scheduler builds copied, repository-local MHA/A2A/GEMM variants under
`scripts/thunderkittens/p1_extensions/`. It does not edit the ThunderKittens
checkout. AG and RS use their clean official extension modules through the
independent rotation runner.

Static validation without a GPU:

```bash
scripts/run_tk_p1_rotation.sh --dry-run
```
