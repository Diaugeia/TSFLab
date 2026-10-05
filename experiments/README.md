# experiments/

Local research workspace. Only this README and each experiment's `scripts/`
are version-controlled; results stay on the machine that produced them
(see `.gitignore`).

| Path | Contents | Tracked |
| --- | --- | --- |
| `benchmark-1.0/scripts/` | `make_runs.py` and execution policies for the 1.0 static benchmark (see its README) | yes |
| `benchmark-1.0/runs/` | generated run files and overlays | no |
| `repo-bench/scripts/` | `baseline.sh` / `final.sh` — code-quality measurements of TSFLab vs. Time-Series-Library, TFB, PyOmniTS | yes |
| `repo-bench/measurements/` | ruff / radon / vulture / jscpd reports, before and after | no |
| `repo-bench/repos/` | cloned comparison repositories | no |
| `data/` | raw server result bundles | no |
