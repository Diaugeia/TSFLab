"""Run a list of sessions with bounded concurrency per GPU.

    python3 queue.py <jobs.tsv> <out-root> [--gpus 0,1,2,3] [--per-gpu 2]

jobs.tsv columns (tab-separated, '#' comments): task  spec  rep  [agent]  [model]  [env...]
  spec is an arm or arm@commit; env entries are KEY=VALUE pairs passed to the session.
A job whose run directory already has metrics.json is skipped, so the queue can be
restarted. Each session runs run_session.sh with TRAIN_GPU set; logs go to
<out-root>/<name>.log. Start it detached (setsid nohup ... &); progress is printed
to stdout with timestamps.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def jobs(path: Path):
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        f = line.split("\t")
        task, spec, rep = f[0], f[1], f[2]
        agent = f[3] if len(f) > 3 and f[3] else "codex"
        model = f[4] if len(f) > 4 and f[4] else ""
        env = dict(kv.split("=", 1) for kv in f[5:] if "=" in kv)
        arm, _, commit = spec.partition("@")
        tag = model.split("/")[-1] if model else agent
        name = f"{task}-{spec.replace('@', '-')}-{tag}-r{rep}"
        yield dict(task=task, arm=arm, commit=commit, agent=agent, model=model, env=env, name=name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs"); ap.add_argument("root")
    ap.add_argument("--gpus", default="0,1,2,3"); ap.add_argument("--per-gpu", type=int, default=2)
    a = ap.parse_args()
    root = Path(a.root).resolve(); root.mkdir(parents=True, exist_ok=True)
    gpus = a.gpus.split(",")
    todo = [j for j in jobs(Path(a.jobs)) if not (root / j["name"] / "metrics.json").exists()]
    for j in todo:  # a partial run directory from an interrupted queue is moved aside
        d = root / j["name"]
        if d.exists():
            d.rename(root / f"{j['name']}.stale-{int(time.time())}")
    running: dict[str, tuple[subprocess.Popen, str]] = {}
    log = lambda m: print(time.strftime("%Y-%m-%d %H:%M:%S"), m, flush=True)
    log(f"{len(todo)} jobs, gpus {gpus}, {a.per_gpu} per gpu")
    while todo or running:
        for name, (p, g) in list(running.items()):
            if p.poll() is not None:
                ok = (root / name / "metrics.json").exists()
                log(f"done {name} gpu {g} exit {p.returncode} metrics {ok}")
                del running[name]
        load = {g: sum(1 for _, gg in running.values() if gg == g) for g in gpus}
        while todo:
            g = min(gpus, key=lambda x: load[x])
            if load[g] >= a.per_gpu:
                break
            j = todo.pop(0)
            env = dict(os.environ, AGENT=j["agent"], TRAIN_GPU=g, TSFLAB_COMMIT=j["commit"], **j["env"])
            cmd = [str(HERE / "run_session.sh"), str(HERE / "tasks" / f"{j['task']}.json"), j["arm"], str(root / j["name"])]
            if j["model"]:
                cmd.append(j["model"])
            out = open(root / f"{j['name']}.log", "w")
            running[j["name"]] = (subprocess.Popen(cmd, env=env, stdout=out, stderr=subprocess.STDOUT,
                                                   stdin=subprocess.DEVNULL, start_new_session=True), g)
            load[g] += 1
            log(f"start {j['name']} gpu {g}")
            time.sleep(20)  # stagger installs and API sessions
        time.sleep(30)
    log("queue empty")


if __name__ == "__main__":
    main()
