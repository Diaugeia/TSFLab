"""Render the task prompt for one session: render_prompt.py <task.json> <data_dir>."""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> None:
    task = json.loads(Path(sys.argv[1]).read_text())
    data_dir = sys.argv[2]
    template = (HERE / "prompts" / f"{task['task']}.md").read_text()
    if task["task"] == "reproduce":
        cells = ", ".join(f"{c['dataset']} at {c['pred_len']}" for c in task["cells"])
        print(template.format(paper_url=task["paper_url"], data_dir=data_dir, cells=cells))
    elif task["task"] == "autoresearch":
        print(
            template.format(
                target_method=task["target_method"],
                dataset=task["dataset"],
                data_dir=data_dir,
                target_val_mse=task["target_val_mse"],
                pred_lens=", ".join(str(p) for p in task["pred_lens"]),
                split=task["split"],
                budget=task["budget"],
            )
        )
    else:
        raise SystemExit(f"unknown task kind: {task['task']}")


if __name__ == "__main__":
    main()
