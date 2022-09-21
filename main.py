import os
from argparse import ArgumentParser
from pathlib import Path

from kolmev import MODEL_REGISTRY

project_path = Path(__file__).parent.resolve()


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, choices=["gpt2", "gpt-neox", "gptj"], required=True)
    parser.add_argument("--task", type=str, choices=["nsmc"], required=True)
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    task_script_path = str(project_path / "src" / "kolmev" / f"{MODEL_REGISTRY[args.model]}" / f"{args.task}.py")
    softlink_path = str(project_path / f"{args.task}.py")

    if os.path.exists(softlink_path):
        os.remove(softlink_path)
    os.system(f"ln -s {task_script_path} {softlink_path}")


if __name__ == "__main__":
    main()
