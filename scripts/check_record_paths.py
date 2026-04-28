"""
Path-resolution check for main.py's --save-video / --video-path / --no-record
flags.  Mocks the argparse + resolution logic so it can be exercised without
firing up Drake or Meshcat.

Run:
    python3 scripts/check_record_paths.py

Re-run after touching the recording flags in main.py to confirm all four
scenarios still produce the expected file paths.
"""
import argparse


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("task", nargs="?", default="pushing",
                   choices=["pushing", "hard_pushing", "shepherding"])
    p.add_argument("--save-video", metavar="OUTPUT.mp4", nargs="?",
                   const="AUTO", default="AUTO")
    p.add_argument("--video-path", type=str, nargs="?",
                   const="AUTO", default="AUTO", metavar="PATH.html")
    p.add_argument("--no-record", action="store_true")
    return p


def resolve(args: argparse.Namespace, run_stamp: str):
    task_name = args.task
    if args.no_record:
        return None, None
    if args.save_video == "AUTO":
        video_path = f"results/{task_name}_{run_stamp}.mp4"
    elif args.save_video == "":
        video_path = f"results/{task_name}.mp4"
    else:
        video_path = args.save_video
    if args.video_path == "AUTO":
        html_path = f"results/{task_name}_{run_stamp}.html"
    elif args.video_path == "":
        html_path = f"results/{task_name}.html"
    else:
        html_path = args.video_path
    return video_path, html_path


def main() -> None:
    p = build_parser()
    stamp = "20260427_141523"
    cases = [
        (["pushing"],
         "1: defaults (auto-named mp4 + html sharing the timestamp)"),
        (["pushing", "--save-video", "results/west.mp4",
                      "--video-path",  "results/west.html"],
         "2: explicit overrides"),
        (["pushing", "--no-record"],
         "3: --no-record (txt log only)"),
        (["pushing", "--save-video", "--video-path"],
         "4: bare --save-video --video-path (auto-naming)"),
    ]
    for argv, label in cases:
        args = p.parse_args(argv)
        v, h = resolve(args, stamp)
        print(f"[{label}]")
        print(f"  args.save_video = {args.save_video!r}")
        print(f"  args.video_path = {args.video_path!r}")
        print(f"  args.no_record  = {args.no_record}")
        print(f"  → video_path = {v}")
        print(f"  → html_path  = {h}")


if __name__ == "__main__":
    main()
