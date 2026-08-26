#!/usr/bin/env python3
"""Render one representative successful run per Figure 8 object.

The shortest validated success is used to keep the gallery compact. Videos are
rendered from the recorded state logs; this script does not rerun experiments.
"""
from __future__ import annotations

import argparse
import html
import math
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.plot_fig8 import ORDER, collect  # noqa: E402


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def render(log: Path, task: str, output: Path, duration: float,
           *, fps: int, target_frames: int, force: bool) -> None:
    if output.exists() and not force:
        print(f"keeping existing {output.relative_to(ROOT)}")
        return
    # Planner logs are normally sampled every 0.075 s. Select a stride that
    # keeps short runs fluid while bounding long-run gallery artifacts.
    stride = max(1, math.ceil(duration / (0.075 * target_frames)))
    with tempfile.TemporaryDirectory(prefix="fig8-success-frames-", dir="/tmp") as tmp:
        frames = Path(tmp)
        run([sys.executable, "tools/visualizer/render_log_drake_scene.py",
             str(log), "--task", task, "--stride", str(stride),
             "--out-dir", str(frames)])
        run([sys.executable, "tools/visualizer/paint_log_sidepanel.py",
             "--frames-dir", str(frames), "--log-path", str(log),
             "--output", str(output), "--fps", str(fps)])


def write_gallery(items: list[dict[str, str]], output: Path) -> None:
    cards = []
    for i, item in enumerate(items):
        active = " active" if i == 0 else ""
        cards.append(
            f'<button class="object{active}" data-video="{html.escape(item["video"])}" '
            f'data-title="{html.escape(item["object"])}" '
            f'data-meta="{html.escape(item["meta"])}" aria-pressed="{str(i == 0).lower()}">'
            f'<img src="{html.escape(item["thumb"])}" alt="" loading="lazy">'
            f'<span><strong>{html.escape(item["object"])}</strong>'
            f'<small>{html.escape(item["meta"])}</small></span></button>'
        )
    first = items[0]
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Figure 8 successful runs</title>
<style>
:root {{ color-scheme: light; font-family: Inter,system-ui,sans-serif; color:#182033; }}
body {{ margin:0; background:#f5f7fb; }}
main {{ width:min(1180px,calc(100% - 32px)); margin:36px auto 60px; }}
h1 {{ margin-bottom:6px; }} .intro {{ color:#5d6676; margin:0 0 24px; }}
.viewer {{ background:#111827; border-radius:14px; overflow:hidden; box-shadow:0 14px 38px #17203324; }}
video {{ display:block; width:100%; aspect-ratio:26/9; background:#090d15; }}
.caption {{ padding:14px 18px; color:white; display:flex; gap:12px; justify-content:space-between; }}
.caption span {{ color:#aab4c5; }}
.objects {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(190px,1fr)); gap:10px; margin-top:18px; }}
.object {{ border:2px solid transparent; border-radius:10px; padding:0; overflow:hidden; background:white;
  color:inherit; cursor:pointer; text-align:left; box-shadow:0 2px 10px #17203312; }}
.object:hover,.object:focus-visible {{ border-color:#f59e0b; outline:none; }}
.object.active {{ border-color:#e8710a; box-shadow:0 2px 14px #e8710a30; }}
.object img {{ width:100%; aspect-ratio:16/9; object-fit:cover; display:block; background:#dfe4ec; }}
.object span {{ display:flex; justify-content:space-between; align-items:baseline; padding:9px 10px; gap:6px; }}
.object small {{ color:#6b7280; white-space:nowrap; }}
</style></head><body><main>
<h1>Figure 8 — successful randomized-goal runs</h1>
<p class="intro">One representative validated success per object. Click an object to play its video.</p>
<section class="viewer"><video id="video" controls playsinline preload="metadata" src="{html.escape(first['video'])}"></video>
<div class="caption"><strong id="title">{html.escape(first['object'])}</strong><span id="meta">{html.escape(first['meta'])}</span></div></section>
<section class="objects" aria-label="Successful objects">{''.join(cards)}</section>
</main><script>
const video=document.querySelector('#video'), title=document.querySelector('#title'), meta=document.querySelector('#meta');
document.querySelectorAll('.object').forEach(button=>button.addEventListener('click',()=>{{
  document.querySelectorAll('.object').forEach(other=>{{other.classList.remove('active');other.setAttribute('aria-pressed','false')}});
  button.classList.add('active'); button.setAttribute('aria-pressed','true');
  video.pause(); video.src=button.dataset.video; video.load(); title.textContent=button.dataset.title; meta.textContent=button.dataset.meta;
  video.play().catch(()=>{{}});
}}));
</script></body></html>"""
    output.write_text(page, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--target-frames", type=int, default=240)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    data, records = collect()
    output_dir = ROOT / "results" / "fig8_success_gallery"
    video_dir, thumb_dir = output_dir / "videos", output_dir / "thumbnails"
    video_dir.mkdir(parents=True, exist_ok=True)
    thumb_dir.mkdir(parents=True, exist_ok=True)
    task_by_object = dict(ORDER)
    selected = []
    for object_name, times in data.items():
        if not times:
            continue
        candidates = [r for r in records if r["object"] == object_name]
        record = min(candidates, key=lambda r: float(r["time_to_goal_s"]))
        source = ROOT / record["log"]
        name = slug(object_name)
        video = video_dir / f"{name}.mp4"
        thumb = thumb_dir / f"{name}.jpg"
        duration = float(record["time_to_goal_s"])
        print(f"rendering {object_name}: {source.relative_to(ROOT)} ({duration:.2f}s)")
        render(source, task_by_object[object_name], video, duration,
               fps=args.fps, target_frames=args.target_frames, force=args.force)
        if args.force or not thumb.exists():
            run(["ffmpeg", "-y", "-loglevel", "error", "-ss", "0.2", "-i", str(video),
                 "-frames:v", "1", "-q:v", "3", str(thumb)])
        selected.append({"object": object_name, "meta": f"{duration:.2f} s",
                         "video": str(video.relative_to(output_dir)),
                         "thumb": str(thumb.relative_to(output_dir))})
    if not selected:
        raise RuntimeError("no successful Figure 8 records found")
    write_gallery(selected, output_dir / "index.html")
    print(f"wrote {output_dir / 'index.html'} ({len(selected)} objects)")


if __name__ == "__main__":
    main()
