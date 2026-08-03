#!/usr/bin/env bash
# make_run_video.sh — composite run video: Drake scene (left panel) +
# scrolling log panel (right panel), rendered from a results/<RUN>.txt log.
#
# Pipeline (the one the 2026-08-02 video session used, now scripted):
#   1. tools/visualizer/render_log_drake_scene.py  — Drake-scene PNG frames
#      from the run log (camera via PORT_CAM_EYE / PORT_CAM_TARGET env).
#   2. tools/visualizer/paint_log_sidepanel.py     — 2080x720 composite mp4:
#      scene left, sticky-milestone + rolling [STEP]/[GS]/[C3+] log right.
#
# Usage:
#   scripts/make_run_video.sh RUN_NAME [--task push_t] [--stride 2] [--fps 10]
#
#   RUN_NAME     stem under results/ (e.g. tight_goal_p149_finalqp_180s)
#   --task       task name for scene geometry (default: push_t)
#   --stride     sim-tick stride for frame rendering (default: 2)
#   --fps        output mp4 fps (default: 10)
#   --keep-frames  keep the intermediate frames dir (default: delete)
#
# Output: results/RUN_NAME_sidepanel.mp4, plus a copy to the D-drive sink
# /d/projects/ERL/push_anything_ADMM/ when that mount exists.
#
# Camera override example:
#   PORT_CAM_EYE="1.25,-0.75,0.75" PORT_CAM_TARGET="0.45,0.05,0.05" \
#     scripts/make_run_video.sh tight_goal_p149_finalqp_180s

set -euo pipefail

cd "$(dirname "$0")/.."

RUN_NAME="${1:?usage: make_run_video.sh RUN_NAME [--task T] [--stride N] [--fps N]}"
shift

TASK="push_t"
STRIDE=2
FPS=10
KEEP_FRAMES=0
while [ $# -gt 0 ]; do
  case "$1" in
    --task)   TASK="$2";   shift 2 ;;
    --stride) STRIDE="$2"; shift 2 ;;
    --fps)    FPS="$2";    shift 2 ;;
    --keep-frames) KEEP_FRAMES=1; shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

LOG="results/${RUN_NAME}.txt"
OUT="results/${RUN_NAME}_sidepanel.mp4"
[ -f "$LOG" ] || { echo "no such log: $LOG" >&2; exit 1; }
command -v ffmpeg >/dev/null || { echo "ffmpeg not found" >&2; exit 1; }

FRAMES_DIR="$(mktemp -d /tmp/${RUN_NAME}_frames.XXXX)"
cleanup() { [ "$KEEP_FRAMES" = 1 ] || rm -rf "$FRAMES_DIR"; }
trap cleanup EXIT

# Default camera = the p147/p148 session's perspective view; env overrides win.
export PORT_CAM_EYE="${PORT_CAM_EYE:-1.25,-0.75,0.75}"
export PORT_CAM_TARGET="${PORT_CAM_TARGET:-0.45,0.05,0.05}"

echo "[make_run_video] 1/2 rendering scene frames (stride=$STRIDE) -> $FRAMES_DIR"
python3 tools/visualizer/render_log_drake_scene.py "$LOG" \
  --task "$TASK" --stride "$STRIDE" --out-dir "$FRAMES_DIR"

echo "[make_run_video] 2/2 compositing scene + log side panel -> $OUT"
python3 tools/visualizer/paint_log_sidepanel.py \
  --frames-dir "$FRAMES_DIR" --log-path "$LOG" --output "$OUT" --fps "$FPS"

echo "[make_run_video] wrote $OUT ($(du -h "$OUT" | cut -f1))"

SINK="/d/projects/ERL/push_anything_ADMM"
if [ -d "$SINK" ]; then
  cp "$OUT" "$SINK/" && echo "[make_run_video] copied to $SINK/$(basename "$OUT")"
fi
