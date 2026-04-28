#!/usr/bin/env python3
"""
Concatenate Meshcat HTML replays into a single labeled MP4 for Slack sharing.

Uses headless Chromium (via Xvfb) + ffmpeg's x11grab to screen-record each
HTML replay, then concatenates the clips with text overlays.

Usage:
    python scripts/html_replays_to_mp4.py \
        --inputs task1.html:"Task 1 - North" \
                 task2.html:"Task 2 - East" \
                 task3.html:"Task 3 - South" \
                 task4.html:"Task 4 - West" \
        --clip-duration 15 \
        --output combined.mp4

Each --input argument is:  <html_path>:<label>
All clips are recorded for --clip-duration seconds (default 15).
Camera preset is controlled by --camera-preset (default: topview).

Camera presets (Three.js Y-up; Drake Z maps to Three.js Y):
  topview     : position=(0, 1.5, 0.3001), target=(0, 0, 0.3)  — original
  topview-v2  : position=(0, 2.5, 0.3001), target=(0, 0, 0.3)  — height-only fix
  tilted      : position=(0, 1.4, 0.5),   target=(0, 0, 0.3)   — angled view
"""
import argparse
import hashlib
import subprocess
import shutil
import time
import os
import json
import urllib.request
from pathlib import Path

DEFAULT_RESOLUTION = "1280x720"
DEFAULT_FPS = 30

# The conda-bundled ffmpeg lacks x11grab; prefer the system one for capture.
_SYSTEM_FFMPEG = "/usr/bin/ffmpeg"
_FFMPEG = _SYSTEM_FFMPEG if os.path.exists(_SYSTEM_FFMPEG) else shutil.which("ffmpeg")

# ── Camera presets ────────────────────────────────────────────────────────────

# Each preset: (pos_x, pos_y, pos_z, target_x, target_y, target_z)
# All use up=(0,1,0) which is standard for a Y-up Three.js scene.
CAMERA_PRESETS = {
    "topview": {
        "position": (0, 1.5, 0.3001),
        "target":   (0, 0,   0.3),
        "description": "Original overhead view, camera Y=1.5",
    },
    "topview-v2": {
        "position": (0, 2.5, 0.3001),
        "target":   (0, 0,   0.3),
        "description": "Height-only fix: camera Y raised from 1.5 to 2.5, XZ unchanged",
    },
    "tilted": {
        "position": (0, 1.4, 0.5),
        "target":   (0, 0,   0.3),
        "description": "Angled view: camera pulled back in Z for mild tilt",
    },
}


def _build_camera_js(preset: dict) -> str:
    px, py, pz = preset["position"]
    tx, ty, tz = preset["target"]
    return f"""
(function() {{
    var cam = viewer.camera;
    var ctrl = viewer.controls;
    ctrl.target.set({tx}, {ty}, {tz});
    cam.position.set({px}, {py}, {pz});
    cam.up.set(0, 1, 0);
    cam.lookAt({tx}, {ty}, {tz});
    ctrl.update();
    if (ctrl.saveState) ctrl.saveState();
    return 'camera_set';
}})()
"""


# ── DrvFS-safe copy with integrity verification ───────────────────────────────

def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_copy_to_drvfs(src: Path, dst: Path) -> None:
    """Copy src → dst (DrvFS), flush OS write buffers, then assert size and md5 match.

    On WSL2 DrvFS, copy2 alone can return before the Windows filesystem has
    flushed metadata. os.sync() drains the kernel write queue before we stat
    and hash, so the assertion is meaningful rather than racing the flush.
    Raises RuntimeError if size or md5 mismatch — never silently passes a
    corrupt or truncated file.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    os.sync()

    src_size = src.stat().st_size
    dst_size = dst.stat().st_size
    if src_size != dst_size:
        raise RuntimeError(
            f"DrvFS copy size mismatch: src={src_size} dst={dst_size}  ({dst})"
        )

    src_md5 = _md5(src)
    dst_md5 = _md5(dst)
    if src_md5 != dst_md5:
        raise RuntimeError(
            f"DrvFS copy md5 mismatch: src={src_md5} dst={dst_md5}  ({dst})"
        )

    print(f"[verify] {dst.name}: {src_size} bytes  md5={dst_md5}  OK", flush=True)


def find_chromium():
    # Prefer the snap Chromium's inner binary — the snap wrapper
    # (chromium-browser) runs under snap confinement and won't render to Xvfb.
    snap_binary = "/snap/chromium/current/usr/lib/chromium-browser/chrome"
    if os.path.exists(snap_binary):
        return snap_binary
    for name in ("google-chrome", "chromium", "chromium-browser"):
        path = shutil.which(name)
        if path:
            return path
    raise RuntimeError("No chromium/chrome found. Install: sudo apt install chromium-browser")


# ── CDP helpers ──────────────────────────────────────────────────────────────

def _cdp_get_ws_url(port: int, timeout: float = 60.0) -> str:
    """Poll CDP HTTP endpoint until the Meshcat page tab appears."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = urllib.request.urlopen(f"http://127.0.0.1:{port}/json", timeout=2)
            tabs = json.loads(resp.read())
            tab = next((t for t in tabs if t.get("type") == "page"), None)
            if tab:
                return tab["webSocketDebuggerUrl"]
        except Exception:
            pass
        time.sleep(0.5)
    raise TimeoutError(f"CDP port {port} not ready after {timeout}s")


def _cdp_eval(ws, code: str, cmd_id: int = 1):
    """Send Runtime.evaluate over an open websocket-client connection."""
    import websocket as _ws_mod  # noqa: F401 — already imported at call site
    ws.send(json.dumps({
        "id": cmd_id,
        "method": "Runtime.evaluate",
        "params": {"expression": code, "returnByValue": True},
    }))
    deadline = time.time() + 15
    while time.time() < deadline:
        try:
            msg = json.loads(ws.recv())
            if msg.get("id") == cmd_id:
                return msg.get("result", {}).get("result", {}).get("value")
        except Exception:
            break
    return None


def _wait_and_play(port: int, camera_preset: dict) -> None:
    """
    Wait for the Meshcat animator to finish loading, then reset+play so the
    animation starts from t=0 when recording begins.

    Background: the embedded animation data has play:true in its options, so
    it auto-starts immediately on page load. By the time the scene is fully
    rendered (~5–15 s), the 8-second animation has already finished and the
    scene is frozen at the last frame. We need to reset+play right before
    recording starts.
    """
    import websocket

    ws_url = _cdp_get_ws_url(port)
    ws = websocket.WebSocket()
    ws.connect(ws_url, origin=f"http://127.0.0.1:{port}")
    ws.settimeout(10)

    # Poll until animation actions are loaded (may take 5–20 s for large HTML)
    print("    [CDP] waiting for animator...", flush=True)
    deadline = time.time() + 60
    while time.time() < deadline:
        val = _cdp_eval(ws, "typeof viewer !== 'undefined' && viewer.animator && viewer.animator.actions ? viewer.animator.actions.length : 0", 1)
        if val and int(val) > 0:
            break
        time.sleep(0.5)

    # reset() puts all actions back to t=0; play() restarts from there
    _cdp_eval(ws, "viewer.animator.reset(); viewer.animator.play(); 'ok'", 2)

    # Set camera according to preset.
    # Three.js scene is Y-up: Drake Z-up maps to Three.js Y.
    camera_js = _build_camera_js(camera_preset)
    result = _cdp_eval(ws, camera_js, 3)
    print(f"    [CDP] camera result: {result}  "
          f"(preset: {camera_preset['description']})", flush=True)

    ws.close()
    print("    [CDP] animation started", flush=True)


# ── Recording ────────────────────────────────────────────────────────────────

def record_one(html_path: Path, output_mp4: Path, duration_s: int, label: str,
               display_num: int, resolution: str, fps: int,
               camera_preset: dict) -> None:
    """Screen-record one HTML replay for `duration_s` seconds.

    ffmpeg writes to /tmp to avoid DrvFS flush issues, then the result is
    verified and copied to output_mp4 via safe_copy_to_drvfs.
    """
    chromium = find_chromium()
    cdp_port = 9900 + display_num
    url = f"file://{html_path.resolve()}"

    label_file = Path(f"/tmp/drawtext_label_{id(output_mp4)}.txt")
    label_file.write_text(label)

    # Write to /tmp first; copy to (potentially DrvFS) destination after verify.
    tmp_mp4 = Path(f"/tmp/clip_{display_num}_{id(output_mp4)}.mp4")

    print(f"  [{label}] launching Xvfb + Chromium...", flush=True)
    # xvfb-run handles WSL2 socket setup. The snap Chromium inner binary is
    # used to bypass snap confinement (wrapper won't render to Xvfb).
    inner_script = f"""
{chromium} --no-sandbox --disable-dev-shm-usage \\
    --use-angle=swiftshader \\
    --disable-background-timer-throttling \\
    --disable-backgrounding-occluded-windows \\
    --disable-renderer-backgrounding \\
    --remote-debugging-port={cdp_port} \\
    --remote-allow-origins='*' \\
    --window-size={resolution.replace('x', ',')} \\
    --kiosk '{url}' > /dev/null 2>&1 &
CHROME_PID=$!
echo $CHROME_PID > /tmp/chrome_{display_num}.pid
# Block here — parent process will kill us when done
wait $CHROME_PID
"""
    xvfb = subprocess.Popen(
        ["xvfb-run", f"--server-num={display_num}",
         f"--server-args=-screen 0 {resolution}x24 -ac",
         "bash", "-c", inner_script],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

    try:
        # Wait for animator to load, then reset+play from t=0
        _wait_and_play(cdp_port, camera_preset)

        # One frame warmup so the freshly-reset scene is composited to X11
        time.sleep(0.3)

        print(f"  [{label}] recording {duration_s}s -> /tmp ...", flush=True)
        display_str = f":{display_num}"
        ffmpeg_cmd = [
            _FFMPEG, "-y",
            "-video_size", resolution,
            "-framerate", str(fps),
            "-f", "x11grab",
            "-i", display_str,
            "-t", str(duration_s),
            "-vf",
            f"drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
            f":textfile={label_file}:fontcolor=white:fontsize=48"
            f":box=1:boxcolor=black@0.6:boxborderw=12:x=24:y=24",
            "-codec:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            str(tmp_mp4),
        ]
        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  [{label}] ffmpeg stderr:\n{result.stderr[-1000:]}")
            raise subprocess.CalledProcessError(result.returncode, ffmpeg_cmd)

    finally:
        xvfb.terminate()
        try:
            xvfb.wait(timeout=5)
        except subprocess.TimeoutExpired:
            xvfb.kill()
        _kill_stale_xvfb([display_num])
        label_file.unlink(missing_ok=True)

    safe_copy_to_drvfs(tmp_mp4, output_mp4)
    tmp_mp4.unlink(missing_ok=True)
    print(f"  [{label}] done -> {output_mp4}", flush=True)


# ── Concat ───────────────────────────────────────────────────────────────────

def concat_mp4s(clip_paths: list, output_mp4: Path) -> None:
    """Concatenate clips via ffmpeg's concat demuxer.

    Writes to /tmp first, then verifies and copies to output_mp4 (DrvFS-safe).
    """
    concat_list = Path("/tmp/mp4_concat_list.txt")
    with open(concat_list, "w") as f:
        for p in clip_paths:
            f.write(f"file '{p.resolve()}'\n")

    tmp_out = Path(f"/tmp/concat_out_{id(output_mp4)}.mp4")
    print(f"Concatenating {len(clip_paths)} clips -> /tmp ...", flush=True)
    subprocess.run(
        [_FFMPEG, "-y", "-f", "concat", "-safe", "0",
         "-i", str(concat_list),
         "-c", "copy", str(tmp_out)],
        check=True,
    )
    concat_list.unlink()

    safe_copy_to_drvfs(tmp_out, output_mp4)
    tmp_out.unlink(missing_ok=True)
    print(f"Done: {output_mp4}", flush=True)


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_input(spec: str) -> tuple:
    # rsplit on last colon so the path can contain colons (e.g. Windows paths).
    parts = spec.rsplit(":", 1)
    if len(parts) != 2:
        raise ValueError(f"Input spec must be 'path:label', got {spec!r}")
    path = Path(parts[0])
    label = parts[1]
    if not path.exists():
        raise FileNotFoundError(f"HTML file not found: {path}")
    return path, label


def _kill_stale_xvfb(display_nums: list[int]) -> None:
    """Kill any leftover Xvfb processes for the displays we're about to use."""
    import signal as _sig
    result = subprocess.run(["ps", "axo", "pid,command"], capture_output=True, text=True)
    for line in result.stdout.splitlines():
        for num in display_nums:
            if f"Xvfb :{num}" in line or f"Xvfb :{num} " in line:
                try:
                    pid = int(line.split()[0])
                    os.kill(pid, _sig.SIGKILL)
                except Exception:
                    pass
    time.sleep(0.3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True,
                        help="List of 'html_path:label' specs")
    parser.add_argument("--clip-duration", type=int, default=15,
                        help="Duration in seconds for each clip (default: 15)")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output concatenated MP4 path")
    parser.add_argument("--resolution", default=DEFAULT_RESOLUTION,
                        help=f"Capture resolution WxH (default {DEFAULT_RESOLUTION})")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS,
                        help=f"Capture FPS (default {DEFAULT_FPS})")
    parser.add_argument("--keep-clips", action="store_true",
                        help="Keep individual per-task MP4 clips")
    parser.add_argument(
        "--camera-preset",
        choices=list(CAMERA_PRESETS.keys()),
        default="topview",
        help=(
            "Camera configuration preset (default: topview). "
            "topview: original (Y=1.5); "
            "topview-v2: height-only fix (Y=2.5, XZ unchanged); "
            "tilted: angled view (Y=1.4, Z=0.5)"
        ),
    )
    args = parser.parse_args()

    preset = CAMERA_PRESETS[args.camera_preset]
    print(f"Camera preset: {args.camera_preset!r} — {preset['description']}", flush=True)

    specs = [parse_input(s) for s in args.inputs]
    clips_dir = args.output.parent / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    clip_paths = []
    base_display_num = 99
    display_nums = [base_display_num + i for i in range(len(specs))]
    _kill_stale_xvfb(display_nums)

    for i, (html, label) in enumerate(specs):
        clip_path = clips_dir / f"clip_{i+1:02d}_{label.lower().replace(' ', '_')}.mp4"
        print(f"\n=== Clip {i+1}/{len(specs)}: {html.name} ({args.clip_duration}s, label={label}) ===")
        record_one(
            html_path=html,
            output_mp4=clip_path,
            duration_s=args.clip_duration,
            label=label,
            display_num=base_display_num + i,
            resolution=args.resolution,
            fps=args.fps,
            camera_preset=preset,
        )
        clip_paths.append(clip_path)

    print()
    concat_mp4s(clip_paths, args.output)

    if not args.keep_clips:
        for p in clip_paths:
            p.unlink()
        clips_dir.rmdir()
        print("Cleaned up individual clips")

    size_mb = args.output.stat().st_size / (1024 * 1024)
    print(f"\nFinal MP4: {args.output} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
