#!/usr/bin/env python3
"""
Launch a Meshcat HTML replay in headless Chromium (via Xvfb + CDP) and:
  1. Dump window globals to identify the camera accessor path.
  2. Probe the Three.js scene to understand the coordinate frame.
  3. Test camera repositioning and verify it sticks.

Usage:
    python scripts/discover_meshcat_api.py <html_path>
"""
import subprocess
import shutil
import sys
import time
import os
import json
import signal
import urllib.request
from pathlib import Path


# ── Chromium / Xvfb helpers ──────────────────────────────────────────────────

def find_chromium():
    snap_binary = "/snap/chromium/current/usr/lib/chromium-browser/chrome"
    if os.path.exists(snap_binary):
        return snap_binary
    for name in ("google-chrome", "chromium", "chromium-browser"):
        path = shutil.which(name)
        if path:
            return path
    raise RuntimeError("No chromium found")


def _cdp_get_ws_url(port: int, timeout: float = 60.0) -> str:
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
    import websocket  # noqa
    ws.send(json.dumps({
        "id": cmd_id,
        "method": "Runtime.evaluate",
        "params": {"expression": code, "returnByValue": True},
    }))
    deadline = time.time() + 20
    while time.time() < deadline:
        try:
            msg = json.loads(ws.recv())
            if msg.get("id") == cmd_id:
                return msg.get("result", {}).get("result", {}).get("value")
        except Exception:
            break
    return None


# ── JS snippets ───────────────────────────────────────────────────────────────

_JS_WAIT = "typeof viewer !== 'undefined' && viewer.animator && viewer.animator.actions ? viewer.animator.actions.length : 0"

_JS_CAMERA_STATE = r"""
(function() {
    if (typeof viewer === 'undefined') return 'viewer: UNDEFINED';
    const cam = viewer.camera;
    const ctrl = viewer.controls;
    const lines = ['--- camera state ---'];
    lines.push('camera type: ' + (cam.constructor ? cam.constructor.name : '?'));
    lines.push('isPerspective: ' + !!cam.isPerspectiveCamera);
    lines.push('position: ' + JSON.stringify({x:cam.position.x.toFixed(4),y:cam.position.y.toFixed(4),z:cam.position.z.toFixed(4)}));
    lines.push('up:       ' + JSON.stringify({x:cam.up.x.toFixed(4),y:cam.up.y.toFixed(4),z:cam.up.z.toFixed(4)}));
    if (ctrl) {
        lines.push('controls.target: ' + JSON.stringify({x:ctrl.target.x.toFixed(4),y:ctrl.target.y.toFixed(4),z:ctrl.target.z.toFixed(4)}));
        lines.push('controls.enableDamping: ' + ctrl.enableDamping);
        lines.push('controls.minPolarAngle: ' + ctrl.minPolarAngle.toFixed(4));
        lines.push('controls.maxPolarAngle: ' + ctrl.maxPolarAngle.toFixed(4));
        // Compute current polar angle from position
        const off = {
            x: cam.position.x - ctrl.target.x,
            y: cam.position.y - ctrl.target.y,
            z: cam.position.z - ctrl.target.z
        };
        const r = Math.sqrt(off.x*off.x + off.y*off.y + off.z*off.z);
        const phi = Math.acos(off.y / r);
        lines.push('current polar angle (phi): ' + phi.toFixed(4) + ' rad (' + (phi*180/Math.PI).toFixed(1) + ' deg)');
        lines.push('  (0=top, 90=equatorial, 180=bottom)');
    }
    return lines.join('\n');
})()
"""

# Probe Three.js scene objects to understand the coordinate frame.
# Walks the scene graph and reports world positions of named objects.
_JS_SCENE_PROBE = r"""
(function() {
    if (typeof viewer === 'undefined') return 'viewer: UNDEFINED';
    const scene = viewer.scene;
    if (!scene) return 'viewer.scene: UNDEFINED';
    const lines = ['--- scene probe (world positions of named objects) ---'];
    lines.push('scene type: ' + (scene.constructor ? scene.constructor.name : '?'));
    lines.push('scene.children.length: ' + scene.children.length);

    // Traverse the scene and collect object names + world positions
    const results = [];
    scene.traverse(obj => {
        if (obj.name && obj.name.length > 1) {
            try {
                const wp = new THREE.Vector3();
                obj.getWorldPosition(wp);
                results.push({
                    name: obj.name,
                    type: obj.constructor ? obj.constructor.name : '?',
                    wx: wp.x, wy: wp.y, wz: wp.z
                });
            } catch(e) {}
        }
    });

    // Sort by name and show first 40
    results.sort((a,b) => a.name.localeCompare(b.name));
    lines.push('named objects (first 40):');
    for (const r of results.slice(0, 40)) {
        lines.push('  ' + r.name.padEnd(50) +
                   ' pos(' + r.wx.toFixed(3) + ',' + r.wy.toFixed(3) + ',' + r.wz.toFixed(3) + ')' +
                   ' [' + r.type + ']');
    }
    lines.push('total named: ' + results.length);
    return lines.join('\n');
})()
"""

# Check if THREE is accessible (it may be bundled, not global)
_JS_THREE_CHECK = r"""
(function() {
    const lines = [];
    lines.push('typeof THREE: ' + typeof THREE);
    if (typeof THREE !== 'undefined') {
        lines.push('THREE.REVISION: ' + THREE.REVISION);
        lines.push('THREE.Vector3 available: ' + (typeof THREE.Vector3 !== 'undefined'));
    } else {
        // Try to find THREE in known bundle locations
        const candidates = ['viewer.renderer', 'viewer.scene', 'viewer.camera'];
        for (const c of candidates) {
            try {
                const obj = eval(c);
                if (obj && obj.constructor && obj.constructor.REVISION) {
                    lines.push('Found THREE revision via ' + c + ': ' + obj.constructor.REVISION);
                }
            } catch(e) {}
        }
        lines.push('THREE not global — scene.traverse will need alternative approach');
    }
    return lines.join('\n');
})()
"""

# Traverse scene without THREE.Vector3 (use matrixWorld directly)
_JS_SCENE_PROBE_NO_THREE = r"""
(function() {
    if (typeof viewer === 'undefined') return 'viewer: UNDEFINED';
    const scene = viewer.scene;
    if (!scene) return 'viewer.scene: UNDEFINED';
    const lines = ['--- scene probe (matrixWorld positions) ---'];
    lines.push('scene.children.length: ' + scene.children.length);

    // Simple recursive traversal extracting world translation from matrixWorld
    const results = [];
    const walk = (obj, depth) => {
        if (depth > 8) return;
        if (obj.name && obj.name.length > 1) {
            // matrixWorld elements [12,13,14] are the translation (column-major)
            const m = obj.matrixWorld ? obj.matrixWorld.elements : null;
            if (m) {
                results.push({
                    name: obj.name,
                    type: obj.type || (obj.constructor ? obj.constructor.name : '?'),
                    wx: m[12], wy: m[13], wz: m[14]
                });
            }
        }
        if (obj.children) {
            for (const child of obj.children) walk(child, depth+1);
        }
    };
    walk(scene, 0);

    // Filter to just non-zero positions and interesting names
    const interesting = results.filter(r =>
        Math.abs(r.wx)+Math.abs(r.wy)+Math.abs(r.wz) > 0.01 ||
        r.name.toLowerCase().includes('table') ||
        r.name.toLowerCase().includes('box') ||
        r.name.toLowerCase().includes('push') ||
        r.name.toLowerCase().includes('panda') ||
        r.name.toLowerCase().includes('robot')
    );
    interesting.sort((a,b) => a.name.localeCompare(b.name));

    lines.push('objects with non-zero world position (first 40):');
    for (const r of interesting.slice(0, 40)) {
        lines.push('  ' + r.name.substring(0,50).padEnd(52) +
                   ' pos(' + r.wx.toFixed(3) + ',' + r.wy.toFixed(3) + ',' + r.wz.toFixed(3) + ')');
    }
    lines.push('total with non-zero pos: ' + interesting.length + ' / ' + results.length + ' named total');
    return lines.join('\n');
})()
"""

# Test injecting a top-down camera and verify position is retained after a render tick
_JS_SET_TOPDOWN_AND_VERIFY = r"""
(function() {
    if (typeof viewer === 'undefined') return 'viewer: UNDEFINED';
    const cam = viewer.camera;
    const ctrl = viewer.controls;
    const lines = ['--- top-down camera injection test ---'];

    // In a Y-up Three.js scene, "above" is +Y.
    // camera at (0, 2.5, 0) looks straight down at (0,0,0).
    // But view direction (0,-1,0) is anti-parallel to up=(0,1,0) => DEGENERATE lookAt.
    // Fix: use a tiny Z offset, or change cam.up to (0,0,-1).
    const TARGET_X = 0, TARGET_Y = 0, TARGET_Z = 0;
    const HEIGHT = 2.5;  // metres above scene origin in Three.js Y

    if (ctrl) {
        ctrl.target.set(TARGET_X, TARGET_Y, TARGET_Z);
    }
    cam.position.set(TARGET_X, HEIGHT, TARGET_Z + 0.0001);  // tiny Z offset avoids degenerate lookAt
    cam.up.set(0, 1, 0);   // keep Y-up
    cam.lookAt(TARGET_X, TARGET_Y, TARGET_Z);

    if (ctrl) {
        const wasDamping = ctrl.enableDamping;
        ctrl.enableDamping = false;
        ctrl.update();   // syncs internal spherical to new position
        ctrl.enableDamping = wasDamping;
        if (ctrl.saveState) ctrl.saveState();  // bake as the OrbitControls reset state
    }

    lines.push('AFTER SET:');
    lines.push('  cam.position: x=' + cam.position.x.toFixed(4) +
               ' y=' + cam.position.y.toFixed(4) +
               ' z=' + cam.position.z.toFixed(4));
    lines.push('  cam.up:       x=' + cam.up.x.toFixed(4) +
               ' y=' + cam.up.y.toFixed(4) +
               ' z=' + cam.up.z.toFixed(4));
    if (ctrl) {
        lines.push('  ctrl.target:  x=' + ctrl.target.x.toFixed(4) +
                   ' y=' + ctrl.target.y.toFixed(4) +
                   ' z=' + ctrl.target.z.toFixed(4));
    }

    // Compute polar angle to verify overhead-ness
    if (ctrl) {
        const dx = cam.position.x - ctrl.target.x;
        const dy = cam.position.y - ctrl.target.y;
        const dz = cam.position.z - ctrl.target.z;
        const r = Math.sqrt(dx*dx + dy*dy + dz*dz);
        const phi = Math.acos(Math.max(-1, Math.min(1, dy/r)));
        lines.push('  polar angle: ' + phi.toFixed(4) + ' rad (' + (phi*180/Math.PI).toFixed(2) + ' deg)');
        lines.push('  (0 deg = directly above in Y-up; target: < 5 deg)');
    }

    return lines.join('\n');
})()
"""

# Read camera position 2s after injection (to check if controls reset it)
_JS_READ_CAMERA_POS = r"""
(function() {
    if (typeof viewer === 'undefined') return 'viewer: UNDEFINED';
    const cam = viewer.camera;
    const ctrl = viewer.controls;
    const dx = cam.position.x - (ctrl ? ctrl.target.x : 0);
    const dy = cam.position.y - (ctrl ? ctrl.target.y : 0);
    const dz = cam.position.z - (ctrl ? ctrl.target.z : 0);
    const r = Math.sqrt(dx*dx + dy*dy + dz*dz);
    const phi = Math.acos(Math.max(-1, Math.min(1, dy/r)));
    return JSON.stringify({
        px: cam.position.x.toFixed(4),
        py: cam.position.y.toFixed(4),
        pz: cam.position.z.toFixed(4),
        polar_deg: (phi*180/Math.PI).toFixed(2),
        note: phi < 0.1 ? 'TOP-DOWN OK' : (phi < 0.5 ? 'NEARLY TOP-DOWN' : 'NOT top-down')
    });
})()
"""


def main():
    if len(sys.argv) < 2:
        print("Usage: python discover_meshcat_api.py <html_path>")
        sys.exit(1)

    html_path = Path(sys.argv[1])
    if not html_path.exists():
        print(f"File not found: {html_path}")
        sys.exit(1)

    chromium = find_chromium()
    cdp_port = 9997
    display_num = 97
    url = f"file://{html_path.resolve()}"

    # Kill stale Xvfb
    result = subprocess.run(["ps", "axo", "pid,command"], capture_output=True, text=True)
    for line in result.stdout.splitlines():
        if f"Xvfb :{display_num}" in line:
            try:
                os.kill(int(line.split()[0]), signal.SIGKILL)
            except Exception:
                pass
    time.sleep(0.3)

    inner_script = f"""
{chromium} --no-sandbox --disable-dev-shm-usage \\
    --use-angle=swiftshader \\
    --disable-background-timer-throttling \\
    --disable-backgrounding-occluded-windows \\
    --disable-renderer-backgrounding \\
    --remote-debugging-port={cdp_port} \\
    --remote-allow-origins='*' \\
    --window-size=1280,720 \\
    --kiosk '{url}' > /dev/null 2>&1 &
CHROME_PID=$!
echo $CHROME_PID > /tmp/chrome_{display_num}.pid
wait $CHROME_PID
"""
    print(f"Launching Xvfb :{display_num} + Chromium (CDP :{cdp_port})...", flush=True)
    xvfb = subprocess.Popen(
        ["xvfb-run", f"--server-num={display_num}",
         f"--server-args=-screen 0 1280x720x24 -ac",
         "bash", "-c", inner_script],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

    try:
        import websocket

        ws_url = _cdp_get_ws_url(cdp_port)
        ws = websocket.WebSocket()
        ws.connect(ws_url, origin=f"http://127.0.0.1:{cdp_port}")
        ws.settimeout(15)

        # Wait for animator
        print("Waiting for Meshcat animator (up to 60s)...", flush=True)
        deadline = time.time() + 60
        cmd = 10
        while time.time() < deadline:
            val = _cdp_eval(ws, _JS_WAIT, cmd)
            cmd += 1
            if val and isinstance(val, int) and val > 0:
                print(f"  Animator ready (actions={val})", flush=True)
                break
            time.sleep(0.5)
        else:
            print("  WARNING: animator still not ready", flush=True)

        # Reset + play from t=0 (mirrors main script behaviour)
        _cdp_eval(ws, "viewer.animator.reset(); viewer.animator.play(); 'ok'", cmd); cmd += 1

        sep = "=" * 60
        print(f"\n{sep}\nMESHCAT API DIAGNOSTIC OUTPUT\n{sep}")

        for label, js in [
            ("0 THREE availability",       _JS_THREE_CHECK),
            ("1 camera state (before)",    _JS_CAMERA_STATE),
            ("2 scene objects",            _JS_SCENE_PROBE_NO_THREE),
            ("3 inject top-down + verify", _JS_SET_TOPDOWN_AND_VERIFY),
        ]:
            result_val = _cdp_eval(ws, js.strip(), cmd); cmd += 1
            print(f"\n[{label}]")
            print(result_val if result_val else "<no result>")

        # Wait 1s then re-read camera position to see if controls drifted it back
        print("\n[waiting 1s to check for drift...]", flush=True)
        time.sleep(1.0)
        pos_after = _cdp_eval(ws, _JS_READ_CAMERA_POS.strip(), cmd); cmd += 1
        print(f"\n[4 camera position 1s after injection]\n{pos_after}")

        time.sleep(1.0)
        pos_after2 = _cdp_eval(ws, _JS_READ_CAMERA_POS.strip(), cmd); cmd += 1
        print(f"\n[5 camera position 2s after injection]\n{pos_after2}")

        print(f"\n{sep}")
        ws.close()

    finally:
        xvfb.terminate()
        try:
            xvfb.wait(timeout=5)
        except subprocess.TimeoutExpired:
            xvfb.kill()


if __name__ == "__main__":
    main()
