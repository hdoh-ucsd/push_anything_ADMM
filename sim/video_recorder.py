"""
Top-down 2D experiment recorder → MP4.

Collects (sim_time, ee_xy, obj_xy) at each control step, then renders
a top-down view of the table workspace and encodes it to MP4.

Requirements
------------
    conda install -c conda-forge ffmpeg
    pip install matplotlib          # usually already present with Drake

Usage
-----
    from sim.video_recorder import ExperimentRecorder

    recorder = ExperimentRecorder(
        output_path="results/pushing.mp4",
        fps=30,
        task_name="pushing",
        goal_xy=[0.3, 0.0],
        obj_shape="box",   # or "sphere"
        obj_size=0.10,     # side length (box) or diameter (sphere) in metres
    )

    # Inside the main simulation loop:
    recorder.record(sim_time, ee_xy_2d, obj_xy_2d)

    # After the loop:
    recorder.save()        # blocks while rendering, then writes to disk
"""

from __future__ import annotations

from pathlib import Path
import numpy as np


class ExperimentRecorder:
    """
    Lightweight frame buffer + MP4 encoder for C3+ experiments.

    Parameters
    ----------
    output_path : str | Path
        Destination file (parent dirs created automatically).
    fps         : int    Frames per second in the output video.
    task_name   : str    Shown in the video title bar.
    goal_xy     : array  2D goal position [x, y] in world frame.
    obj_shape   : str    "box" or "sphere" — controls the rendered patch.
    obj_size    : float  Characteristic size in metres:
                         side length for box, diameter for sphere.
    """

    # Top-down viewport bounds (world metres, X horizontal / Y forward)
    _XLIM = (-0.75,  0.75)
    _YLIM = (-0.80,  0.80)
    _ROBOT_BASE = np.array([0.0, -0.6])   # arm base weld position

    def __init__(
        self,
        output_path: str | Path,
        fps: int = 30,
        task_name: str = "pushing",
        goal_xy: list | np.ndarray | None = None,
        obj_shape: str = "box",
        obj_size: float = 0.10,
    ) -> None:
        self._path      = Path(output_path)
        self._fps       = max(1, int(fps))
        self._task_name = task_name
        self._goal_xy   = np.array(goal_xy if goal_xy is not None else [0.3, 0.0])
        self._obj_shape = obj_shape
        self._obj_size  = float(obj_size)

        self._times:   list[float]      = []
        self._ee_xys:  list[np.ndarray] = []
        self._obj_xys: list[np.ndarray] = []

    # ------------------------------------------------------------------
    # Data collection
    # ------------------------------------------------------------------

    def record(
        self,
        sim_time: float,
        ee_xy: np.ndarray,
        obj_xy: np.ndarray,
    ) -> None:
        """
        Append one state snapshot.  Call once per simulation control step.

        Parameters
        ----------
        sim_time : float       Simulation time in seconds.
        ee_xy    : (2,) array  End-effector XY in world frame.
        obj_xy   : (2,) array  Object (box/ball) XY in world frame.
        """
        self._times.append(float(sim_time))
        self._ee_xys.append(np.asarray(ee_xy, dtype=float).copy())
        self._obj_xys.append(np.asarray(obj_xy, dtype=float).copy())

    @property
    def num_frames(self) -> int:
        return len(self._times)

    # ------------------------------------------------------------------
    # Rendering & encoding
    # ------------------------------------------------------------------

    def save(self, dpi: int = 120) -> None:
        """
        Render all buffered frames and write to MP4.

        Requires ffmpeg to be on PATH (conda install -c conda-forge ffmpeg).
        Blocks until encoding finishes (a few seconds for typical runs).

        Parameters
        ----------
        dpi : int  Pixel density for each rendered frame (default 120).
        """
        if not self._times:
            print("[VideoRecorder] No frames buffered — nothing to save.")
            return

        import subprocess
        import matplotlib
        matplotlib.use("Agg")              # non-interactive, safe for scripts
        # Import only non-pyplot modules to avoid filesystem issues with pyplot.py
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        import matplotlib.patches as mpatches

        ee_xys  = np.array(self._ee_xys)
        obj_xys = np.array(self._obj_xys)
        times   = np.array(self._times)
        n       = len(times)
        trail   = max(1, int(self._fps * 1.2))   # 1.2-second fading trail

        dist_init = float(np.linalg.norm(obj_xys[0] - self._goal_xy)) + 1e-9

        # ---- figure setup ------------------------------------------------
        fig = Figure(figsize=(7, 8), facecolor="#12121e", dpi=dpi)
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)
        ax.set_facecolor("#1a1a2e")
        ax.set_xlim(*self._XLIM)
        ax.set_ylim(*self._YLIM)
        ax.set_aspect("equal")
        ax.set_xlabel("X (m)", color="#ccccdd", fontsize=9)
        ax.set_ylabel("Y (m)", color="#ccccdd", fontsize=9)
        ax.tick_params(colors="#888899", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333355")

        # -- static geometry --

        # Table surface
        table = mpatches.FancyBboxPatch(
            (-1.0, -1.0), 2.0, 2.0,
            linewidth=1.5, edgecolor="#555577",
            facecolor="#22223a",
            boxstyle="round,pad=0.02", zorder=0,
        )
        ax.add_patch(table)

        # Robot base marker
        ax.plot(
            *self._ROBOT_BASE, "o",
            color="#7799ff", markersize=11, zorder=3,
            label="Robot base", markeredgecolor="#aabbff", markeredgewidth=0.8,
        )

        # Goal star + dashed ring
        ax.plot(
            *self._goal_xy, "*",
            color="#00ee88", markersize=20, zorder=4,
            label="Goal", markeredgecolor="white", markeredgewidth=0.4,
        )
        ax.add_patch(mpatches.Circle(
            self._goal_xy, 0.05,
            color="#00ee88", fill=False, linewidth=1.2,
            linestyle="--", zorder=3, alpha=0.45,
        ))
        ax.annotate(
            "goal",
            xy=self._goal_xy,
            xytext=(self._goal_xy[0] + 0.06, self._goal_xy[1] + 0.07),
            color="#00ee88", fontsize=8, zorder=5, alpha=0.8,
        )

        # -- dynamic elements (updated per frame) --

        ee_trail,  = ax.plot([], [], color="#4488ff", lw=1.4,
                             alpha=0.55, zorder=5)
        obj_trail, = ax.plot([], [], color="#ff6644", lw=1.4,
                             alpha=0.55, zorder=5)
        ee_dot,    = ax.plot([], [], "o", color="#88bbff",
                             markersize=9, zorder=8, label="End-effector")

        # Object patch (box → FancyBboxPatch, sphere → Circle)
        half = self._obj_size / 2.0
        if self._obj_shape == "sphere":
            obj_patch = mpatches.Circle(
                (0.0, 0.0), half,
                color="#ff6644", zorder=7, label="Ball",
            )
        else:
            obj_patch = mpatches.FancyBboxPatch(
                (-half, -half), self._obj_size, self._obj_size,
                linewidth=1, edgecolor="#ffaa88", facecolor="#ff6644",
                boxstyle="round,pad=0.006", zorder=7, label="Box",
            )
        ax.add_patch(obj_patch)

        # HUD text
        dist_text = ax.text(
            0.03, 0.97, "",
            transform=ax.transAxes, color="white", fontsize=9,
            va="top", family="monospace",
            bbox=dict(facecolor="#00000055", edgecolor="none", pad=3),
        )
        time_text = ax.text(
            0.97, 0.97, "",
            transform=ax.transAxes, color="#9999bb", fontsize=9,
            va="top", ha="right", family="monospace",
        )
        ax.set_title(
            f"C3+ MPC  ·  {self._task_name.replace('_', ' ').title()}",
            color="white", fontsize=11, pad=9, fontweight="bold",
        )
        ax.legend(
            loc="lower right", fontsize=7.5,
            facecolor="#1e1e30", edgecolor="#444466",
            labelcolor="white", markerscale=0.85,
        )

        # ---- encoding loop (direct ffmpeg pipe, no pyplot dependency) -----
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if self._path.exists():
            self._path.unlink()
            print(f"[VideoRecorder] Overwriting existing file: {self._path}")
        print(f"[VideoRecorder] Rendering {n} frames → {self._path} ...")

        canvas.draw()
        buf = canvas.buffer_rgba()
        frame_arr = np.frombuffer(buf, dtype=np.uint8)
        h, w = int(fig.get_figheight() * dpi), int(fig.get_figwidth() * dpi)

        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{w}x{h}", "-pix_fmt", "rgba", "-r", str(self._fps),
            "-i", "-",
            "-vcodec", "libx264", "-pix_fmt", "yuv420p",
            "-b:v", "2000k",
            str(self._path),
        ]
        proc = subprocess.Popen(
            ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL
        )

        try:
            for i in range(n):
                lo = max(0, i - trail)

                # Trails
                ee_trail.set_data(ee_xys[lo : i + 1, 0],
                                  ee_xys[lo : i + 1, 1])
                obj_trail.set_data(obj_xys[lo : i + 1, 0],
                                   obj_xys[lo : i + 1, 1])

                # EE dot
                ee_dot.set_data([ee_xys[i, 0]], [ee_xys[i, 1]])

                # Object patch position
                ox, oy = obj_xys[i]
                if self._obj_shape == "sphere":
                    obj_patch.set_center((ox, oy))
                else:
                    obj_patch.set_x(ox - half)
                    obj_patch.set_y(oy - half)

                # HUD
                dist = float(np.linalg.norm(obj_xys[i] - self._goal_xy))
                pct  = max(0.0, 100.0 * (1.0 - dist / dist_init))
                dist_text.set_text(f"dist : {dist:.3f} m\nprog : {pct:.0f}%")
                time_text.set_text(f"t = {times[i]:.2f} s")

                canvas.draw()
                buf = canvas.buffer_rgba()
                proc.stdin.write(buf)

        finally:
            proc.stdin.close()
            proc.wait()

        fig.clf()
        print(f"[VideoRecorder] Done → {self._path}  ({n} frames, {self._fps} fps)")
