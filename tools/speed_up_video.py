"""
speed_up_video.py — mimics reference process_lcm_logs.py:478-499.

Reference produces `<name>.mp4` at native fps, then speeds it up by
integer factor (2x, 4x) by keeping every Nth frame.

Usage:
    python tools/speed_up_video.py <input.mp4> <speed_factor>

Example:
    python tools/speed_up_video.py results/tight_goal_p19_240def_run1.mp4 4
    # writes results/tight_goal_p19_240def_run1_4x.mp4

Reference source:
    /root/reference_repos/dairlib_sampling_c3/examples/sampling_c3/process_lcm_logs.py
    lines 478-499 (speed_up_video method).
"""
import sys
import cv2


def speed_up_video(video_filepath: str, speed_factor: int) -> str:
    new_video_path = video_filepath.replace(".mp4", f"_{speed_factor}x.mp4")

    cap = cv2.VideoCapture(video_filepath)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_filepath}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(new_video_path, fourcc, fps, (width, height))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % speed_factor == 0:
            out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"Sped-up video saved to: {new_video_path}")
    return new_video_path


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python tools/speed_up_video.py <input.mp4> <speed_factor>")
        sys.exit(1)
    speed_up_video(sys.argv[1], int(sys.argv[2]))
