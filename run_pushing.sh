#!/bin/bash
set -e
mkdir -p results
python main.py pushing --save-video results/run.mp4
