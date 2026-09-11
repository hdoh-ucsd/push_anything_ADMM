#!/usr/bin/env python3
"""Record OBJECT_STATE_SIMULATION vs a fixed goal via pydrake's LCM."""
import argparse, json, math, sys, time
sys.path.insert(0, '/root/.claude/jobs/cfe8fda1/tmp/pylcm')
from dairlib import lcmt_object_state
from pydrake.lcm import DrakeLcm

p = argparse.ArgumentParser()
p.add_argument('--goal', type=float, nargs=3, required=True, help='x y yaw')
p.add_argument('--out', required=True)
p.add_argument('--url', required=True)
p.add_argument('--duration', type=float, default=180.0)
args = p.parse_args()

lc = DrakeLcm(args.url)
out = open(args.out, 'w')
state = {'first_success_t': None, 'last': None, 'n': 0}

def on_msg(data):
    m = lcmt_object_state.decode(data)
    q = m.position[:4]
    x, y = m.position[4], m.position[5]
    yaw = math.atan2(2*(q[0]*q[3]+q[1]*q[2]), 1-2*(q[2]*q[2]+q[3]*q[3]))
    pos_err = math.hypot(x-args.goal[0], y-args.goal[1])
    ang_err = abs(math.remainder(yaw-args.goal[2], 2*math.pi))
    t = m.utime/1e6
    rec = {'t': t, 'x': x, 'y': y, 'yaw': yaw,
           'pos_err': pos_err, 'ang_err': ang_err}
    state['last'] = rec; state['n'] += 1
    if state['n'] % 10 == 0:
        out.write(json.dumps(rec)+'\n'); out.flush()
    if pos_err < 0.02 and ang_err < 0.1 and state['first_success_t'] is None:
        state['first_success_t'] = t
        print(f"SUCCESS t={t:.2f}s pos_err={pos_err:.4f} ang_err={ang_err:.4f}",
              flush=True)

lc.Subscribe('OBJECT_G_shape_video_STATE_SIMULATION', on_msg)
deadline = time.time() + args.duration
while time.time() < deadline:
    lc.HandleSubscriptions(timeout_millis=500)
last = state['last'] or {}
print("FINAL " + json.dumps({'first_success_t': state['first_success_t'],
                             'samples': state['n'], **last}), flush=True)
