#!/usr/bin/env python3
"""Record franka joints + object pose to JSONL, plus success scoring."""
import argparse, json, math, sys, time
sys.path.insert(0, '/root/.claude/jobs/cfe8fda1/tmp/pylcm')
from dairlib import lcmt_object_state, lcmt_robot_output
from pydrake.lcm import DrakeLcm

p = argparse.ArgumentParser()
p.add_argument('--goal', type=float, nargs=3, required=True)
p.add_argument('--out', required=True)
p.add_argument('--url', required=True)
p.add_argument('--duration', type=float, default=180.0)
p.add_argument('--exit-on-success', action='store_true')
args = p.parse_args()

lc = DrakeLcm(args.url)
out = open(args.out, 'w')
S = {'q': None, 'first_success_t': None, 'n': 0, 'last_write': -1.0, 'last': None}

def on_robot(data):
    m = lcmt_robot_output.decode(data)
    S['q'] = list(m.position[:7])

def on_obj(data):
    m = lcmt_object_state.decode(data)
    q = m.position[:4]; x, y, z = m.position[4], m.position[5], m.position[6]
    yaw = math.atan2(2*(q[0]*q[3]+q[1]*q[2]), 1-2*(q[2]*q[2]+q[3]*q[3]))
    t = m.utime/1e6
    pos_err = math.hypot(x-args.goal[0], y-args.goal[1])
    ang_err = abs(math.remainder(yaw-args.goal[2], 2*math.pi))
    S['n'] += 1
    rec = {'t': t, 'q': S['q'], 'obj': list(q)+[x, y, z],
           'pos_err': pos_err, 'ang_err': ang_err}
    S['last'] = rec
    if S['q'] is not None and t - S['last_write'] >= 0.1:
        out.write(json.dumps(rec)+'\n'); out.flush(); S['last_write'] = t
    if pos_err < 0.02 and ang_err < 0.1 and S['first_success_t'] is None:
        S['first_success_t'] = t
        print(f"SUCCESS t={t:.2f}", flush=True)

lc.Subscribe('FRANKA_STATE_SIMULATION', on_robot)
lc.Subscribe('OBJECT_G_shape_video_STATE_SIMULATION', on_obj)
deadline = time.time() + args.duration
while time.time() < deadline:
    lc.HandleSubscriptions(timeout_millis=500)
    if args.exit_on_success and S['first_success_t'] is not None:
        t_end = time.time() + 5.0
        while time.time() < t_end:
            lc.HandleSubscriptions(timeout_millis=200)
        break
last = S['last'] or {}
last.pop('q', None)
print("FINAL " + json.dumps({'first_success_t': S['first_success_t'],
                             'samples': S['n'], **last}), flush=True)
