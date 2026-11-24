#!/usr/bin/env python3
"""natnet_watchdog.py (2026-11-04), ON THE CAR, started by car_stack.sh: keeps /natnet_ros streaming.
Why: the NatNet SDK races its own threads inside Connect() and aborts (assert HostPresent; most stack launches on 11-04);
launch respawns the node but never configures the new process, so it sits 'unconfigured' and /dl_f1tenth/pose stays
silent. A watchdog on the ros2 CLI failed (its daemon hung: every query timed out), so this talks to the lifecycle services
directly. Trigger = the symptom: no /dl_f1tenth/pose for NO_MOCAP_S. Then: unconfigured -> configure, inactive -> activate.
Exits when its parent (the stack's launch) is gone."""
import os, sys, time, rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import PoseStamped
from lifecycle_msgs.srv import GetState, ChangeState
from lifecycle_msgs.msg import Transition
NODE, TOPIC, NO_MOCAP_S, GRACE_S = '/natnet_ros', '/dl_f1tenth/pose', 3.0, 8.0
rclpy.init(); n = rclpy.create_node('natnet_watchdog')
last = {'t': 0.0}
n.create_subscription(PoseStamped, TOPIC, lambda m: last.update(t=time.monotonic()), QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
get = n.create_client(GetState, f'{NODE}/get_state'); chg = n.create_client(ChangeState, f'{NODE}/change_state')
log = lambda s: print(f'[natnet watchdog {time.strftime("%H:%M:%S")}] {s}', flush=True)
def call(cli, req, timeout):
    if not cli.wait_for_service(timeout_sec=1.0): return None
    f = cli.call_async(req); t = time.monotonic()
    while not f.done() and time.monotonic() - t < timeout: rclpy.spin_once(n, timeout_sec=0.05)
    return f.result() if f.done() else None
def state():
    r = call(get, GetState.Request(), 3.0); return r.current_state.label if r else None
def transition(tid, timeout):
    r = call(chg, ChangeState.Request(transition=Transition(id=tid)), timeout); return bool(r and r.success)
parent, t0, actions = os.getppid(), time.monotonic(), 0
log(f'watching {NODE} via {TOPIC}')
while rclpy.ok() and os.getppid() == parent:
    end = time.monotonic() + 1.0
    while time.monotonic() < end: rclpy.spin_once(n, timeout_sec=0.05)
    now = time.monotonic()
    if now - t0 < GRACE_S or now - last['t'] < NO_MOCAP_S: continue
    st = state()
    if st == 'unconfigured':
        actions += 1; log(f'no mocap for {now - max(last["t"], t0):.0f} s and {NODE} is unconfigured (a crashed + respawned NatNet): configure')
        ok = transition(Transition.TRANSITION_CONFIGURE, 20.0); log(f'configure {"ok" if ok else "FAILED (the node may have aborted again; it respawns, retrying)"}')
        st = state() if ok else None
    if st == 'inactive':
        ok = transition(Transition.TRANSITION_ACTIVATE, 10.0); log(f'activate {"ok" if ok else "FAILED"}')
    elif st == 'active' and now - last['t'] > 10.0 and int(now) % 10 == 0:
        log(f'{NODE} is active but no mocap for {now - max(last["t"], t0):.0f} s: is Motive tracking the car / streaming?')
    elif st is None and int(now) % 10 == 0:
        log(f'{NODE} not answering (restarting?)')
log('stack gone: exiting')
