# Manual onboard policy

From `~/ILC2ILC` on branch `quad`:

```bash
./run_policy.sh --check     # offline checkpoint/plan inference; no ROS node
./run_policy.sh --dry-run   # live state inference, no command publisher
./run_policy.sh             # LIVE: square track, 2.05 laps
./run_policy.sh --plan mocap_figfast_b25 --laps 2.05
```

The launcher sources Foxy, this workspace and env_tight.sh. It runs the primary
v3drBtL_s0 checkpoint through the existing NumPy policy_runtime.Controller at
40 Hz, with the trained slow start (30 A until 1 m/s). Use --policy to select
another checkpoint, e.g. ilc_f1tenth/ilc2real/runs/v3drBtL_s2/policy_final.npz.

Existing hardware stack, valid localization must already be
running; this script never starts or stops those processes. The launch file's
current defaults do not match this room; use server_ip:=172.26.30.22 and
client_ip:=172.26.50.69 when an operator launches it. Do not launch a duplicate.

Raw mocap defaults to /dl_f1tenth/pose; --mocap-topic /NAME/pose overrides it.
Missing or stale mocap, odometry or VESC state prevents startup or stops a trial.
Header timestamps, quaternion validity, position jumps and EKF covariance are
checked as well. Startup requires rest and proximity to the selected track.
Path error beyond 0.5 m, stall, non-finite actions, competing
/drive publishers, a missing /drive subscriber or control deadline failures stop execution.
Finite actions are clipped to the existing hardware limits and logged.
A live launch times out after 5 s; any trial times out after 180 s.
The launcher takes an exclusive lock to reject a second copy (exit code 73).

The terminal prints mode and policy at startup, then speed, path error, steering,
current and lap progress once per second. To also save logs:

```bash
./run_policy.sh 2>&1 | tee ~/manual_policy.log
# another terminal:
tail -f ~/manual_policy.log
```

Ctrl+C in the policy terminal stops it. After live commands have been sent,
it sends zero-duty commands (jerk=1, acceleration=0) for 1 second before exit.
The relay is disabled: the downstream mux timeout handles missing commands; there is no separate relay watchdog. Zero duty removes
drive power; physical braking behavior depends on the configured VESC.
Dry-run creates no drive publisher and never sends stop commands. At rest its
reported control action is the launch action; neural inference still runs each
step, but handover requires measured speed to reach 1 m/s.

No OptiTrack calibration or launch configuration is modified by these scripts.
Offline and isolated dry-run checks do not certify real-car tracking performance.

Onboard middleware: run_opti.sh and run_policy.sh explicitly select Fast DDS
(rmw_fastrtps_cpp) and clear CYCLONEDDS_URI after sourcing workspaces.
Use ./run_opti.sh only after the existing hardware launch has been stopped by
the operator; it supplies this room's Motive and Jetson IPs without editing
the launch file. Do not source scripts/dds_env.sh for onboard operation.

Onboard direct control: opti_launch.py defaults drive_relay:=false.
The manual policy publishes both control and stop messages directly to /drive,
and rejects another publisher on that topic (including a still-running relay).
Stop the old hardware launch yourself before starting the updated stack.

Path placement: square b25 (+0.25 m x), figure-eight b25 (+0.25 m x, +0.50 m y), including mu60/mu80 fallbacks. Dynamics and gains unchanged.
No rectangle/geofence is enforced. Localization, freshness, action validity and path-tracking safety checks remain. Available y extends to approximately 2 m.

Heading-error checks are disabled at startup and during execution.
Deviation remains limited to 0.50 m. Rejections print measured values and thresholds.
