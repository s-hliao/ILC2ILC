#!/usr/bin/env python3
"""Foreground policy runner. --dry-run and --check never publish commands."""
import argparse
import math
import signal
import sys
import time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'ilc_f1tenth/ilc2real'))
import bank
import car_model as cm
import policy_runtime as prt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--policy', default=str(ROOT / 'ilc_f1tenth/ilc2real/runs/v3drBtL_s0/policy_final.npz'))
    p.add_argument('--plan', default='mocap_square2fast_b25')
    p.add_argument('--laps', type=float, default=2.05)
    p.add_argument('--mocap-topic', default='/dl_f1tenth/pose')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--check', action='store_true', help='Offline inference check, no ROS node')
    a = p.parse_args()
    if not math.isfinite(a.laps) or a.laps <= 0:
        p.error('laps={} must be finite and > 0'.format(a.laps))
    if Path(a.plan).name != a.plan:
        p.error('plan={!r} must be a plan name without directory components'.format(a.plan))
    if not a.mocap_topic.startswith('/'):
        p.error('mocap_topic={!r} must start with /'.format(a.mocap_topic))
    for f in (Path(a.policy), Path(bank.PLAN_DIR)/(a.plan+'.npz'), Path(bank.PLAN_DIR)/(a.plan+'_K.npy')):
        if not f.is_file():
            p.error('missing checkpoint, plan or cached gains: '+str(f))
    plan, policy = bank.load_plan(a.plan), prt.load_policy(a.policy)
    if not isinstance(policy, list) or not policy or any(not np.isfinite(v).all() for layer in policy for v in layer):
        p.error('expected finite trained MLP weights')
    x = np.r_[plan['xy'][0], plan['theta'][0], plan['z'][0,2:]]
    out = prt.Controller(plan, policy, idx0=0).step(x)
    if np.shape(out['cmd']) != (2,) or not np.isfinite(out['cmd']).all():
        p.error('invalid checkpoint inference')
    print('CHECK OK: policy={} layers={} plan={} neural_cmd={}'.format(a.policy, [w.shape for w,b in policy], a.plan, out['cmd']), flush=True)
    if a.check:
        return 0
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from nav_msgs.msg import Odometry
    from geometry_msgs.msg import PoseStamped
    from vesc_msgs.msg import VescStateStamped
    from ackermann_msgs.msg import AckermannDriveStamped
    stopped = [False]
    rclpy.init(args=[])
    signal.signal(signal.SIGINT, lambda *_: stopped.__setitem__(0, True))
    signal.signal(signal.SIGTERM, lambda *_: stopped.__setitem__(0, True))

    class Runner(Node):
        def __init__(self):
            super().__init__('manual_policy')
            self.x, self.v, self.ctl = None, None, None
            self.rx, self.stamps = {}, {}
            self.bad, self.delta, self.armed = '', 0.0, False
            self.pub = None if a.dry_run else self.create_publisher(AckermannDriveStamped, '/drive', 1)
            self.create_subscription(Odometry, '/odometry/filtered', self.odom, qos_profile_sensor_data)
            self.create_subscription(VescStateStamped, '/sensors/core', self.vesc, qos_profile_sensor_data)
            self.create_subscription(PoseStamped, a.mocap_topic, self.mocap, qos_profile_sensor_data)

        def stamp(self, key, header):
            ns = header.stamp.sec*1000000000 + header.stamp.nanosec
            age = (self.get_clock().now().nanoseconds-ns)/1e9
            if ns <= 0 or age > 0.15 or age < -0.1:
                self.bad = '{} timestamp_ns={}, age_s={:.6f}; require timestamp_ns>0 and age_s in [-0.10,0.15]'.format(key,ns,age); return False
            if ns > self.stamps.get(key,0):
                self.stamps[key], self.rx[key] = ns, time.monotonic()
            return True

        def odom(self,m):
            p,q,t = m.pose.pose.position,m.pose.pose.orientation,m.twist.twist
            v = [p.x,p.y,q.x,q.y,q.z,q.w,t.linear.x,t.linear.y,t.angular.z]
            cov = np.asarray(m.pose.covariance)
            if not np.isfinite(v).all() or not np.isfinite(cov).all() or not 0.9 < np.linalg.norm(v[2:6]) < 1.1:
                self.bad = 'invalid odometry: [x,y,qx,qy,qz,qw,vx,vy,yaw_rate]={}; quaternion_norm={}; covariance={}; require finite values and quaternion_norm in (0.9,1.1)'.format(v,np.linalg.norm(v[2:6]),cov.tolist()); return
            if max(cov[0],cov[7]) > 0.04 or cov[35] > 0.25:
                self.bad = 'localization covariance: var_x={} var_y={} (max 0.04 m^2), var_yaw={} (max 0.25 rad^2)'.format(cov[0],cov[7],cov[35]); return
            yaw = math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))
            x = np.array([p.x,p.y,yaw,t.linear.x,t.linear.y,t.angular.z])
            if self.x is not None:
                dt = max(0,time.monotonic()-self.rx.get('odom',time.monotonic()))
                if np.linalg.norm(x[:2]-self.x[:2]) > 0.15+8*dt:
                    self.bad = 'localization position jump: distance_m={:.6f} > limit_m={:.6f}; dt_s={:.6f}'.format(np.linalg.norm(x[:2]-self.x[:2]),0.15+8*dt,dt); return
                x[2] = self.x[2]+math.atan2(math.sin(yaw-self.x[2]),math.cos(yaw-self.x[2]))
            if self.stamp('odom',m.header): self.x = x

        def vesc(self,m):
            v = np.array([m.state.speed/2*2*math.pi/60/11.82,m.state.avg_iq])
            if not np.isfinite(v).all():
                self.bad = 'invalid VESC state: wheel_speed_rad_s={} current_A={}; require finite values'.format(*v); return
            if self.stamp('vesc',m.header): self.v = v

        def mocap(self,m):
            p,q = m.pose.position,m.pose.orientation
            if not np.isfinite([p.x,p.y,p.z,q.x,q.y,q.z,q.w]).all() or not 0.9 < np.linalg.norm([q.x,q.y,q.z,q.w]) < 1.1:
                self.bad = 'invalid mocap pose: position={} quaternion={} norm={}; require finite values and norm in (0.9,1.1)'.format([p.x,p.y,p.z],[q.x,q.y,q.z,q.w],np.linalg.norm([q.x,q.y,q.z,q.w])); return
            self.stamp('mocap',m.header)

        def health(self):
            if self.bad: return self.bad
            missing = [k for k in ('odom','vesc','mocap') if time.monotonic()-self.rx.get(k,0) > 0.1]
            if missing: return 'missing/stale '+', '.join('{} receive_age_s={:.6f} > 0.100000'.format(k,time.monotonic()-self.rx[k]) if k in self.rx else '{} received=False (required=True)'.format(k) for k in missing)
            if not a.dry_run:
                if self.count_publishers('/drive') != 1: return '/drive publisher_count={} != required 1'.format(self.count_publishers('/drive'))
                if self.count_subscribers('/drive') < 1:
                    return '/drive subscriber_count={} < required 1 (hardware stack/mux)'.format(self.count_subscribers('/drive'))
            return ''

        def send(self,cmd=None):
            if self.pub is None: return
            m = AckermannDriveStamped()
            m.header.stamp = self.get_clock().now().to_msg()
            m.header.frame_id = 'base_link'
            m.drive.steering_angle = self.delta if cmd is None else float(cmd[0])
            m.drive.jerk = 1.0 if cmd is None else 2.0
            m.drive.acceleration = 0.0 if cmd is None else float(cmd[1])
            self.pub.publish(m)

    node,code = Runner(),0
    try:
        print('MODE={} | 40 Hz | laps={} | mocap={} | Ctrl+C stops'.format('DRY RUN: NO PUBLISHER' if a.dry_run else 'LIVE',a.laps,a.mocap_topic),flush=True)
        print('Waiting up to 10 s for fresh odometry, VESC and raw mocap. Live mode publishes directly to /drive and requires a subscriber.',flush=True)
        deadline = time.monotonic()+10
        while not stopped[0]:
            rclpy.spin_once(node,timeout_sec=0.02)
            reason = node.health()
            if not reason: break
            if node.bad or time.monotonic() > deadline: raise RuntimeError(reason)
        if stopped[0]: return 0
        if math.hypot(node.x[3],node.x[4]) > 0.2: raise RuntimeError('start_speed_m_s={:.6f} > limit 0.2'.format(math.hypot(node.x[3],node.x[4])))
        node.ctl = prt.Controller(plan,policy,launch=dict(v_hand=1.0,current=30.0),x0=node.x)
        i = node.ctl.idx
        _,ey,epsi = node.ctl.pl.path_coords(i,node.x)
        distance = float(np.linalg.norm(node.x[:2]-node.ctl.pl.xy[i]))
        violations = []
        if distance > 0.5: violations.append('start_distance_m={:.6f} > limit 0.5'.format(distance))
        if abs(ey) > 0.5: violations.append('start_abs_ey_m={:.6f} > limit 0.5'.format(abs(ey)))
        if violations: raise RuntimeError('; '.join(violations))
        print('START: distance_m={:.6f} ey_m={:.6f}; heading check disabled'.format(distance,ey),flush=True)
        print('READY: starting trial; launch uses 30 A until 1.0 m/s, then neural policy.',flush=True)
        tick,last_log = time.monotonic(),0
        started_at = tick
        while not stopped[0]:
            rclpy.spin_once(node,timeout_sec=0.001)
            now = time.monotonic()
            if now < tick: continue
            if now-tick > 0.1: raise RuntimeError('control_lateness_s={:.6f} > limit 0.1'.format(now-tick))
            tick = now+prt.DT
            reason = node.health()
            if reason: raise RuntimeError(reason)
            x = np.r_[node.x,node.v,node.delta]
            out = node.ctl.step(x)
            invalid = ['{}={}'.format(k,out[k]) for k in ('cmd','mu','obs','err','progress') if not np.isfinite(out[k]).all()]
            if invalid: raise RuntimeError('non-finite policy output: '+ '; '.join(invalid))
            if not a.dry_run and not out['launched'] and now-started_at > 5:
                raise RuntimeError('launch_elapsed_s={:.6f} > limit 5; speed_m_s={:.6f} < handover 1.0'.format(now-started_at,math.hypot(x[3],x[4])))
            if now-started_at > 180:
                raise RuntimeError('trial_elapsed_s={:.6f} > limit 180'.format(now-started_at))
            if abs(out['err'][0]) > 0.5:
                raise RuntimeError('abs_ey_m={:.6f} > limit 0.5'.format(abs(out['err'][0])))
            if out['launched'] and x[3] < 0.3:
                raise RuntimeError('vx_m_s={:.6f} < stall limit 0.3 after handover'.format(x[3]))
            if out['progress'] >= a.laps*node.ctl.pl.L:
                print('Trial complete.',flush=True); break
            cmd = np.asarray(out['cmd'])
            if cmd.shape != (2,):
                raise RuntimeError('invalid action shape: '+str(cmd.shape))
            raw_cmd = cmd.copy()
            cmd = np.clip(cmd, [-float(cm.NOMINAL['s_max']), -25.0], [float(cm.NOMINAL['s_max']), 50.0])
            if now-last_log >= 1 and not np.array_equal(raw_cmd, cmd):
                print('LIMIT: action clipped to trained hardware limits: '+str(cmd),flush=True)
            if time.monotonic()-now > 0.1: raise RuntimeError('inference_elapsed_s={:.6f} > limit 0.1'.format(time.monotonic()-now))
            reason = node.health()
            if reason: raise RuntimeError(reason)
            if stopped[0]: break
            node.delta,node.armed = float(cmd[0]),True
            node.send(cmd)
            if now-last_log >= 1:
                print('{} vx={:.2f} ey={:.3f} steer={:.3f} current={:.2f} laps={:.3f}'.format('DRY' if a.dry_run else 'LIVE',x[3],out['err'][0],cmd[0],cmd[1],out['progress']/node.ctl.pl.L),flush=True)
                last_log = now
    except Exception as e:
        print('STOP: '+str(e),file=sys.stderr,flush=True)
        code = 1
    finally:
        if node.armed and not a.dry_run:
            print('Sending zero-duty stop for 1 second (jerk=1, acceleration=0).',flush=True)
            end = time.monotonic()+1
            while time.monotonic() < end and rclpy.ok():
                node.send()
                time.sleep(0.025)
        node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()
        print('Policy exited; dry-run sends no drive or stop messages.' if a.dry_run else 'Policy exited.',flush=True)
    return code


if __name__ == '__main__':
    sys.exit(main())
