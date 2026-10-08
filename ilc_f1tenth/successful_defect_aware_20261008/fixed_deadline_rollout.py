"""Absolute monotonic deadlines; asynchronous public ROS odometry only."""
import time
import threading
import numpy as np
from rclpy.executors import SingleThreadedExecutor


def rollout(node, controls, initial_state, dt, gear_ratio, pole_pairs, lam, mass, rw, max_steer_angle):
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    stop = threading.Event()
    errors = []
    def receive():
        try:
            while not stop.is_set():
                executor.spin_once(timeout_sec=0.002)
        except Exception as error:
            errors.append(error)
    thread = threading.Thread(target=receive, daemon=True)
    thread.start()
    states, actual, scheduled, ages = [], [], [], []
    I, steer, velocity = initial_state[7], initial_state[8], initial_state[3]
    start = time.monotonic()
    def snapshot():
        with node.audit_lock:
            observed = np.array(node.first_6_state, copy=True)
            node.audit_snapshot_stamp = node.audit_odom_stamp
            age = time.monotonic() - node.audit_receive_time
        if errors:
            raise RuntimeError('Odometry executor failed') from errors[0]
        if age > 0.1 or not np.all(np.isfinite(observed)) or abs(observed[5]) > 20:
            raise RuntimeError(f'Invalid or stale odometry: age={age}, state={observed}')
        return np.r_[observed, observed[3]/rw, I, steer], age
    try:
        for k, command in enumerate(controls):
            deadline = start + k*dt
            remaining = deadline - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
            state, age = snapshot()
            states.append(state)
            I += command[0]*dt
            velocity = max(0., velocity + gear_ratio*1.5*pole_pairs*lam*I/(mass*rw)*dt)
            steer = float(np.clip(steer + command[1]*dt, -max_steer_angle, max_steer_angle))
            if not np.isfinite(velocity) or not np.isfinite(steer):
                raise RuntimeError('Invalid command')
            sent = time.monotonic()
            if sent-deadline > dt:
                raise RuntimeError('Missed an entire control period; stopping instead of bursting commands')
            node.publish_control(steer, velocity)
            actual.append(sent); scheduled.append(deadline); ages.append(age)
        remaining = start + len(controls)*dt - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)
        states.append(snapshot()[0])
        node.schedule_diagnostics = dict(actual=actual, scheduled=scheduled, odometry_age=ages,
                                        lateness=np.asarray(actual)-np.asarray(scheduled), dt=dt,
                                        deadline_miss_tolerance=0.001)
        return np.asarray(states)
    finally:
        stop.set()
        thread.join(timeout=1.)
        executor.remove_node(node)
        executor.shutdown()
