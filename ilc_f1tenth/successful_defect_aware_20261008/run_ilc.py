from pathlib import Path
import sys,json,threading,time,hashlib,os,argparse
import numpy as np
package=Path(__file__).resolve().parent
sys.path.insert(0,str(package))
os.chdir(package)
parser=argparse.ArgumentParser()
parser.add_argument('--path',choices=['s_curve','figure_eight'],default='s_curve')
parser.add_argument('--epochs',type=int,default=10)
parser.add_argument('--alpha',type=float,default=.5)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
if args.epochs<1 or not 0<args.alpha<=1:parser.error('epochs must be positive; alpha must be in (0,1]')
if args.output.exists():parser.error('Output directory already exists; use a new directory')
args.output.mkdir(parents=True)
import ilc_f1tenth_ilqr_defect_aware as exp
from fixed_deadline_rollout import rollout
out=args.output;source=package/'references'/f'{args.path}.npz';h=np.load(source);ref=h['reference'];dt=float(h['dt']);car=exp.F110();p=exp._get_tire_params(car);_,u=exp.solve_nmpc_initial_solution(ref,dt);np.savez(out/'initial_nmpc.npz',controls=u)
weights=json.loads((package/'weights.json').read_text());Q,Qf,R=np.array(weights['Q']),np.array(weights['Q_f']),np.array(weights['R']);alphas=[args.alpha]
config=dict(epochs=args.epochs,dt=dt,initial_controls_source=str(source),initial_controls_epoch=0,initialization='Original blend NMPC initialization function on new figure-eight reference',model='9-state blend + previous-trial fixed additive defect',acceptance_model='defect-aware',nominal_role='diagnostic',Q=Q.tolist(),Q_f=Qf.tolist(),R=R.tolist(),alphas=alphas,execution_schedule='fixed_deadline',information_boundary='Public odometry and controller states only')
(out/'config.json').write_text(json.dumps(config,indent=2));states=[];controls=[];metrics=[]
class Node(exp.F1tenth_ILC):
 def __init__(self):self.audit_lock=threading.Lock();super().__init__()
 def pose_callback(self,msg):
  with self.audit_lock:
   super().pose_callback(msg);self.audit_odom_stamp=msg.header.stamp.sec*10**9+msg.header.stamp.nanosec;self.audit_receive_time=time.monotonic()
def execute(node,u):
 if not exp.reset_and_wait(node,ref[0]):raise RuntimeError('Reset failed')
 x=rollout(node,u,ref[0],dt,car['gear_ratio'],car['pole_pairs'],car['lambda'],car['mass'],car['rw'],car['max_steer'])
 if not np.isfinite(x).all():raise RuntimeError('Nonfinite rollout')
 for _ in range(10):node.publish_control(0.,0.);exp.rclpy.spin_once(node,timeout_sec=.01)
 np.savez(out/f'schedule_{len(states)}.npz',**node.schedule_diagnostics)
 return x
def save(x,u):
 states.append(x.copy());controls.append(u.copy());cost=exp.trajectory_cost(x,u,ref,Q,Qf,R);rmse=float(exp.trajectory_rmse(ref,x)[0]);metrics.append(dict(epoch=len(states)-1,measured_cost=cost,position_rmse=rmse))
 np.savez(out/'history.npz',reference=ref,states=np.asarray(states),controls=np.asarray(controls),dt=dt,config=json.dumps(config))
 (out/'metrics.json').write_text(json.dumps(metrics,indent=2));print('MEASURED',json.dumps(metrics[-1]),flush=True)
exp.rclpy.init();node=Node()
try:
 x=execute(node,u);save(x,u)
 for epoch in range(1,args.epochs+1):
  folder=out/f'update_{epoch}';report=exp.prepare_update(x,u,ref,dt,car,p,Q,Qf,R,alphas,folder)
  proposed=np.load(folder/'prepared_candidate.npz')['controls'];old=metrics[-1];x=execute(node,proposed);u=proposed.copy();save(x,u)
  actual=dict(previous_measured=old,new_measured=metrics[-1],actual_cost_improvement=old['measured_cost']-metrics[-1]['measured_cost'],actual_position_rmse_improvement=old['position_rmse']-metrics[-1]['position_rmse'],selection=report['selection'])
  (folder/'actual_metrics.json').write_text(json.dumps(actual,indent=2));print('ACTUAL COMPARISON',json.dumps(actual),flush=True)
 (out/'status.json').write_text(json.dumps(dict(completed=True,rollouts=len(states))))
finally:
 for _ in range(10):node.publish_control(0.,0.);exp.rclpy.spin_once(node,timeout_sec=.01)
 node.destroy_node();exp.rclpy.shutdown()
