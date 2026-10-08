from pathlib import Path
import sys,json,numpy as np
sys.path.insert(0,str(Path.cwd()))
from scipy.interpolate import splprep,splev
from params import F110
p=Path(__file__).resolve().parent/'references';car=F110();dt=1/35;L=car['lf']+car['lr'];lr=car['lr']
# Same figure-eight waypoint layout as notebook cells 8/9.
wx=np.array([.5,1.5,2.5,3.5,4.5,3.5,2.5,1.5,.5]);wy=np.array([.5,-.5,.5,1.5,.5,-.5,.5,1.5,.5]);tck,knots=splprep([wx,wy],s=0,per=True)
# Start at crossing waypoint; unwrap one full lap from this parameter.
u=np.linspace(knots[2],knots[2]+1,10001);up=u%1
xy=np.array(splev(up,tck)).T;der=np.array(splev(up,tck,der=1)).T;der2=np.array(splev(up,tck,der=2)).T
curv=(der[:,0]*der2[:,1]-der[:,1]*der2[:,0])/np.linalg.norm(der,axis=1)**3
scale=max(3.,np.max(np.abs(curv))*L/np.tan(.28));xy*=scale;curv/=scale
phi=np.unwrap(np.arctan2(der[:,1],der[:,0]));rot=phi[0];matrix=np.array([[np.cos(rot),np.sin(rot)],[-np.sin(rot),np.cos(rot)]]);xy=(xy-xy[0])@matrix.T;phi-=rot
arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(xy,axis=0),axis=1))];length=arc[-1];accel_time=1.5;nominal_v=2.;N=int(np.ceil((length/nominal_v+accel_time)/dt));t=np.arange(N+1)*dt;T=t[-1];v=np.ones_like(t)*nominal_v;a=np.zeros_like(t)
for mask,z,sign in [(t<accel_time,t/accel_time,1),(t>T-accel_time,(T-t)/accel_time,-1)]:
 v[mask]=nominal_v*(3*z[mask]**2-2*z[mask]**3);a[mask]=sign*nominal_v*6*z[mask]*(1-z[mask])/accel_time
s=np.r_[0,np.cumsum((v[:-1]+v[1:])*.5*dt)];ratio=length/s[-1];v*=ratio;a*=ratio;s*=ratio
rear=np.column_stack([np.interp(s,arc,xy[:,i]) for i in range(2)]);heading=np.interp(s,arc,phi);kappa=np.interp(s,arc,curv);yaw=v*kappa;steer=np.arctan(L*kappa)
cog=rear+lr*np.column_stack([np.cos(heading),np.sin(heading)]);cog-=cog[0]
factor=car['gear_ratio']*1.5*car['pole_pairs']*car['lambda']/(car['mass']*car['rw']);current=a/factor
ref=np.column_stack([cog,heading,v,lr*yaw,yaw,v/car['rw'],current,steer]);assert np.isfinite(ref).all();assert np.max(np.abs(steer))<car['max_steer'];assert np.max(np.abs(cog))<18
np.savez(p/'reference.npz',reference=ref,dt=dt,waypoints=np.column_stack([wx,wy]),arc_length=s)
meta=dict(source='interpolator.ipynb cells8/9 figure-eight waypoints',scale=float(scale),length=float(length),duration=T,samples=N+1,max_speed=float(v.max()),max_steering_deg=float(np.degrees(np.max(np.abs(steer)))),dt=dt,reference='Rear-axle spline lifted to CoG; vx=v,vy=lr*yaw; yaw=v*signed_curvature; current=acceleration/torque conversion; smooth start and stop')
(p/'reference_config.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta))
