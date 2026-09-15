import casadi

def srbd_model():
    x = ca.MX.sym('x', 9)   # state: x, y, phi, vx, vy, omega, omega_w, current, delta
    u = ca.MX.sym('u', 2)   # control rate: curent slew rate, steer rate
    p = ca.MX.sym('p', 5)


def get_ilc_model(self):
