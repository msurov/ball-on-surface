import casadi as ca
import numpy as np



def test():
    psi = ca.SX.sym('psi', 4) # ball orientation wrt table
    dpsi = ca.SX.sym('dpsi', 4) # ball orientation derivative quat
    theta = ca.SX.sym('theta', 1) # plate orientation angle
    dtheta = ca.SX.sym('dtheta', 1) # plate orientation angle deriv
    ddtheta = ca.SX.sym('ddtheta', 1) # plate orientation angle second deriv
    x = ca.SX.sym('x') # ball center x wrt table
    y = ca.SX.sym('y') # ball center y wrt table
    dx = ca.SX.sym('dx') # ball center x wrt table
    dy = ca.SX.sym('dy') # ball center y wrt table
    I = ca.DM.eye(3)
    ex = I[:,0]
    ey = I[:,1]
    ez = I[:,2]

    gencoords = ca.vertcat(psi, x, y)
    genvels = ca.vertcat(dpsi, x, y)

    phi = ca.vertcat(ca.cos(theta)/2, ez * ca.sin(theta/2)) # table orientation wrt world
    mu = quat_mul(phi, psi) # ball orientation wrt world
    dmu = ca.jtimes(mu, ca.vertcat(theta, psi), ca.vertcat(dtheta, dpsi))
    ball_angvel = 2 * quat_conj_mul(mu, dmu)
    tmp = ca.jacobian(ball_angvel, genvels)
    inertia_mat_1 = tmp.T @ tmp / 2

    p = quat_rot(phi, ca.vertcat(x, y, 0)) # ball position wrt world
    v = ca.jtimes(p, ca.vertcat(x, y, theta), ca.vertcat(dx, dy, dtheta)) # ball velocity wrt world
    tmp = ca.jacobian(v, genvels)
    inertia_mat_2 = tmp.T @ tmp / 2

    inertia_mat = inertia_mat_1 + inertia_mat_2
    
    term1 = ca.jtimes(inertia_mat, gencoords, genvels) @ genvels
    term2 = ca.jtimes(inertia_mat, ca.vertcat(theta, dtheta), ca.vertcat(dtheta, ddtheta)) @ genvels
    term3 = -0.5 * ca.jtimes()


test()
