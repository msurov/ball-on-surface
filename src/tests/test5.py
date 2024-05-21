import sympy as sy
from sympy.physics.mechanics import dynamicsymbols


def quat_mul(a : sy.Tuple, b : sy.Tuple):
  aw,ax,ay,az = a
  bw,bx,by,bz = b
  cw = aw * bw - ax * bx - ay * by - az * bz
  cx = aw * bx + ax * bw + ay * bz - az * by
  cy = aw * by + ay * bw + az * bx - ax * bz
  cz = aw * bz + az * bw + ax * by - ay * bx
  c = sy.Matrix([cw,cx,cy,cz])
  return c

def quat_mul_vec(q : sy.Tuple, v : sy.Tuple):
  aw,ax,ay,az = q
  bx,by,bz = v
  cw = -ax * bx - ay * by - az * bz
  cx =  aw * bx + ay * bz - az * by
  cy =  aw * by + az * bx - ax * bz
  cz =  aw * bz + ax * by - ay * bx
  c = sy.Matrix([cw,cx,cy,cz])
  return c

def quat_conj(q : sy.Tuple):
  w,x,y,z = q
  c = sy.Matrix([w,-x,-y,-z])
  return c

def rot_quat(angle, axis):
  q = sy.zeros(4,1)
  q[0] = sy.cos(angle/2)
  q[1] = axis[0] * sy.sin(angle/2)
  q[2] = axis[1] * sy.sin(angle/2)
  q[3] = axis[2] * sy.sin(angle/2)
  return q

def dynamics():
  I = sy.eye(3)
  ex = I[:,2]
  ey = I[:,2]
  ez = I[:,2]

  theta = dynamicsymbols('theta', real=True)
  x,y = dynamicsymbols('x y', real=True)
  print(x)
  t, = x.args
  mu = rot_quat(theta, ez)
  q_ = dynamicsymbols('q_w q_x q_y q_z', real=True)
  q = sy.Matrix(q_)
  r = quat_mul(mu, q)
  angvel = 2 * quat_mul(quat_conj(r), r.diff())
  velocity = sy.Matrix([x.diff(), y.diff(), 0])
  energy = angvel.T @ angvel / 2 + velocity.T @ velocity / 2
  genvel = [q[0].diff(), q[1].diff(), q[2].diff(), q[3].diff(), x.diff(), y.diff()]
  impulse = energy.jacobian(genvel)
  impulse.simplify()
  gencoords = [q[0], q[1], q[2], q[3], x, y]
  forces = energy.jacobian(gencoords)
  eqs = impulse.diff(t) - forces
  eqs.simplify()
  sy.pprint(eqs)
  

dynamics()

