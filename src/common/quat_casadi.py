import casadi as ca


def quat_mul(a, b):
    aw,ax,ay,az = a.elements()
    bw,bx,by,bz = b.elements()
    cw = aw * bw - ax * bx - ay * by - az * bz
    cx = aw * bx + ax * bw + ay * bz - az * by
    cy = aw * by + ay * bw + az * bx - ax * bz
    cz = aw * bz + az * bw + ax * by - ay * bx
    c = ca.vertcat(cw, cx, cy, cz)
    return c

def quat_conj(q):
    return ca.vertcat(q[0], -q[1], -q[2], -q[3])

def quat_conj_mul(a, b):
  aw,ax,ay,az = a.elements()
  bw,bx,by,bz = b.elements()
  cw = aw * bw + ax * bx + ay * by + az * bz
  cx = aw * bx - ax * bw - ay * bz + az * by
  cy = aw * by - ay * bw - az * bx + ax * bz
  cz = aw * bz - az * bw - ax * by + ay * bx
  c = ca.vertcat(cw, cx, cy, cz)
  return c

def get_quat_mat(q):
  w,x,y,z = q.elements()
  Q = ca.vertcat(
    ca.horzcat(w, -x, -y, -z),
    ca.horzcat(x, w, -z, y),
    ca.horzcat(y, z, w, -x),
    ca.horzcat(z, -y, x, w)
  )
  return Q

def get_quat_right_mat(q):
  w,x,y,z = q.elements()
  Q = ca.vertcat(
    ca.horzcat(w, -x, -y, -z),
    ca.horzcat(x, w, z, -y),
    ca.horzcat(y, -z, w, x),
    ca.horzcat(z, y, -x, w)
  )
  return Q

def quat_mul_conj(a, b):
  aw,ax,ay,az = a.elements()
  bw,bx,by,bz = b.elements()
  cw =   aw * bw + ax * bx + ay * by + az * bz
  cx = - aw * bx + ax * bw - ay * bz + az * by
  cy = - aw * by + ay * bw - az * bx + ax * bz
  cz = - aw * bz + az * bw - ax * by + ay * bx
  c = ca.vertcat(cw,cx,cy,cz)
  return c

def quat_mul_vec(q, v):
  aw,ax,ay,az = q.elements()
  bx,by,bz = v.elements()
  cw = -ax * bx - ay * by - az * bz
  cx =  aw * bx + ay * bz - az * by
  cy =  aw * by + az * bx - ax * bz
  cz =  aw * bz + ax * by - ay * bx
  c = ca.vertcat(cw,cx,cy,cz)
  return c

def quat_vec_mul(v, q):
  ax,ay,az = v.elements()
  bw,bx,by,bz = q.elements()
  cw = -ax * bx - ay * by - az * bz
  cx =  ax * bw + ay * bz - az * by
  cy =  ay * bw + az * bx - ax * bz
  cz =  az * bw + ax * by - ay * bx
  c = ca.vertcat(cw,cx,cy,cz)
  return c

def quat_vec_part(q):
  return q[1:4]

def quat_rot(q, v):
  r = quat_mul_conj(quat_mul_vec(q, v), q)
  return quat_vec_part(r)
