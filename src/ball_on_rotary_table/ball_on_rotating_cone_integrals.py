from sympy import *
from sympy.physics.mechanics import dynamicsymbols
import numpy as np
from .ball_on_rotating_cone_dynamics import (
    BallOnRotatingConeParameters,
    BallOnRotatingConeDynamics
)


class SymbolicDynamics:
    def __init__(self):
        α = symbols('alpha', real=True)
        k = symbols('k', real=True, positive=True)
        r = symbols('r', real=True, positive=True)
        gravity_accel = symbols('g', real=True, positive=True)

        Ω = symbols('Omega', real=True)
        ρ = symbols('rho', real=True)
        ϕ = symbols('phi', real=True)
        ζρ = symbols('zeta_rho', real=True)
        ζϕ = symbols('zeta_phi', real=True)
        ζn = symbols('zeta_n', real=True)

        sin_α = sin(α)
        cos_α = cos(α)

        f = Matrix([
            0,
            r * cos(α) * ζϕ,
            -r * ζρ / ρ,
            Ω * (k + 1) * cos_α * ζϕ - r * cos_α * ζϕ * ζρ / ρ,
            -gravity_accel*k*sin_α/r + ρ*k*cos_α/r*Ω**2 - (k + 1) * cos_α * Ω * ζρ + r * cos_α * ζρ**2 / ρ + (1 - k) * sin_α * Ω * ζn - r * (1 - k) * sin_α * ζρ * ζn / ρ,
            -sin_α * ζϕ * Ω + sin_α * r * ζϕ * ζρ / ρ
        ])
        g = Matrix([
            1,
            0,
            0,
            (k - 1) * sin_α + k * ρ / r,
            0,
            -cos_α 
        ])

        self.α = α
        self.k = k
        self.r = r
        self.gravity_accel = gravity_accel
        self.Ω = Ω
        self.ρ = ρ
        self.ϕ = ϕ
        self.ζρ = ζρ
        self.ζϕ = ζϕ
        self.ζn = ζn
        self.rhs = [f, g]

def ad(a, b, x):
    R"""
        @brief Lie bracket [a, b]
    """
    return b.jacobian(x) @ a - a.jacobian(x) @ b

def accessability():
    symdyn = SymbolicDynamics()
    f, g = symdyn.rhs
    x = (symdyn.Ω, symdyn.ρ, symdyn.ϕ, symdyn.ζρ, symdyn.ζϕ, symdyn.ζn)
    L1 = ad(f, g, x)
    L1.simplify()
    L2 = ad(f, L1, x)
    L2.simplify()
    # L3 = ad(f, L2, x)
    # L4 = ad(f, L3, x)

    print('f = ', latex(f))
    print('g = ', latex(g))
    print('L_1 = ', latex(L1))
    print('L_2 = ', latex(L2))

def test():
    par = BallOnRotatingConeParameters(
      cone_side_angle = -0.1,
      gravity_accel = 9.81,
      ball_mass = 0.07,
      ball_radius = 0.03
    )
    dyn1 = BallOnRotatingConeDynamics(par)
    ρ = 0.34452
    ϕ = 0.634255
    ζ = np.array([1.24546, 0.214365, 2.63543])
    state = np.concatenate([[ρ,ϕ],ζ])
    Ω = -0.2634534
    dΩ = 686.0
    print(dyn1(state, Ω, dΩ))

    symdyn = SymbolicDynamics()
    f, g = symdyn.rhs
    values = {
        symdyn.α: par.cone_side_angle,
        symdyn.k: par.ball_mass * par.ball_radius**2 / (par.ball_mass * par.ball_radius**2 + par.ball_inertia),
        symdyn.r: par.ball_radius,
        symdyn.gravity_accel: par.gravity_accel,
        symdyn.ρ: ρ,
        symdyn.ϕ: ϕ,
        symdyn.ζρ: ζ[0],
        symdyn.ζϕ: ζ[1],
        symdyn.ζn: ζ[2],
        symdyn.Ω: Ω,
    }

    f_val = f.subs(values)
    g_val = g.subs(values)
    rhs = np.array(f_val + g_val * dΩ, float)
    pprint(rhs.T)

np.set_printoptions(suppress=True)
accessability()
