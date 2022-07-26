import numpy as np
from sysopt import Metadata, Composite, Block
from sysopt.solver import SolverContext
from sysopt.blocks.builders import FullStateOutput
from sysopt.blocks.common import ConstantSignal
from sysopt.symbolic import Variable, PiecewiseConstantSignal, get_time_variable
from numpy import cos, sin, pi, deg2rad

g = 9.81
mass = 136077


def dxdt(t, states, inputs, parameters):
    x, h, vel, gamma = states
    thrust, lift, drag, alpha = inputs

    return [
        vel * cos(gamma),
        vel * sin(gamma),
        (thrust * cos(alpha) - drag)/mass - g * sin(gamma),
        (thrust * sin(alpha) + lift)/(mass * vel) + (g/vel) * cos(gamma)

    ]


def x0(p):
    return [0, *p]


metadata = Metadata(
    states=['position', 'Altitude', 'Velocity', 'Flight Path Angle'],
    inputs=['thrust', 'lift', 'drag', 'angle of attack'],
    parameters=['Initial Altitude',
                'Initial Velocity', 'Initial Flight Path Angle']
)




class Trimmed2DAeroForces(Block):
    def __init__(self):
        super().__init__(
            Metadata(
                inputs=['Velocity', 'Angle of Attack'],
                outputs=['Lift', 'Drag'],
                parameters=['Air Density',
                            'Reference Area',
                            'CL_a0', 'CL_a', 'CD_0', 'C_kk', 'CL_0'])
        )

    def compute_outputs(self, t, states, algebraics, inputs, parameters):
        rho, a_ref, CL_a0, CL_a, CD_0, C_kk, CL_0 = parameters
        v, alpha = inputs

        alpha_deg = 180 * alpha / pi
        q_A = 0.5 * rho * a_ref * v ** 2

        CL = CL_a0 + CL_a * alpha_deg
        CD = CD_0 + C_kk * (CL - CL_0) ** 2
        return q_A * CL, q_A *CD


def test_hsv_codesign():
    model, sim_constants, codesign_constants  = build_model()

    t_final = 300
    with SolverContext(model,
                       constants=sim_constants,
                       t_final=t_final) as solver:
        soln = solver.integrate(t_final=300)

    # model works!
    # set up codesign problem

    design_variables = [
        PiecewiseConstantSignal(name=key, frequency=1)
        if key.endswith('Angle Of Attack') or key.endswith('Thrust')
        else Variable(name=key)
        for key in sim_constants
        if key not in codesign_constants
    ]
    upper_bounds = [
        deg2rad(15),
        -0.0318,
         4.04E-02
    ]
    lower_bounds = [
        -deg2rad(15),
        -0.0518,
        2.04E-02
    ]

    initial_values = [
        deg2rad(2.4),
        -0.0418,
        3.04E-02
    ]
    constraints = [
        v <= b for v, b in zip(design_variables, upper_bounds)
    ] + [
        v >= b for v, b in zip(design_variables, lower_bounds)
    ]
    distance = model.outputs[0]
    altitude = model.outputs[1]
    with SolverContext(model,
                       constants=codesign_constants,
                       t_final=t_final) as solver:
        cost = -distance(t_final)
        constraints += [
            altitude(solver.t) < 100_000,
            altitude(solver.t) > 10,
            altitude(t_final) < 20_000
        ]

        problem = solver.problem(
            arguments=design_variables,
            cost=cost,
            subject_to=constraints
        )

        initial_cost = problem(initial_values)
        codesign_soln = problem.solve(initial_values)

        assert codesign_soln is not None
        assert codesign_soln.cost < initial_cost
        assert len(codesign_soln.argmin) == len(design_variables)
        aoa, *optimal_params = codesign_soln.argmin
        assert all(
            l <= optimal <= u for l, optimal, u in
            zip(lower_bounds[1:], optimal_params[1:], upper_bounds[1:])
        )


def build_model():
    dynamics = FullStateOutput(metadata=metadata, dxdt=dxdt, x0=x0)
    forces = Trimmed2DAeroForces()
    thrust = ConstantSignal(['Thrust'])
    angle_of_attack = ConstantSignal(['Angle Of Attack'])

    model = Composite()
    model.components = [dynamics, forces, thrust, angle_of_attack]
    model.wires = [
        (dynamics.outputs['Flight Path Angle'], forces.inputs[1]),
        (angle_of_attack.outputs, forces.inputs[0]),
        (thrust.outputs, dynamics.inputs[0]),
        (forces.outputs, dynamics.inputs[1:3]),
        (angle_of_attack.outputs, dynamics.inputs[3]),
        (dynamics.outputs[0:4], model.outputs[0:4]),
        (angle_of_attack.outputs, model.outputs[4])
    ]

    constants = {
        f'{dynamics}/Initial Velocity': 2350,  # m /s
        f'{dynamics}/Initial Flight Path Angle': 0,  # rad
        f'{dynamics}/Initial Altitude': 26000,  # N
        f'{forces}/Air Density': 0.03,  # kg/m3
        f'{forces}/Reference Area': 557,  # m2
        f'{thrust}/Thrust': 1.5e6,
        f'{forces}/CD_0': 3.40E-02,  # -
        f'{forces}/C_kk': 1.95E-02,  # -
        f'{forces}/CL_0': 3.49E-01  # -
    }

    sim_constants = {
        f'{angle_of_attack}/Angle Of Attack': deg2rad(2.4),  # rad
        f'{forces}/CL_a0': -0.0418,  # -
        f'{forces}/CL_a': 3.04E-02,  # 1/deg
    }
    sim_constants.update(constants)

    missing = set(sim_constants.keys()) - set(model.parameters)
    assert not missing, missing
    return model, sim_constants, constants
