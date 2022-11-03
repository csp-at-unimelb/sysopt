"""
Test Problem TP2 from

Herber, Daniel R., and James T. Allison. "Nested and simultaneous solution
strategies for general combined plant and control design problems." Journal of
Mechanical Design 141.1 (2019).

"""

from sysopt.modelling.builders import FullStateOutput
from sysopt.blocks import ConstantSignal
from sysopt import Metadata, Composite
from sysopt.symbolic import Variable, PiecewiseConstantSignal
from sysopt.problems import SolverContext


def dxdt(t, x, u, p):
    k, *_ = p
    return [x[1], -k * x[0] + u]


def x0(p):
    _, xi0, v0 = p
    return [xi0, v0]


plant_metadata = Metadata(
    inputs=['u'],
    states=['x', 'v'],
    outputs=['x', 'v'],
    parameters=['k', 'x0', 'v0']
)


def test_problem_2():

    plant = FullStateOutput(plant_metadata, dxdt=dxdt, x0=x0, name='Plant')
    controller = ConstantSignal(['u'], name='Controller')
    model = Composite(name='model')
    model.components = [plant, controller]
    model.declare_outputs(['x', 'v', 'u'])
    model.wires = [
        (controller.outputs, plant.inputs),
        (plant.outputs, model.outputs[0:2]),
        (controller.outputs, model.outputs[2]),
    ]

    u = PiecewiseConstantSignal('u', 100)
    k = Variable('k')

    constants = {
        plant.parameters['x0']: 0,
        plant.parameters['v0']: -1,
        controller.parameters['u']: u,
        plant.parameters['k']: k
    }
    t_f = 2
    with SolverContext(model=model,
                       t_final=t_f,
                       parameters=constants) as solver:

        _, _, u_t = model.outputs(solver.t)
        x_f, v_f, _ = model.outputs(t_f)

        cost = solver.integral(u_t ** 2)

        constraints = [
            x_f <= 0, x_f >= 0,
            v_f <= 0, v_f >= 0
        ]
        problem = solver.problem(
            arguments=[k, u],
            cost=cost,
            subject_to=constraints
        )

        soln = problem.solve([1, 0])

        k_min, u_min = soln.argmin
        y_final = soln.outputs[0:2, -1]
        y_start = soln.outputs[0:2, 0]
        eps = 1e-6
        k_min_est = 3.55

        assert abs(y_start[0]) < eps,   "Failed to meet initial constraint"
        assert abs(y_start[1] + 1) < eps, "Failed to meet initial constraint"

        assert abs(y_final[0]) < eps, "Failed to meet terminal constraint"
        assert abs(y_final[1]) < eps, "Failed to meet terminal constraint"

        assert abs(k_min - k_min_est) < 1e-2

