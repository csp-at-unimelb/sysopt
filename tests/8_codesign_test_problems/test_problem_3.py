"""
Test Problem TP3 from

Herber, Daniel R., and James T. Allison. "Nested and simultaneous solution
strategies for general combined plant and control design problems." Journal of
Mechanical Design 141.1 (2019).

"""


from sysopt import Metadata, Composite
from sysopt.problems import SolverContext
from sysopt.modelling.builders import FullStateOutput
from sysopt.blocks import ConstantSignal
from sysopt.symbolic import PiecewiseConstantSignal, Variable
from sysopt.backends.casadi import codesign_solver

J = 1


def dxdt(t, x, u, p):
    return [x[1], - p[0] * x[0] / J + u[0]/J]


def x0(p):
    return [0, 0]


plant_metadata = Metadata(
    inputs=['u'],
    states=['x', 'v'],
    parameters=['k']
)


def test_problem_3():
    plant = FullStateOutput(plant_metadata, dxdt, x0)
    controller = ConstantSignal(['u'], name='Controller')
    model = Composite(name='Model', components=[plant, controller])
    model.declare_outputs(['x', 'v', 'u'])
    model.wires = [
        (controller.outputs, plant.inputs),
        (plant.outputs, model.outputs[0:2]),
        (controller.outputs, model.outputs[2])
    ]
    t_f = 2
    k_star = 0.8543
    k = Variable('k')
    u = PiecewiseConstantSignal('u', 100)
    parameters = {
        controller.parameters['u']: u,
        plant.parameters['k']: k
    }

    with SolverContext(model=model,
                       t_final=t_f,
                       parameters=parameters) as solver:
        y_f = model.outputs(solver.t_final)
        cost = -y_f[0]
        constraints = [
            u <= 1,
            u >= -1,
            y_f[1] <= 0,
            y_f[1] >= 0
        ]

        problem = solver.problem(
            arguments=[k, u],
            cost=cost,
            subject_to=constraints
        )
        soln = problem.solve([0, 0])
        k_min, u_min = soln.argmin.values()
        assert abs(k_min - k_star) < 1e-2
