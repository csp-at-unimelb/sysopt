"""
Test Problem TP1 from

Herber, Daniel R., and James T. Allison. "Nested and simultaneous solution
strategies for general combined plant and control design problems." Journal of
Mechanical Design 141.1 (2019).

"""
from sysopt.modelling.builders import FullStateOutput
from sysopt.blocks import ConstantSignal, Gain
from sysopt import Composite, Metadata, PiecewiseConstantSignal
from sysopt.problems import SolverContext, Parameter


def dxdt(t, x, u, p):
    b, _ = p
    return -b * x + u


def x0(p):
    _, xi0 = p
    return xi0


plant_metadata = Metadata(
        inputs=['u'],
        states=['xi'],
        parameters=['b', 'xi_0']
    )
w_c = 1
w_p = 0.3
r = 1
q = 10
xi_0 = 2


def test_problem_1_open_loop():

    open_loop_plant = FullStateOutput(plant_metadata, dxdt=dxdt, x0=x0)
    open_loop_controller = ConstantSignal(outputs=1)

    open_loop_model = Composite(
        components=[open_loop_plant, open_loop_controller]
    )

    open_loop_model.declare_outputs(['x', 'u'])
    open_loop_model.wires = [
        (open_loop_controller.outputs, open_loop_plant.inputs),
        (open_loop_plant.outputs, open_loop_model.outputs['x']),
        (open_loop_controller.outputs, open_loop_model.outputs['u'])
    ]

    constants = {
        f'{str(open_loop_plant)}/xi_0': xi_0
    }
    b = Parameter(open_loop_plant, 0)
    u_in = PiecewiseConstantSignal(open_loop_controller.parameters[0], 100)
    with SolverContext(model=open_loop_model,
                       t_final=25,
                       constants=constants) as solver:

        xi, u = open_loop_model.outputs(solver.t)
        running_cost = q * xi ** 2 + r * u ** 2
        integral = solver.integral(running_cost)
        cost = w_c * integral / xi_0 ** 2 + w_p * b
        problem = solver.problem(
            cost=cost,
            arguments=[b, u_in],
            subject_to=[b >= 0, u_in < 1, u_in > -1]
        )

        soln = problem.solve([1, 0])
        assert soln.cost < 2.32, "Failed to get close to optimum"
        b_min, _ = soln.argmin
        assert 3 < b_min < 4, "Failed to find argmin"


def test_problem_1_closed_loop():

    closed_loop_plant = FullStateOutput(plant_metadata, dxdt=dxdt, x0=x0)
    closed_loop_controller = Gain(channels=1)

    closed_loop_model = Composite(
        components=[closed_loop_plant, closed_loop_controller]
    )

    closed_loop_model.declare_outputs(['x', 'u'])
    closed_loop_model.wires = [
        (closed_loop_controller.outputs, closed_loop_plant.inputs),
        (closed_loop_plant.outputs, closed_loop_controller.inputs),
        (closed_loop_plant.outputs, closed_loop_model.outputs['x']),
        (closed_loop_controller.outputs, closed_loop_model.outputs['u'])
    ]
    k = Parameter(closed_loop_controller, 0)
    b = Parameter(closed_loop_plant, 0)
    constants = {
        f'{str(closed_loop_plant)}/xi_0': xi_0
    }
    with SolverContext(model=closed_loop_model,
                       t_final=25,
                       constants=constants) as solver:
        xi, u = closed_loop_model.outputs(solver.t)
        running_cost = q * xi ** 2 + r * u ** 2
        integral = solver.integral(running_cost)
        cost = w_c * integral / xi_0 ** 2 + w_p * b
        problem = solver.problem(
            cost=cost,
            arguments=[b, k],
            subject_to=[b >= 0, k <= 0]
        )

        soln = problem.solve([1, 1])
        assert soln.cost < 2.3
        b_min, k_min = soln.argmin

        assert -2 < k_min < -1
        assert 3 < b_min < 4
