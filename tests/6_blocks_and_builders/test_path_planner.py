from sysopt.symbolic.symbols import symbolic_vector
import numpy as np
from sysopt.blocks.optimal_controllers import PathPlanner
from sysopt.symbolic.symbols import (
    MinimumPathProblem, Variable, Algebraic
)


def test_setup_path_planner():
    x = symbolic_vector('x', 2)
    x0 = symbolic_vector('x(0)', 2)
    u = symbolic_vector('u')
    A = np.array([[-0.5, -1], [1, -0.5]])
    B = np.array([[1, 0]]).T
    dx = A @ x + B @ u

    running_cost = u.T @ u
    terminal_cost = x.T @ x

    x_max = np.array([np.inf, np.inf])
    x_min = np.array([-np.inf, -np.inf])

    problem = MinimumPathProblem(
        state=(x, (x_min, x_max)),
        control=(u, (-1, 1)),
        terminal_cost=terminal_cost,
        running_cost=running_cost,
        initial_state=x0,
        vector_field=dx,
        parameters=x0
    )

    planner = PathPlanner(problem)

    assert planner.metadata.outputs == ['x^(2)_0', 'x^(2)_1', 'u^(1)_0']
    assert not planner.metadata.states
    assert not planner.metadata.inputs
    assert planner.metadata.parameters == ['T', 'x(0)^(2)_0', 'x(0)^(2)_1']

    # Numerical Evaluation

    params_numeric = [5, 1, 0]
    t_numeric = 3
    numerical_result = planner.compute_outputs(
        t_numeric, None, None, None, params_numeric)
    numerical_result = numerical_result.full()
    # test to see the numerical result make sense

    params = Variable('p', (3,))
    t = Variable('t')
    symbolic_result = planner.compute_outputs(
        t, None, None, None, params)

    assert symbolic_result.symbols() == {t, params}

    assert isinstance(symbolic_result, Algebraic)

    call_result = symbolic_result.call({t: t_numeric, params: params_numeric})
    call_result = call_result.full()

    assert (call_result == numerical_result).all()

