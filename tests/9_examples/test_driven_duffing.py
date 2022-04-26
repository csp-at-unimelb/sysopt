# @nb.code_cell
import numpy as np

# Used to build custom block
from sysopt import Block, Metadata

# Used to assemble system
from sysopt import Composite
from sysopt.blocks import Gain, Oscillator

# Used to run simulation
from sysopt.solver import SolverContext

# @nb.text_cell
r"""
# The Duffing Oscillator.

(see http://scholarpedia.org/article/Duffing_oscillator)

The Duffing oscillator is a nonlinear system made up of a
harmonic oscillator with a nonlinear restoring force, is a simple 
physical system that can produce chaos, and provides a useful way
to introduce modelling in sysopt.

We assume some familiarity with `python`, and basic understanding of 
inheritance.
 
In what follows, we will:
1. Use the `sysopt` object oriented interface to build a `Block`s for the duffing oscillator.
2. Use the `Composite` block to connect the duffing oscillator block to a signal source and gain stage.
3. Simulate the evolution of the system over.  
   
"""

# @nb.text_cell
r"""
## Modelling the Equations.

The damped and forced duffing system is given by the following equation
$$\ddot{x} + \delta \dot{x} +\beta x +\alpha x^3  = u$$.

where (assuming unit mass).
- $x, \dot{x},\ddot{x}$ are position, velocity and acceleration of the system. 
- $\delta$ is the damping rate (si unit: Hz)
- $\beta$ is the spring constant (si unit: N/m)  
- $\alpha$ is the spring nonlinearity (si unit N/m^3) 
- $u$ is the driving force.

An example mechanical system would be a stiff beam clamped at one end, 
then the variable $x$ would refer to the position of the free end.  

We can model this system as a single `sysopt` block.  

"""


# @nb.code_cell
class DuffingComponent(Block):
    def __init__(self,
                 initial_position=0,
                 initial_velocity=0):
        metadata = Metadata(
            inputs=['force'],
            state=['position', 'velocity'],
            parameters=['damping', 'stiffness', 'nonlinearity'],
            outputs=['position', 'velocity']
        )
        super().__init__(metadata)

        self.x0 = [initial_position, initial_velocity]

    def initial_state(self, parameters):
        return self.x0

    def compute_dynamics(self, t, state, algebraic, inputs, parameters):
        delta, alpha, beta = parameters
        x, dx = state
        u, = inputs
        return [
            dx,
            -delta * dx - alpha * x - beta * x ** 3 + u
        ]

    def compute_outputs(self, t, state, algebraic, inputs, parameters):
        x, dx = state
        return [x, dx]


# @nb.text_cell
r"""
# A breakdown of the class definition.

### Inside the `__init__` function.
The `Metadata` object defines the input, output and state space for the system in question, along with parameters.
This is passed into the `Block` constructor (via the `super().__init__(metadata)` call, and allows 
the base class to construct the appropriate input-output ports for the object.


### `initial_state(self, parameters)`
Any object that has state defined in the metadata, needs to have a way of getting initial values.
Here, we assume that the initial position and velocity are specified upon object creation, so we
can pull those from the object attributes.
More generally, we allow for the initial state to be generate parametrically - hence why, `parameters`
is an argument. 
Initial state is always at $t=0$.

### `compute_*`
If we take each block to have the form 
$$ 
\begin{aligned}
\dot{x} &=& f(t x,z,u; p)\\ y &=& g(t,x,z,u;p)\\ 0 &=& h(t, x,z,u;p)\end{aligned} $$
then 
- `x,z,u,p` are the state variables, algebraic variables, inputs and parameters for this block as defined by the metadata.
- `compute_dynamics` is $f$
- `compute_outputs` is $g$
- `evaluate_residuals` is $h$, but since we have no algebraic terms, we can leave it out. 

"""

# @nb.text_cell
r"""
# Model Composition.

To construct the full model, we want to connect the duffing component to a 
sine wave oscillator with a gain stage.

To do this we use two common blocks: 
- `Oscillator`, a unit-amplitude sine wave source  
- `Gain` a variable-channel gain stage

The general assembly procedure is to first create each sub-component,
add them to the composite model, then specify the input-output relationships
between components. 

"""

# @nb.code_cell
class DuffingSystem(Composite):
    def __init__(self):
        super().__init__()
        self.oscillator = Oscillator()
        self.resonator = DuffingComponent()
        self.gain = Gain(channels=1)

        self.components = [self.resonator, self.gain, self.oscillator]
        self.wires = [
            (self.oscillator.outputs, self.gain.inputs),
            (self.gain.outputs, self.resonator.inputs),
            (self.gain.outputs[0], self.outputs[1]),
            (self.oscillator.outputs[0], self.outputs[0])
        ]


# @nb.text_cell
r"""
The input-output relationships take two forms:
1. Subcomponent output to subcomponent input, defining an 'internal' wire.
2. Composite input to subcomponent input, defining an external input to the composed system.
3. Subcomponent output to composite output, defining an external output variable.  

Here, we have no wires of the second class, which is necessary for us
to be able to simulate the system. 

"""

# @nb.text_cell
r"""
# Simulation

In order to run the simulation, we need to set up the parameters.
This is done by constructing a dictionary of full parameter names and the
corresponding numerical value.
The name can be index directly from the objects parameter attribute,
or specified as a string.  

Once the parameters are fully specified, one can construct a simulation/
solver window (using the `SolverContext` context manager) and then
perform the integration with the `integrate` method.

The output `x_t` is a function that provides cubic interpolation between
solution data points.

"""


# @nb.code_cell
def simulate(t=10):
    duffing_system = DuffingSystem()

    default_parameters = {
        duffing_system.oscillator.parameters[0]: 1,
        duffing_system.oscillator.parameters[1]: 0,
        duffing_system.gain.parameters[0]: 0.3,
        f'{duffing_system.resonator}/damping': 0.2,
        f'{duffing_system.resonator}/stiffness': -1,
        f'{duffing_system.resonator}/nonlinearity': 1,
    }
    # get an integrator
    with SolverContext(duffing_system, t, default_parameters) as solver:
        x_t = solver.integrate(resolution=100)

    return x_t
# @nb.text_cell
r"""
# Results

Lets run the simulation for 10 seconds and plot the results. 

"""

# @nb.code_cell_from_text
r"""
import matplotlib.pyplot as plt
trajectory = simulate(10)
T = np.linspace(0, 10, 50)
x = np.empty(shape=(2,50))
for i,t_i in enumerate(T):
    x[:, i:i+1] = trajectory(t_i)[:, 0]

plt.plot(T, x[0,:], 'r', label='pos')
plt.plot(T, x[1,:], 'b', label='vel') 
ax = plt.gca()
ax.set_xlabel('t')
ax.legend()
ax.set_xlim(0, 10)
ax.set_ylim(-2, 2)
ax.set_title(f'Duffing Trajectory')
plt.show()
"""

# @nb.skip
def test_simulate():
    simulate()
