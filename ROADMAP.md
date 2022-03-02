Sysopt Roadmap
==============

Scheduled
---------

Release 0.0.3: Documentation and UX.
 - Improve documentation
 - Improve usability tracking of metadata through system.

Release 0.0.4:
 - Improve integrator functionality allowing the user to control tolerance etc.
 - Add common blocks:
    - PID controller
    - Infinite Horizon LQR
 - Implement the cart-pole demo

Release 0.0.5:
 -  

Release 0.1.0: Parametric Optimisation
 - Handle derived parameters, and how derived parameter gradients can be computed.
 - Specify constraints on parameters from within a composite block.
 - Specify an optimisation objective in terms of a functional, as well as path constraints
 - Evaluate a path with respect to the objective function, and take gradients with respect to design parameters
 - Run optimisation cycles with respect to the loss function


Proposed
--------
- Alternative autodiff backends.
- Units tracking
