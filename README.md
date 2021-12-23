# hypersonics_codesign

### Goals & Philosophy
1. Use Python to construct system models
2. Modular and heirarchical


A model is made up of a number of dynamic constraints $R_i$ along with a cost function $C$::
    
    J(x,u,y,p,t) = int_0^T f(x,u,y,p) dt + J_T(x(T), u(T), y(T), p)
        
    R_i(dx_i', dt', x_i, u_i, y_i, t, p_i) = 0                          # dynamics
    A_i[x_i; u_i; p_i] <= 0                                             # constraints
    c_i(x_i, u_i, y_i, p_i, dx_i, du_i, dy_i, dp_i)                     # cost to evaulate 
    

Examples:
    cart model -> diffeq
    controller -> finite time MPC


Key Features:

    Code gen -> C
    Global search
    Local search
    user-driven parameter tuning
    interface with external applications (eg. CFD solvers)
    interface with actual hardware

    easy to generate plots


Process:

    Within a problem context
    - Define subsystems using python classes.
    - Define relationships between variables
     
    
    
    
    
