### ECON 8185 HW1
### Bipul Verma
### Sept 2021

## Programming Language: Julia v1.6

## Package Requirements
- LinearAlgebra
- Plots
- Parameters
- NLsolve
- Distributions

## List of Subroutines 
The PS1 folder contains the following files/subroutines
More info on how the subroutine works is commented in the respective files

- newton_julia.jl    : calculates the root of a single variable function using Newtons Root finding Method
- num_quadrature.jl  : calculates the integral of a function  
- riccati_eqn.jl     : calculates the P and F matrices by solving Riccati equation
- vaughan.jl         : calculates P and F matrices using Vaughan Method
- taylor_R_Z.jl      : Gives the second order taylor approximation of objective function around steady state
- Tauchen.jl         : Gives the Transition Matrix for AR(1) process usign Tauchen's Method


## List of scripts for estimation

- parameters.jl      : contains different set of parameters for which we run the model
- steady_state.jl   : calculates the steady state values of c, k , h
- UF_functions.jl    : contains the utility , production function
- vfi.jl             : calculates the value function, policy functions using vfi
- LQV.jl             : calculates policy functions using LQ approximation
- LQV3.jl               : calculates policy function for state space representation in 1-b-(iii)

## Dependencies
% use the include("file_name.jl") to load the dependencies before running a script
                                   /->  vfi.jl                                  
parameters.jl ->  UF_functions.jl -> steady_state.jl -> (LQV.jl / LQV3.jl)
                                   

## How to get the results

1 (a) VFI
- Run the vfi.jl script
- The plots for value function and policy function will be saved in the ./figs  directory

- Run LQV.jl script
- The plots for policy function using Riccati Iteration and Vaughan Method will be saved in the ./figs directory.




