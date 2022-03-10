### ECON 8185 HW2
### Bipul Verma
### Oct 2021

## Programming Language: Julia v1.6

## Package Requirements
- LinearAlgebra
- Plots
- Parameters
- NLsolve
- Distributions

## List of Subroutines 
The PS2 folder contains the following files/subroutines
More info on how the subroutine works is commented in the respective files

- riccati_eqn.jl     : calculates the P and F matrices by solving Riccati equation
- vaughan.jl         : calculates P and F matrices using Vaughan Method
- taylor_R_Z.jl      : Gives the second order taylor approximation of objective function around steady state



## List of scripts for estimation

- parameters.jl      : contains different set of parameters for which we run the model
- steady_state.jl   : calculates the steady state values of c, k , h
- UF_functions.jl    : contains the utility , production function
- LQV.jl             : calculates policy functions using LQ approximation


## Dependencies
% use the include("file_name.jl") to load the dependencies before running a script
                                                                
parameters.jl ->  UF_functions.jl -> steady_state.jl -> (LQV.jl)
                                   

## How to get the results

- Run LQV.jl script
- The plots for policy function using Riccati Iteration and Vaughan Method will be saved in the ./figs directory.




