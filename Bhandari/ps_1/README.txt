### ECON 8185-002 HW1
### Bipul Verma
### Nov 2021

## Programming Language: Julia v1.6

## Package Requirements
- LinearAlgebra
- Plots
- Parameters
- NLsolve
- CSV
- Optim 
- Distributions

## List of Subroutines 
The PS1 folder contains the following files/subroutines
More info on how the subroutine works is commented in the respective files

- kalman.jl : This subroutine implemets the kalman filter on any stochatic process which can be written in the state-space form.
- stationary_P.jl : This subroutine is used in an intermediate step of EM Algorithm


## List of scripts for estimation

- kalman_EM.jl : This is the main script to estimate AR(1) paramters using Kalman Filter and EM.
- plots.jl : This script is used to get the plots 



## How to get the results

- Run kalman_EM.jl 



