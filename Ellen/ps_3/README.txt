### ECON 8185 HW3
### Bipul Verma
### Oct 2021

## Programming Language: Julia v1.6

## Package Requirements
- LinearAlgebra
- Plots
- Parameters
- Random
- Distributions
- Optim


## List of Subroutines 
The PS3 folder contains the following files/subroutines
More info on how the subroutine works is commented in the respective files

- kalman_filter.jl     : given the input matrices it gives time series for estimated x, y, v and associated matrices P, F, G, K
- parameter_estimation.jl : applies kalman filter and MLE to process given in HW3
- stationary_P.jl       : subroutine to obtain te stationary P
        

## How to get the results

- Run parameter_estimation.jl (This will save all the plots in ./figs/ subdirectory. This estimates will be displayed as the script runs.) 




