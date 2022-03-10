

## Parameters
include("parameters.jl")

## 
@unpack β, θ, ρ, σ, ψ, γ_z, γ_n, δ = parameters_bipul()
γ = 5

##
k_grid = collect(range(0.05, 20, length=100)) # this is 1x100 vector
h_grid = collect(range(0.01, 0.99, length=50))  # this is 1x10 vector  # may imporove the grid points later

## This subroutine defines the new utility function 

function U(c::Real, h::Real)
    if c>0 && h<1 && h>0
        return ((c*((1-h)^ψ))^(1-γ))/(1-γ)
    else
        return -Inf
    end    
end

function F_hat(k::Real, h::Real, z::Real)
    if k>0 && h<1 && h>0
        return (k^θ)*(z*h)^(1-θ) + (1-δ)*k
    else
        return -Inf
    end             
end