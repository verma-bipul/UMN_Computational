# This file will test the Finite elements method on simple neoclassical growth model.


## Setting Parameters
β = 0.962
α = 0.35

A = 1/(β*α)

## Defining the policy function approximation
include("linear_basis.jl")
include("exp_grid.jl")
X_grid = [0.01, 1, 3, 6]
#X_grid = exp_grid(0.001, 6, 30)

c(k::Real, θ::Vector) = sum([θ[i]*ψ(X_grid, k, i) for i in 1:length(X_grid)])    # cⁿ(k) = ∑ᵢθᵢ*ψᵢ(k)

## Defining the Residual equation 
function R(k::Real, θ::Vector)
    k_p = A*k^α - c(k, θ)
    if k_p > 0
        return 1.0 - (c(k, θ)/c(k_p, θ))*A*α*β*(k_p)^(α-1)
    else
        return -1000
    end
end


## Finite Element Glarken Method to minimize the weigthed residual.
# We use the numerical_quadrature submodule for PS1 for numerical integration.

include("num_quadrature.jl")


function weighted_residual(θ::Vector, i::Int)  # wᵢ(θ) =∫ψ(k, i)*R(k, θ)dk
   return Num_quad(k->ψ(X_grid, k, i)*R(k, θ), X_grid[1], X_grid[end], 100) # we can use different methods to evaluate the integrals
end

# In setting up the system of equations we have to make sure that the boundary condition holds.
# We have to set the boundary condition θ_1 = 0 to make sure c(0) = 0
function wres(θ::Vector)
    x = vcat(0, θ)
    return [weighted_residual(x, i) for i in 2:length(X_grid)]  # we are ignoring the residual for i = 1 as θ1=0
end   # we then have a system of 3 eqn in 3 var



## Applying multivar newton root θⱼ₊₁ = θⱼ - ∇G(θⱼ)⁻¹*G(θⱼ)
include("multivar_newt_root.jl")
θ_ini = ones(length(X_grid)-1) .* range(1, stop = 5, length = length(X_grid)-1)
θ_est =  multivar_newt_root(wres, θ_ini)


## Let us try some other root finding algorithm. 
# This is because sometime the Jacobain is not invertibale if the choice is not good

## Using Inbuilt root solver
using NLsolve

function f!(F, θ)
    x = vcat(0, θ)
    for i in 1:length(X_grid)-1
    F[i] = weighted_residual(x, i+1)
    end
end

nlsolve(f!, θ_ini)

## Our estimated policy function 
c_est(k::Real) = c(k, [0; θ_est]) 


## Approximation with increased exponentially increasing grid size

