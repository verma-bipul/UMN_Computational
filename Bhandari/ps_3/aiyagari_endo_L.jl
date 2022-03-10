# This code will run the will run Aiyagari model taking into account code optimizations

## Loading Packages
using Plots, ForwardDiff, Interpolations, LinearAlgebra, Roots, SparseArrays, NLsolve, Parameters, Dierckx

## Calling our Packages
include("Tauchen.jl")
include("aiyagari_functions.jl")

## Defining the household instance 

Household = @with_kw (
           
 β = 0.98,
 μ=2.0,
 frish=2.0,
 δ = 0.075,
 #ψ = 150.0,
 θ = 0.3,

 τy = 0.4,
 ϕg = 0.2,
 ϕb = 1.0,

 Na = 150, #no of points in asset grid
 NaL = 5000, # no of grid points used for Q; L for large
 Ne = 5,  # no of points in AR(1) shock
 amin = 0,
 amax = 15,

 ϵ = exp.(Tauchen(0.0, 0.6, 0.3, Ne)[1]), 
 Pϵ = Tauchen(0.0, 0.6, 0.3, Ne)[2],
 a_grid = get_grids(amin, amax, Na, 1.5),
 a_gridL = get_grids(amin, amax, NaL, 1.5),
 Amat = [a_grid[i] for i in 1:Na, j in 1:Ne],
 AmatL = [a_gridL[i] for i in 1:NaL, j in 1:Ne],
 ϵmat = [ϵ[j] for i in 1:Na, j in 1:Ne],
 ϵmatL = [ϵ[j] for i in 1:NaL, j in 1:Ne]
   
)

hh = Household()  


## Defining the necessary functions

Uc(c, μ) = c.^(-μ) 
Uc_inv(y, μ) = (y).^(-1/μ) 
n(c, ϵ, w, τy, ψ, frish, μ) = @. ((1-τy)*w*ϵ*Uc(c, μ)/ψ)^(1/frish)  # This is from the MRS


## Finding all the roots at once
function f!(F, x)  # x= r, T, ψ, A
    # Eqn 1:4 calibrates for N = 0.28, Ydd = 1.0, govt_budget_balance, asset market clearing
    F[1], F[2], F[3], F[4] = get_aggregates(hh, x[1], x[2], x[3], x[4])[1:4]
end

#r_eqmb2, T_est, ψ_est, A_est =  nlsolve(f!, [0.019 ; 0.2; 50; 2.0]).zero  
# Not bad this is fast !!

r_eqmb2, T_est, ψ_est, A_est =   0.019351478537570316, 0.1903798363077366, 58.17502452177836, 2.175401629803149


## Getting some Plots
a_new, c_new, n_new = pol_EGM(hh, r_eqmb2, T_est, ψ_est, A_est; tol=1e-8, maxiter=100_00)


## Getting Plots
get_a_plots(900, hh, a_new)  # plot for asset policy 
#savefig("figs\\endoL_A_dis.png")
get_c_plots(20, hh, c_new)  # plot for consumption policy 
get_n_plots(20, hh, n_new)  # plot for labor policy 

get_wealth_dis_plot(2000, hh, a_new)  # plot for the asset distribution at eqbm r and w

