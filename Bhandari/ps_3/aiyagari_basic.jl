
# This file contains codes to solve the basic aiyagri model without labor supply 
# The code has been optimized for Julia

#= Steps:

1. Define the parameters in the household instance
2. Specify the utility functions
3. Import the relevant function
4. Calulate eqbm for different values of r
5. Calculate eqbm r
6. Plot the policy function and  asset distribution 
=#

# This code will run the will run Aiyagari model taking into account code optimizations

## Loading Packages
using Plots, ForwardDiff, Interpolations, LinearAlgebra, Roots, SparseArrays, NLsolve, Parameters, Dierckx
using StatsPlots
using KernelDensity

## Calling our Packages
include("Tauchen.jl")
include("aiyagari_functions.jl")
include("MarkovC_Simulation.jl")

## Defining the household instance 

Household = @with_kw (
           
 β = 0.98,
 μ=1.5,
 frish=2.0,
 δ = 0.075,
 #ψ = 0.0,
 θ = 0.3,

 τy = 0.0,
 ϕg = 0.0,
 ϕb = 0.0,

 Na = 150, #no of points in asset grid
 NaL = 500, # no of grid points used for Q; L for large
 Ne = 5,  # no of points in AR(1) shock
 amin = 0,
 amax = 30,

 ϵ = exp.(Tauchen(0.0, 0.6, 0.3, Ne)[1]), 
 Pϵ = Tauchen(0.0, 0.6, 0.3, Ne)[2],
 a_grid = get_grids(amin, amax, Na, 2),
 a_gridL = get_grids(amin, amax, NaL, 2),
 Amat = [a_grid[i] for i in 1:Na, j in 1:Ne],
 AmatL = [a_gridL[i] for i in 1:NaL, j in 1:Ne],
 ϵmat = [ϵ[j] for i in 1:Na, j in 1:Ne],
 ϵmatL = [ϵ[j] for i in 1:NaL, j in 1:Ne]
   
)

hh = Household()  


## Defining the necessary functions

Uc(c, μ) = c.^(-μ) 
Uc_inv(y, μ) = (y).^(-1/μ) 
n(c, ϵ, w, τy, ψ, frish, μ) = @. 1.0  #((1-τy)*w*ϵ*Uc(c, μ)/ψ)^(1/frish)   This is from the MRS



## Finding all the roots at once
function f!(F, x)  # x= r, 0.0, 0.0, A
    F[1], F[2] = get_aggregates(hh, x[1], 0.0, 0.0, x[2])[[2, 4]]
end



r_eqmb2, A_est =  nlsolve(f!, [0.018, 0.5]).zero  
#   0.01892726141118009, 0.5554533938091648
# Not bad this is fast !!


## Getting final Policies
a_new, c_new, n_new = pol_EGM(hh, r_eqmb2, 0.0, 0.0, A_est; tol=1e-8, maxiter=100_00)

## Getting Plots
get_a_plots(5, hh, a_new)  # plot for asset policy 
#savefig("figs\\endoL_A_dis.png")
get_c_plots(20, hh, c_new)  # plot for consumption policy 
get_n_plots(20, hh, n_new)  # plot for labor policy 

get_wealth_dis_plot(450, hh, a_new)  # plot for the asset distribution at eqbm r and w


## Trying a simulation based method to get the stationary distribution
# The is one agent at each (ϵ, a) grid.

#function get_st_dis_sim(hh, T, a_new)

T = 1000_0 # number of time periods for simulation

function a_pol_sim(hh, a_new, T)
    """
    This function return the simulated ϵ and assets for T periods in form of matrix
        # Returns
        - a_sim :: Matrix of simulated policy, row represnts agents, column time period

        # Input
        - hh :: Household instance
        - a_new :: Policy function for the (a, ϵ) grid
        - T :: Time period of simulation

    """

    @unpack Amat, AmatL Pϵ, NaL, Ne = hh
    Ns = NaL*Ne

    a_pol = get_a_spline(Amat, a_new, Ne)

    ϵ0 = [1, 2, 3, 4, 5]
    ϵ0mat = [ϵ0[j] for i in 1:NaL, j in 1:Ne]

    ϵ_sim = zeros(Ns, T)
    ϵ_sim[:, 1] = vec(ϵ0mat)
    for i in 1:Ns   #simulating shocks for T periods for households i 
        ϵ_sim[i, 2:end] =  markov_simulate(Pϵ, Int(ϵ_sim[i, 1]), T)[2:end]
    end

    a_sim = zeros(Ns, T)
    a_sim[:, 1] = vec(AmatL)
    for t in 2:T
      a_sim[:, t] = [a_pol[Int(ϵ_sim[i, t])](a_sim[i,t-1]) for i in 1:Ns]
    end 

    return a_sim    
end

a_pol_sim_T = a_pol_sim(hh, a_new, 1000_0)[:, end]

function get_st_a_sim(hh, a_pol_sim_T)
    """
    This function calculates the stationary asset distribution

        # Returns
        - λ_a_sim :: Asset distribution vector

        # Input
        - hh :: Household instance
        - a_pol_sim_T :: The vector of policy under the stationary distribution
               
    """
    @unpack AmatL, Pϵ, NaL, Ne = hh
    Ns = NaL*Ne

    λ_st_sim = zeros(Ns)

    for i in 1:Ns
      k = searchsortedlast(AmatL[:, 1], a_pol_sim_T[i]) # return the index of the last value in a_grid less than or equal to a'(a_i, e_j)

      if (0 < k && k <NaL) # this is to adjust for first and last grid points
        k = k
      elseif k == NaL
        k = NaL-1
      else
        k = 1
      end

     wk = (a_pol_sim_T[i] - AmatL[k, 1])/(AmatL[k+1, 1] - AmatL[k, 1])
     wk = min(max(wk, 0.0), 1.0)

     λ_st_sim[k] += (1-wk)/Ns
     λ_st_sim[k+1] +=  wk/Ns
    end

    λ_st_sim = reshape(λ_st_sim, NaL, Ne)
    λ_a_sim = sum(λ_st_sim, dims=2)

    return λ_a_sim
 
end

λ_a_sim = get_st_a_sim(hh, a_pol_sim_T)

plot(hh.AmatL[1:450, 1], λ_a_sim[1:450])

# Let us look at the behviour of the simulated dunction as we increase the number of grid points



#---- METHOD 2 ---#: Graphical : We'll plot the asset demand and supply curve and look at the intersection to get ebm residual

N = get_aggregates(hh, 0.015, 0.0, 0.0, 1.0)[1] + 0.28 # N is independent of r 
r_vals = collect(LinRange(0.001, 0.0198, 20))

roi(x) =@. hh.θ*(x/N)^(hh.θ-1) - hh.δ
@time k_vals = [get_aggregates(hh, r_vals[i], 0.0, 0.0, 1.0)[end] for i in 1:length(r_vals)]

demand = @. roi(k_vals)
labels =  ["demand for capital" "supply of capital"]
plot(k_vals, [demand r_vals], label = labels)
plot!(xlabel = "capital", ylabel = "interest rate")
## Plot looks good
savefig("figs\\basic_aiya_r.png")



