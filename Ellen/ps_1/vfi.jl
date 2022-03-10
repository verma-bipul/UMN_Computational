# Date: Sep 14, 2021
# Bipul Verma

## Loading Packages
using Distributions  # Required in constructing the markov process using normal density
using Plots         # May use this later to graph value fns / policy fns
using LinearAlgebra # for opnorm
using Interpolations # To be used later for simulation 
using Parameters

include("UF_functions.jl")
include("Tauchen.jl")


#= Loading Parameters and functions
include("parameters.jl")
using Parameters
@unpack β, θ, ρ, σ, ψ, γ_z, γ_n, δ   = parameters_bipul()
=#

household = @with_kw (
    β = 0.95,   # discount parameter
    θ= 0.34,  # capital share in production

    ρ  = 0.50,    # AR(1) coeff
    σ  = 0.05,     # variance of random shocks

    ψ= 1.6,  # labor supply coeff
    γ_z = 0.02, # productivity growth parameter     
    γ_n= 0.02,  # labor growth parameter
    δ = 0.05,  # depreciation

    z_dis  = exp.(Tauchen(ρ, σ, 3, 5)[1]), 
    Pz = Tauchen(ρ, σ, 3, 5)[2],

    k_grid = collect(range(0.05, 6, length=200)), # this is 1x100 vector
    h_grid = collect(range(0.1, 0.8, length=20)) 

)

hh = household()



## Defining the expectation operator
# This function calculates the expected value of the value function based on the transition matrix

function Expected_v(k_prime, z, Value_func, P)
    index_z = findfirst(isequal(z), z_dis)
    index_k = findfirst(isequal(k_prime), k_grid)
    return sum([Value_func[index_k, j]*P[index_z, j] for j in 1:length(z_dis) ] ) 
end


## We are definign a function G(.) = u(c, h) + β*Exp[v(k',z')]
# We would want to max G(.) wrt to {k_prime, h} later
function G(k, k_prime, h, z, V, P)
    c = F_hat(k, h, z) - k_prime*(1+γ_z)*(1+γ_n)
    if c > 0
        return U(c, h) + β*(1+γ_n)*Expected_v(k_prime, z, V,P)
    else
        return -100.0
    end
end


## VFI Algorithm

   @unpack β, θ, ρ, σ, ψ, γ_z, γ_n, δ, z_dis, Pz, k_grid, h_grid = hh

   v_old = [k_grid[i] for i in 1:length(k_grid), j in 1:length(z_dis)] # Initial guess to begin the iteration
   v_new = zeros(length(k_grid), length(z_dis))  # We'll fill this up with new guess of the policy function at each iteration

   k_policy = copy(v_new) # We'll fill in values for the maximizer a for each k,z : policy fn k'(k,z)
   h_policy = copy(v_new)

   normdiff = 100.0
   iter = 1
   tol = 1e-3
   maxiter = 100

function VFI(maxiter, hh)
    @unpack β, θ, ρ, σ, ψ, γ_z, γ_n, δ, z_dis, Pz, k_grid, h_grid = hh

     maxiter = maxiter
     normdiff = 100.0
     iter = 1
     tol = 1e-3
     v_old = zeros(length(k_grid), length(z_dis))

     v_new = zeros(length(k_grid), length(z_dis))  # We'll fill this up with new guess of the policy function at each iteration

     k_policy = copy(v_new) # We'll fill in values for the maximizer a for each k,z : policy fn k'(k,z)
     h_policy = copy(v_new)


 while normdiff > tol && iter < maxiter

    for i in 1:length(z_dis), j in 1:length(k_grid)
    
        v_new[j, i] , k_h_index = findmax([G(k_grid[j], k_prime, h, z_dis[i], v_old, Pz) for k_prime in k_grid, h in h_grid])
         
        k_policy[j, i] = k_grid[k_h_index[1]]
        h_policy[j, i] = h_grid[k_h_index[2]]
    end

    normdiff = norm(v_new - v_old)
    v_old = copy(v_new)
    iter = iter+1
    println( "This is iter =$iter with normdiff = $normdiff, ") 
 end

 return v_old, k_policy, h_policy

end

v_old, k_policy, h_policy = VFI(100, hh)[1:3]

 ## Printing the resuls
 println( "normdiff = $normdiff, iter =$iter")    

## Getting some plots VF
plot(k_grid, v_new[:, 4], title = "Value functions ", label = " z = 3.5")
plot!(k_grid, v_new[:, 3], title = "Value functions ", label = " z = 1")
savefig("figs\\value_fn_3_b.png")


## Getting Plots for k_policy
plot(k_grid, k_policy[:, 4], title = "Policy functions k'(k, z)", xlabel="k", ylabel="k(k, z)", label = " z = 1.09")
plot(k_grid, k_policy[:, 3], title = "Policy functions k'(k, z)", xlabel="k", ylabel="k(k, z)", label = " z= 1")
savefig("figs\\cap_policy_3_b.png")

## Getting plots for h_policy
plot(k_grid, h_policy[:, 4], title = "Policy functions h(k, z)", xlabel="k", ylabel="h(k, z)", label = " z = 3.5")
plot(k_grid, h_policy[:, 3], title = "Policy functions h(k, z)", xlabel="k", ylabel="h(k, z)", label = " z = 1")
savefig("figs\\lab_policy_3_b.png")


## Simulating economy for T periods
T = 200
include("MarkovC_Simulation.jl")

state_tran = markov_simulate(Pz, 3, T)

z_vals = [z_dis[i] for i in state_tran]


k_vals = zeros(T)
k_vals[1] = bar_k
for i in 2:T
    k_vals[i] = LinearInterpolation(k_grid, k_policy[:, state_tran[i]])(k_vals[i-1])
end

h_vals = zeros(T)
h_vals[1] = bar_h
for i in 2:T
    h_vals[i] = LinearInterpolation(k_grid, h_policy[:, state_tran[i]])(k_vals[i-1])
end


y_vals = [F_p(k_vals[i], h_vals[i], z_vals[i]) for i in 1:T]
y_ss_vals = repeat([y_vals[1]], T)
    
plot(y_vals, label="y deviation from ss", xlabel = "Time", ylabel="y", title="VFI")
plot!(y_ss_vals, label="y ss = 0.59")
savefig("figs\\vfi_y_sim.png")
