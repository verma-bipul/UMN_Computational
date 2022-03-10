# This program will calculate policy function using LQ formula

## Loading Packages
using Plots, Parameters

# r(Z) = r(X, u) = U((F_hat(k, k', h) -γ*k'), h)
# X = [1 k log_z]' , u = [k' h]' , Z = [ X u]' = [1 k log_z k' h]


## Loading Parameters and functions
include("parameters.jl")
include("UF_functions.jl")
include("steady_state.jl")

@unpack β, θ, ρ, σ, ψ, γ_z, γ_n, δ   = parameters_bipul()


## Defining matrices for LQ1
A = [1 0 0; 0 0 0; 0 0 ρ]
B = [0 0; 1 0; 0 0]
C = [0, 0, 1]

## Getting SS values from the previous file
bar_X = [1.0, bar_k, 0.0]
bar_u = [bar_k,  bar_h]
bar_Z = [bar_X; bar_u]

## Creating r(Z) = r(X, u) = U((F_hat(k, h, z) -γ*k'), h)
# Z = [1 k log_z k' h]

function r1(Z::Vector)
    c = F_hat(Z[2], Z[5], exp(Z[3])) - Z[4]*(1+γ_z)*(1+γ_n)
    return U(c, Z[5])
end

function r1MM(W::Vector) # W = [k log_z k' h]  # this is just to try out us MM textbook method
    c = F_hat(W[1], W[4], exp(W[2])) - W[3]*(1+γ_z)*(1+γ_n)
    return U(c, W[4])
end

##  Getting the quadratic expansion and other matrices
include("taylor_R_Z.jl")  # importing the taylor exapnsion function
 
r_hat, R, Q, W = Taylor_rZ(r1, bar_X, bar_u)  ## with the LS function

## Running some checks on our taylor expansion
U(bar_c, bar_h)   # = -1.5022160395661426 this is the utility as c_ss, h_ss
r_hat(bar_Z)    # = -1.502216039566167 this is the utility from 2nd order Taylor exapnsion

# the taylor looks good with jakes parameters.


## Getting P, F matrices using Riccati iteration
include("riccati_eqn.jl")
P, F = Riccati(A, B, R, Q, W, β*(1+γ_n))


k_primeLQ(k, log_z) = -F[1,1] - F[1, 2]*k -F[1, 3]*log_z
h_primeLQ(k, log_z) = -F[2, 1] - F[2,2]*k - F[2, 3]*log_z

## Check if the policy functions evaluated at the steady state give back the state state values
k_primeLQ(bar_k, 0.0)  # 1.64 -- this is exactly the same as the steady state value of k
h_primeLQ(bar_k, 0.0)  # 0.35  -- this matches with the steady state value as well

## Results from Vaughan 
include("vaughan.jl")
P2, F2 = Vaughan(A, B, Q, R, W, β*(1+γ_n))

k_primeV(k, log_z) = -F2[1,1] - F2[1, 2]*k -F2[1, 3]*log_z
h_primeV(k, log_z) = -F2[2, 1] - F2[2,2]*k - F2[2, 3]*log_z

## We note that the P and F matrices obtained are the same from both the methods


## Plots LQ 
plot(k_grid, k_primeLQ.(k_grid, 0), title = "k'(k, z) using LQ/Vaughan", xlabel="k", ylabel="k'(k, z)", label = "k' = 0.24 + 0.85k +0.46log(z_ss)") 
plot(k_grid, k_primeLQ.(k_grid, log(z_dis[4])), title = "k'(k, z) using LQ/Vaughan", xlabel="k", ylabel="k'(k, z)", label = "k' = 0.031 + 0.8971k +0.57*log(z[4])") 

savefig("figs\\cap_policy_VLQ.png") # upward sloping curve

plot(k_grid, h_primeLQ.(k_grid, 0), title = "h(k, z) using LQ/Vaughan ", xlabel="k", ylabel="h(k, z)", label = "h = 0.42 - 0.043k + 0.17log(z_ss)") #downward sloping  curve
savefig("figs\\h_policy_VLQ.png")

## Plots Vaughan 
plot(k_grid, k_primeV.(k_grid, 0), title = "k'(k, z) using Vaughan", xlabel="k", ylabel="k'(k, z)") # upward sloping curve
savefig("figs\\cap_policy_1_c_1.png")

plot(k_grid, h_primeV.(k_grid, 0), title = "h(k, z) using Vaughan ", xlabel="k", ylabel="h(k, z)") #downward sloping  curve
savefig("figs\\h_policy_1_c_1.png")



## Simulating economy for T periods
T = 200
include("MarkovC_Simulation.jl")

state_tran = markov_simulate(Pz, 3, T)

z_vals = [z_dis[i] for i in state_tran]


k_vals = zeros(T)
k_vals[1] = bar_k

for i in 2:T
    k_vals[i] = k_primeLQ(k_vals[i-1], log(z_vals[i]))
end
    
h_vals = zeros(T)
h_vals[1] = bar_h
for i in 2:T
    h_vals[i] = h_primeLQ(k_vals[i-1], log(z_vals[i]))
end
    

y_vals = [F_p(k_vals[i], h_vals[i], z_vals[i]) for i in 1:T]
y_ss_vals = repeat([y_vals[1]], T)
    
plot(y_vals, label="y deviation from ss", xlabel = "Time", ylabel="y", title="LQ/Vaughan")
plot!(y_ss_vals, label="y ss = 0.59")
savefig("figs\\LQ_y_sim.png")


