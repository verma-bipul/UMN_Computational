# This script calculates the policy function using LQ approximation

# We'll first express h as fn of k, k', z using the Euler equation
using Parameters
## Loading Parameters and functions
include("parameters.jl")
@unpack β, θ, ρ, σ, ψ, γ_z, γ_n, δ   = parameters_bipul()

include("UF_functions.jl")
include("steady_state.jl")


## In the present case :
# r(Z) = r(X u) = r(k, k',z) = U(c(k, k', h(k, k' z), z), h(k, k' , z))
# X = [1 k log_z], u = [k'] , Z = [X u] = [1 k log_z k']

## Defining the state space matrices
A = [1 0 0; 0 0 0; 0 0 ρ]
B = reshape([0; 1; 0], 3, 1)
C = [0; 0; 1]

## Getting the steady state values for X, u and Z 
bar_X = [1; bar_k; 0.0] 
bar_u = [bar_k]
bar_Z = [bar_X; bar_u]

## Defining the return function for the present case
# Z = [1 k log_z k']
include("newton_julia.jl")

function r2(Z::Vector)
    k = Z[2]
    z = exp(Z[3])
    k_p = Z[4]

    function f(h)
        c = F_hat(k, h, z) - k_p*(1+γ_z)*(1+γ_n)
        return MRS(c, k, h)  
    end
    
    h = Root_Newton(f, 0.5)
    c = F_hat(k, h, z) - k_p*(1+γ_z)*(1+γ_n)

    return U(c, h)

end

## We'll now check if we've defined the function correctly
r2(bar_Z)  # The function works nice and well

##  Getting the quadratic expansion and other matrices
include("taylor_R_Z.jl")  # importing the taylor exapnsion function
 
r_hat2, R, Q, W = Taylor_rZ(r2, bar_X, bar_u)  ## with the LS function

## Running some checks on our taylor expansion
U(bar_c, bar_h)   # = -1.5 this is the utility as c_ss, h_ss
r_hat2(bar_Z)    # = -1.5this is the utility from 2nd order Taylor exapnsion
                # There is no change in the results after chnaging the return function



## Getting P, F matrices using Riccati iteration
include("riccati_eqn.jl")
P, F = Riccati(A, B, R, Q, W, β*(1+γ_n))    # note  β̂ = (1+γ_n)*β

k_primeLQ(k, log_z) = -F[1,1] - F[1, 2]*k -F[1, 3]*log_z

## Check if the policy functions evaluated at the steady state give back the state state values
k_primeLQ(bar_k, 0.0)  # 1.64 -- this is exactly the same as the steady state value of k

## Results from Vaughan 
include("vaughan.jl")
P2, F2 = Vaughan(A, B, Q, R, W, β*(1+γ_n))

k_primeV(k, log_z) = -F2[1,1] - F2[1, 2]*k -F2[1, 3]*log_z
