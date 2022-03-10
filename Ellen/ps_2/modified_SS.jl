# Date: Sep, 2021
# Bipul Verma 

# This program calculates the steady state of the model
## We'll later take the second order Taylor approxiamtion around the shocks

# calling utlity, production functions
include("UF_functions.jl")

## Loading Programs
using NLsolve
using ForwardDiff
using Parameters 

## In this version of the program we propose the following modification:
# S = [1 log(z) τc τh τd τp log(g)]'  7x1



## Creating a household instance
household = @with_kw (
    β = 0.962,   # discount parameter
    θ= 0.35,  # capital share in production

    ρ_z  = 0.50,    # AR(1) coeff
    σ_z  = 0.05,     # variance of random shocks

    ψ= 1.25,  # labor supply coeff
    γ_z = 0.0172, # productivity growth parameter     
    γ_n= 0.01,  # labor growth parameter
    δ = 0.05,  # depreciation

   g_ss = bar_g, # this has to be computed beforehand
   
   ρ_c =  0.5,
   ρ_h =  0.5,
   ρ_d =  0.5,
   ρ_p =  0.5,
   ρ_g =  0.0,
   #ρ_1 = 1.0,  # this is to take care of the constant state

   ρ_P = vec(repeat([0.5], 7, 1)),

   #P[]

   σ_Q = Diagonal(repeat([0.05], 6)),

   g_share = 0.17, # share of govt expenditure of total output.

# To calculate the steady state we'll need the steady state values of taxes as well
   τ_css = 0.065,
   τ_hss = 0.38,
   τ_dss = 0.133,
   τ_pss = 0.36,
   z_ss = 1.0,
   zτg_ss =[0.0, τ_css, τ_hss, τ_dss, τ_pss, log(g_ss)],
   #zτg_ss =[0.0, τ_css, τ_hss, τ_dss, τ_pss],

)


hh = household()
@unpack β, θ, ρ_z, σ_z, ψ, γ_z, γ_n, δ, ρ_c, ρ_h, ρ_d, ρ_p, ρ_g, ρ_P, σ_Q, g_share, τ_css, τ_hss, τ_dss, τ_pss, z_ss, zτg_ss = hh 

#=--- Constructing the modified P matrix--#
 Note that an AR(1) process with non-zero mean is written as z' = (1-ρ)z̄ + ρ*z + ϵ
 Since taxes at steady state are non-zero we'll when we write the law of motion of states X_t+1 = PX_t + Qϵ_t+1,
 we'll have to be careful in Constructing the matrix P  
=#

Pmat = zeros(7, 7)
[Pmat[i, i] = 0.5 for i in 2:7]
Pmat[1,1] = 1.0
[Pmat[i+1, 1] = (1-ρ_P[i])*zτg_ss[i] for i in 1:6]


## Defining MRS, Euler, Budget
∂U_∂c(c, h) = ForwardDiff.derivative(c->U(c, h), c)
∂U_∂h(c, h) = ForwardDiff.derivative(h->U(c,h), h)
∂F_∂k(k, h, z) = ForwardDiff.derivative(k->F_p(k, h, z), k)
∂F_∂h(k ,h, z) = ForwardDiff.derivative(h->F_p(k, h, z), h) 
r(k, h, z) =  ForwardDiff.derivative(k->F_p(k, h, z), k)
w(k, h, z) =  ForwardDiff.derivative(h->F_p(k, h, z), h) 

MRS_SS(c, k, h) = ∂F_∂h(k ,h, z_ss)*((1-τ_hss)/(1+τ_css)) + ∂U_∂h(c, h)/∂U_∂c(c, h)
Euler_SS(k, h) = (1+γ_z) - β*(∂F_∂k(k, h, z_ss)*(1-τ_pss) + 1 -δ*(1-τ_pss))
Resource_SS(c, k, h) = c + ((1+γ_z)*(1+γ_n) -1 + δ)*k - F_p(k, h, z_ss)*(1-g_share)
#Resource_SS(c, k, h) = c + ((1+γ_z)*(1+γ_n) -1 + δ)*k - F_p(k, h, z_ss)



## Creating a system of non-linear equations
function f!(F, x)
    F[1] = MRS_SS(x[1], x[2], x[3])  # x[1] is c, x[2] is k, x[3] is h
    F[2] = Euler_SS(x[2], x[3])
    F[3] = Resource_SS(x[1], x[2], x[3])
end

## Getting steady_state values
ss_vals = nlsolve(f!, [1., 1., 0.8]).zero  # with g set as 17% of ss output kss, css, hss are not very different
## Collecting the SS values 
bar_c, bar_k, bar_h = ss_vals[1], ss_vals[2], ss_vals[3] 


#
bar_y = F_p(bar_k, bar_h, z_ss)
bar_g = bar_y - bar_c - ((1+γ_z)*(1+γ_n) -1 + δ)*bar_k 

#=
bar_x = (1+γ_n)*(1+γ_z)*bar_k -(1-δ)*bar_k

r_ss = ∂F_∂k(bar_k, bar_h, z_ss)
w_ss = ∂F_∂h(bar_k ,bar_h, z_ss)

#steady state transfers
T_ss =τ_css*bar_c + τ_hss*w_ss*bar_h + τ_pss*(r_ss - δ)*bar_k + τ_dss*(r_ss*bar_k - τ_pss*bar_k*(r_ss - δ)) -τ_dss*bar_x -bar_g

c_ss = (1/(1+τ_css))*(r_ss*bar_k + w_ss*bar_h + T_ss -τ_hss*w_ss*bar_h -τ_pss*bar_k*(r_ss -δ) - τ_dss*(r_ss*bar_k - τ_pss*bar_k*(r_ss - δ)) -(1-τ_dss)*bar_x)
=#