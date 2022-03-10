# This subroutine returns the P matrix using the Vaughan Method

## Loading Packages
using LinearAlgebra
using ForwardDiff
using Random
using Plots, Distributions

# Step 1: Write the Euler conditions and the matrices
# Step 2: Log linearize both conditions around the steady state 
# Step 3: Extract the required matrices from the gradients on log linear


# S = [log_z, τ_c, τ_h, τ_d, τ_p log_g]'

## Calling dependent modules
include("UF_functions.jl")
include("modified_SS.jl")
#include("steady_state.jl")
#include("parameters.jl")

## Euler equations and MRS
function MRS(input_var::Vector)   # k, k_p, h, log_z, τ_h, τ_c, log_g  - inputs
    k, k_p, h, log_z, τ_h, τ_c, log_g = input_var
    c = F_hat(k, h, exp(log_z)) - (1+γ_z)*(1+γ_n)*k_p -exp(log_g)
    return 1 + (∂U_∂c(c, h)/∂U_∂h(c, h))*((1-τ_h)/(1+τ_c))*w(k, h, exp(log_z))
end

function Euler(input_var::Vector)
    k, k_p, k_pp, h, h_p, log_z, τ_c, τ_d, log_g, log_z_p, τ_cp, τ_dp, τ_pp, log_g_p = input_var
    c =  F_hat(k, h, exp(log_z)) - (1+γ_z)*(1+γ_n)*k_p -exp(log_g)
    c_p =  F_hat(k_p, h_p, exp(log_z_p)) - (1+γ_z)*(1+γ_n)*k_pp -exp(log_g_p)
    return 1 - (β/(1+γ_z))*((1+τ_c)/(1+τ_cp))*((1-τ_dp)/(1-τ_d))*(∂U_∂c(c_p, h_p)/∂U_∂c(c, h))*((r(k_p, h_p, exp(log_z_p))-δ)*(1-τ_pp)+1)
end

## Lets do a small check whether MRS and Euler are zero at the SS values
input_mrs_ss = [bar_k, bar_k, bar_h, log(z_ss), τ_hss, τ_css, log(bar_g)]
input_euler_ss = [bar_k, bar_k, bar_k, bar_h, bar_h, log(z_ss), τ_css, τ_dss, log(bar_g), log(z_ss), τ_css, τ_dss, τ_pss, log(bar_g)]

MRS(input_mrs_ss)  # Value is 0 which is what we required
Euler(input_euler_ss)  #Value is 0 which is what we requires

## Log Linearization
input_mrs_ss2 = [bar_k, bar_k, bar_h, z_ss, τ_hss, τ_css, bar_g]
input_euler_ss2 = [bar_k, bar_k, bar_k, bar_h, bar_h, z_ss, τ_css, τ_dss, bar_g, z_ss, τ_css, τ_dss, τ_pss, bar_g]

a = ForwardDiff.gradient(MRS, input_mrs_ss).*input_mrs_ss2
b = ForwardDiff.gradient(Euler, input_euler_ss).*input_euler_ss2 

# ? How to check if the coefficients are correct ? No

## Building up the required matrices
A1 = [1 0 0; 0 0 0; 0 b[3] b[5]]
A2 = [0 -1 0; a[1] a[2] a[3]; b[1] b[2] b[4]]
#B = [0 0 0 0 0 0 0 0 0 0 0 0 ; a[4] a[6] a[5] 0 0 a[7] 0 0 0 0 0 0; b[6] b[7] 0 b[8] 0 b[9] b[10] b[11] 0 b[12] b[13] b[14]]


## Solution using the method of undetermined coefficients
## Let us solve the final system of equations to get the final policy functions 
## -- https://makotonakajima.github.io/files/note/note_rbc_uc.pdf   Reference for method of undeter coeffs


function f!(F, x)   # x = [A B C2 D2]  # 14 coefficients to be determined from 4 equations??
    A=x[1]
    B=x[2:7]
    C2= x[8]        
    D2 = x[9:14]

    # setting up the system of 14 equations
    F[1] = a[1] + a[2]*A + a[3]*C2    #-- eqn 1
    F[2:7] = a[2]*B + a[3]*D2 + [a[4], a[6], a[5], 0, 0, a[7]]    #-- eqn 2 t0 7
    F[8] = b[1] + b[2]*A + b[3]*A^2 + b[4]*C2 + b[5]*C2*A       # eqn 8
    F[9:14] = b[2]*B + b[3]*A*B + b[3]*B.*0.5 + b[4]*D2 + b[5]*C2*B + b[5]*B.*0.5+ [b[6], b[7], 0, b[8], 0, b[9]] +[b[10], b[11], 0, b[12], b[13], b[14]].*0.5
end

Sol = nlsolve(f!, ones(14))

A=Sol.zero[1]
B=Sol.zero[2:7]
C2= Sol.zero[8]        
D2 = Sol.zero[9:14]


a[1] + a[2]*A + a[3]*C2  ## this is zero -- showing that the method of undetermined coefficients works well


# Now once we have all the required matrices we can go ahead to construct our final policy functions:
k_policyV(St::Vector) = A*St[1] + dot(B[1:6], St[2:7])   # states = [z̃ τc̃ τh̃ τd̃ τp̃ g̃]
h_policyV(St::Vector) = C2*St[1] + dot(D2[1:6],St[2:7]) 


## Converting the policies from log-deviation to levels
coeff_k = zeros(8)
coeff_h = zeros(8)

coeff_k[1] = bar_k*(1+A + B[2]+B[3]+B[4]+B[5] +B[6]*log(bar_g))  # this is the constant terms
coeff_k[2] = A   # -- coeff on K
coeff_k[3] = B[1]*bar_k   #---- coeff on log_z
[coeff_k[i] = B[i-2]*bar_k/steady_vals[i-1] for i in 4:7]
coeff_k[end] = bar_k*B[end]

coeff_h[1] = bar_h*(1+ C2 + D2[2]+D2[3]+D2[4]+D2[5] +D2[6]*log(bar_g))  # this is the constant terms
coeff_k[2] = A   # -- coeff on K
coeff_k[3] = B[1]*bar_k   #---- coeff on log_z
[coeff_k[i] = B[i-2]*bar_k/steady_vals[i-1] for i in 4:7]
coeff_k[end] = bar_k*B[end]



steady_vals = [bar_k, 1.0, τ_css, τ_hss, τ_dss, τ_pss, bar_g] 



## We'll now plot the time series based on these policy functions
rng = Random.seed!(1234)  # Resetting the random seed to match the results with LQ
T = 200

Σmat = zeros(7, 7)  #Σmat collects the standard deviations for the process [1 z̃ τc̃ τh̃ τd̃ τp̃ g̃]
Σmat[2,2] = 0.05
Σmat[7,7] = 0.05
[Σmat[i, i] = 0.05/zτg_ss[i] for i in 3:6]

shocks_simV = zeros(7, T)    # # states = [1 z̃ τc̃ τh̃ τd̃ τp̃ g̃] -- the extra 1 is added to match the randmon nos with LQ
[shocks_simV[:, t] = 0.5*shocks_simV[:, t-1] + Σmat*randn(7,1) for t in 2:T]

shocks_simV = shocks_simV[2:end, :]

## Check if the mean values of the shcoks are around the ss  -- this check is passed 
mean(shocks_simV[2,:]) #-- should be close to zero
mean(shocks_simV[3,:])  #-- should be close to zero

## Lets us simulate the output deviation from steady state for the simulated shocks
kt_sim = zeros(T)
[kt_sim[t] = k_policyV([kt_sim[t-1]; shocks_simV[:, t]])[1] for t in 2:T]
# check mean = ss 
mean(kt_sim)   #should be close to zero

kV_sim1 = [exp(kt_sim[t] + log(bar_k)) for t in 1:T]
kV_sim2 = [(kt_sim[t]+1)*bar_k for t in 1:T]

ht_sim = zeros(T)
[ht_sim[t] = h_policyV([kt_sim[t-1]; shocks_simV[:, t]])[1] for t in 2:T]
mean(ht_sim)

# ỹ = θk̃ + (1-θ)(z̃ + h̃)
yt_sim = [θ*kt_sim[t] + (1-θ)*(shocks_simV[1, t] + ht_sim[t]) for t in 1:T]
mean(yt_sim)


yV_sim = [exp(yt_sim[t] + log(bar_y)) for t in 1:T]
mean(yV_sim)
plot(yV_sim)


plot!(kt_sim, label="log capital deviation", title="Vaughan ", xlabel="Time", ylabel="log capital")
savefig("figs/log_k_dev_vaughan_new.png")

plot(yt_sim, label="log output deviation", title="Vaughan ", xlabel="Time", ylabel="log output")
plot!(repeat([0.0], T))
savefig("figs/log_y_dev_vaughan_new.png")

    




## Solving thing using Generalized schur decomposition
T = schur(A1, -A2).T  ##-- Notice that only the second root is stable, hence we'll work with that only
S = schur(A1, -A2).S
Z = inv(schur(A1, -A2).Z')

T1 = T[2,2]
S1 = S[2,2]
Z1 = Z[:, 2]

M = Z1[1, 1]*S1^(-1)*T1*Z[1, 1]^(-1)

## Obtaining the eigen values
eig = eigen(A1, -A2) # notice that only one of the roots will be stable here its the second one
V = eig.vectors
Π = eig.values
Λ = Diagonal(Π)

## Rearragements and sorting
Λ1 = Λ[2,2]
V1 = V[:,2] 


## Writing out matrices A and c
M = V1[1,1]*Λ1*inv(V1[1, 1])
C = V1[2:end, 1]*inv(V1[1,1])

a[1] + a[2]*M + a[3]*C[2]  ## this is not zero??

## Let us solve the final system of equations to get the final policy functions 
## -- https://makotonakajima.github.io/files/note/note_rbc_uc.pdf   Reference for method of undeter coeffs


function f!(F, x)
    B=x[1:6]        # N and D2 are of the same dim
    D2 = x[7:12]

    F[1:6] = a[2]*B + a[3]*D2 + [a[4], a[5], a[6], 0, 0, a[7]]
    F[7:12] = b[2]*B + b[3]*A*B + b[3]*B.*0.5 + b[4]*D2 + b[5]*C[2]*B + b[5]*B.*0.5+ [b[6], b[7], 0, b[8], 0, b[9]] +[b[10], b[11], 0, b[12], b[13], b[14]].*0.5
end

Sol = nlsolve(f!, ones(12))

N = Sol.zero[1:6]
D2 = Sol.zero[7:12]

# Now once we have all the required matrices we can go ahead to construct our final policy functions:
k_policyV(St::Vector) = M*St[1] + dot(N[1:6], St[2:7])   # states = [z̃ τc̃ τh̃ τd̃ τp̃ g̃]
h_policyV(St::Vector) = C[2]*St[1] + dot(D2[1:6],St[2:7]) 

#y_ss_vals = repeat([y_vals[1]], T)
    
plot(yt_vals[:, 1], label=col_labels[1], xlabel = "Time", ylabel="y", title="Output deviation")
plot(kt_vals[:, 1], label=col_labels[1], xlabel = "Time", ylabel="y", title="Capital deviation")
plot(ht_vals[:, 1], label=col_labels[1], xlabel = "Time", ylabel="y", title="Labor deviation")


plot(yt_vals, layout = (3, 2), label = ["log_zt" "add tau_ct" "add tau_ht" "add tau_dt" "add tau_pt" "add log_gt"], title=["Output deviation Vaughan" "" "" "" "" ""])
savefig("figs\\Vaughan_y_taxes_sim.png")


