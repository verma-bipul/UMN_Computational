# This program will calculate policy function using LQ formula

## Loading Packages
using Plots
using LinearAlgebra
using Random
using Plots

# r(Z) = r(X, u) = U((F_hat(k, k', h) -γ*k'), h)
# S = [1 log_z, τ_c, τ_h, τ_d, τ_p log_g]'
# X = [1 k S K H]' , u = [k' h]' , Z = [ X u]' = [1 k S K H k' h] = [1 k log_z, τ_c, τ_h, τ_d, τ_p log_g K H k' h]


## Loading Parameters and functions
include("UF_functions.jl")

## Getting SS values from the previous file
include("modified_SS.jl")
bar_S = [1.0, log(z_ss), τ_css, τ_hss, τ_dss, τ_pss, log(g_ss)]
bar_X = [1.0; bar_k; bar_S; bar_k; bar_k; bar_h]
bar_Y = [1.0; bar_k; bar_S]
bar_u = [bar_k,  bar_h]
bar_Z = [bar_X; bar_u]


## 
# Z = = [1 k 1 log_z, τ_c, τ_h, τ_d, τ_p log_g K H k' h]
#=
function r1(Z::Vector)
    k, z, τ_c, τ_h, τ_d, τ_p, g, K, H, k_p, h = Z[2], exp(Z[3]), Z[4], Z[5], Z[6], Z[7], exp(Z[8]), Z[9], Z[10], Z[11], Z[12] 
    rate = θ*(z*H/K)^(1-θ)
    wage = (1-θ)*(z)^(1-θ)*(K/H)^(θ)
    invest = (1+γ_n)*(1+γ_z)*k_p -(1-δ)*k
    c = (1/(1+τ_c))*(rate*k + wage*h + T_ss -τ_h*wage*h -τ_p*k*(rate -δ) - τ_d*(rate*k - τ_p*k*(rate - δ)) -(1-τ_d)*invest)
    return U(c, h)
end
=#
#     [1 2  3  4    5     6    7   8   9    10 11 12 13 14] 
#Z  = [1 k 1 log_z, τ_c, τ_h, τ_d, τ_p log_g K K' H k' h]
# We will later obtain everything in terms of log deviations as well


function r1(Z::Vector)
    k = bar_k*(1+Z[2])
    z = z_ss*exp(Z[4])
    τ_c=τ_css*(1+Z[5])
    τ_h=τ_hss*(1+Z[6])
    τ_d=τ_dss*(1+Z[7])
    τ_p= τ_pss*(1+Z[8])
    g= bar_g*exp(Z[9])
    K= bar_k*(1+Z[10])
    Kp= bar_k*(1+Z[11])
    H= bar_h*(1+Z[12])
    k_p= bar_k*(1+Z[13])
    h =  bar_h*(1+Z[14])
    rate = θ*(z*H/K)^(1-θ)
    wage = (1-θ)*(z)^(1-θ)*(K/H)^(θ)
    X = (1+γ_n)*(1+γ_z)*Kp -(1-δ)*K
    Y = F_p(K, H, z)
    C = Y - X - g
    x= (1+γ_n)*(1+γ_z)*k_p -(1-δ)*k
    transfers = τ_c*C + wage*τ_h*H + τ_p*(rate-δ)*K + τ_d*(rate*K - τ_p*K*(rate - δ)) - τ_d*X - g
    c = (1/(1+τ_c))*(rate*k + wage*h + transfers -τ_h*wage*h -τ_p*k*(rate -δ) - τ_d*(rate*k - τ_p*k*(rate - δ)) -(1-τ_d)*x)
    return U(c, h)
end


## Since all variables are in deviations from SS, their values at SS will be zero
bar_S = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
bar_X = [1.0; 0.0; bar_S; 0.0; 0.0; 0.0]
bar_Y = [1.0; 0.0; bar_S]
bar_u = [0.0,  0.0]
bar_Z = [bar_X; bar_u]

## Checking our function
U(bar_c, bar_h)  #both give the same value hence our function is correct
r1(bar_Z)   #-- the expression is correct

##  Getting the quadratic expansion and other matrices
include("taylor_R_Z.jl")  # importing the taylor exapnsion function
 
r_hat, Q, R, W = Taylor_rZ(r1, bar_X, bar_u)  ## with the LS function

## Running some checks on our taylor expansion
U(bar_c, bar_h)   # -1.75 this is the utility as c_ss, h_ss
r_hat(bar_Z)    # = -1.75 this is the utility from 2nd order Taylor exapnsion

## Redefinig Pmat
Pmat[2:end,1] .= 0.0

## We'll now write down the matrices A, B which will be used in LQ riccati and Vaughan
A = zeros(12, 12)
A[1, 1] = 1
for i in 3:9, j in 3:9
    A[i, j] =Pmat[i-2, j-2]  
end

B = zeros(12, 2)
B[2, 1] = 1

Θ = zeros(3, 9)
Θ[1, 2] = 1

Ψ= zeros(3, 2)
Ψ[2,1] = 1
Ψ[3, 2] = 1

## Getting matrices in the required form
A_t = sqrt(β*(1+γ_n))*(A - B*inv(R)*W')
B_t = sqrt(β*(1+γ_n))*B
Q_t = Q - W*inv(R)*W'


A_ty, A_tz = A_t[1:9, 1:9], A_t[1:9, 10:end]
Q_ty, Q_tz = Q_t[1:9, 1:9], Q_t[1:9, 10:end]

B_ty, B_tz = B_t[1:9, :], B_t[10:end, :]
W_y, W_z = W[1:9, :], W[10:end, :]


## Getting hat matrices
Θ_t = inv((Matrix(1.0I, 3, 3) + Ψ*inv(R)*W_z'))*(Θ - Ψ*inv(R)*W_y')
Ψ_t = inv((Matrix(1.0I, 3, 3) + Ψ*inv(R)*W_z'))*Ψ
A_hat = A_ty + A_tz*Θ_t
Q_hat = Q_ty + Q_tz*Θ_t
B_hat = B_ty + A_tz*Ψ_t

A_bar = A_ty - B_ty*inv(R)*Ψ_t'*Q_tz'


## Results 
H = [inv(A_hat)  inv(A_hat)*B_hat*inv(R)*B_ty'; Q_hat*inv(A_hat)  Q_hat*inv(A_hat)*B_hat*inv(R)*B_ty' + A_bar']
V = eigen(H).vectors    
L = Int(size(V)[1]/2)
# We need to split the matrix  V into 4 equal parts
V12 = V[1:L, L+1:end]
V22 = V[L+1:end, L+1:end]
P = V22*inv(V12)

# We can get P using the Riccati iteration as well
# Lets compare if we get the same result
P_old = rand(8, 8)
P_old = -P_old'*P_old  # We need to ensure that we start with a guess of negative definite matrix

eigen(P_old)    # check if all the eigenvalues are negative


normdiff = 100.0
iter = 1
tolerance = 1e-8
maxiter = 1000

while normdiff > tolerance && iter < maxiter
    P_new = Q_hat + A_bar'*P_old*A_hat -(A_bar'*P_old*B_hat)*inv(R + B_ty'*P_old*B_hat)*(B_ty'*P_old*A_hat)
        
    normdiff = norm(P_new - P_old)

    P_old = copy(P_new)
    iter = iter+1
    println("The normdiff is $normdiff in $iter iterations")
end

# We confirm that we obtain the same P using the two methods

## Obtaining matrix F
F_t = inv(R + B_ty'*P*B_hat)B_ty'*P*A_hat
F = inv(Matrix(1.0I, 2, 2) + inv(R)*W_z'Ψ)*(F_t + inv(R)*(W_y' + W_z'*Θ))

## Calculating policy function from F
# Note that states are y = [1 k log_z, τ_c, τ_h, τ_d, τ_p 1 log_g]
k_policy(Y::Vector) = reshape(-F[1,:], 1, 9)*Y
h_policy(Y::Vector) = reshape(-F[2,:], 1, 9)*Y


## Let us check if we get back the ss values 
#     [1 2  3  4    5     6    7   8   9    10 11 12 13 14] 
#Z  = [1 k 1 log_z, τ_c, τ_h, τ_d, τ_p log_g K K' H k' h]
# y = [1 k 1 log_z, τ_c, τ_h, τ_d, τ_p log_g]
bar_Y = bar_Z[1:9]
k_policy(bar_Y)  # the values very very close to the original ss values
h_policy(bar_Y)

## Plots from LQ
# Note we make sure that the same random number are being generated for LQ and Vaughan
rng = Random.seed!(1234)
T = 200

Σmat = zeros(7, 7)
[Σmat[i, i] = 0.05 for i in 2:7]

shocks_sim = zeros(7, T)    # S = [1 log_z, τ_c, τ_h, τ_d, τ_p log_g]
shocks_sim[:, 1] = bar_S
[shocks_sim[:, t] = Pmat*shocks_sim[:, t-1] + Σmat*randn(7,1) for t in 2:T]

## Check if the mean values of the shcoks are around the ss  -- this check is passed 
mean(shocks_sim[2,:]) #-- should be close to zero
mean(shocks_sim[3,:])  #-- should be close to τ_css



## Lets us simulate the output deviation from steady state for the simulated shocks
k_sim = zeros(T)
k_sim[1] = bar_k
[k_sim[t] = k_policy([1; k_sim[t-1]; shocks_sim[:, t]])[1] for t in 2:T]
# check mean = ss 
mean(k_sim)

h_sim = zeros(T)
h_sim[1] = bar_h
[h_sim[t] = h_policy([1; k_sim[t-1]; shocks_sim[:, t]])[1] for t in 2:T]
mean(h_sim)

y_sim = [F_p(k_sim[t], h_sim[t], exp(shocks_sim[2,t])) for t in 1:T]
mean(y_sim)

plot(k_sim, label="capital deviation", title="LQ with Distortions", xlabel="Time", ylabel="capital")
plot!(fill(bar_k, T), label="ss capital")
savefig("figs/k_dev_LQ_new.png")

plot(h_sim, label="hours deviation", title="LQ with Distortions", xlabel="Time", ylabel="hours")
plot!(repeat([bar_h], T), label="ss hours")
savefig("figs/h_dev_LQ_new.png")

plot(y_sim, label="output deviation", title="LQ with Distortions", xlabel="Time", ylabel="output")
plot!(repeat([bar_y], T), label="ss output")
savefig("figs/y_dev_LQ_new.png")


stock_val_sim = [(1-shocks_sim[5, t])*k_sim[t] for t in 1:T]
plot(stock_val_sim, label="Stock Valuation", title="LQ with Distortions", xlabel="Time", ylabel="S val")
savefig("figs/stock_dev_LQ_new.png")


## Thnk about how to simulate the economy with certain shocks on and other off.



# Lets create an indicator function of which shocks are on and which are not

Is = [1.0 0.0 0.0 0.0 0.0 0.0; 1.0 1.0 0.0 0.0 0.0 0.0; 1.0 1.0 1.0 0.0 0.0 0.0; 1.0 1.0 1.0 1.0 0.0 0.0; 1.0 1.0 1.0 1.0 1.0 0.0; 1.0 1.0 1.0 1.0 1.0 1.0 ]  
Is = Is'
#log_z, τ_c, τ_h, τ_d, τ_p log_g
col_labels = ["log_z", "tau_c", "tau_h", "tau_d", "tau_p", "log_g"]

k_vals = zeros(T, 6)
[k_vals[1, i] = bar_k for i in 1:6]

for j in 1:6
    for i in 2:T
        k_vals[i, j] = k_policy([1.0; k_vals[i-1, j]; Is[:, j].*S_vals[i, :]])[1]
    end
end


    
h_vals = zeros(T, 6)
[h_vals[1, i] = bar_h for i in 1:6]

for j in 1:6
    for i in 2:T
        h_vals[i, j] = h_policy([1.0; k_vals[i-1, j]; Is[:, j].*S_vals[i, :]])[1]
    end
end

y_vals = zeros(T, 6)

for j in 1:6
    for i in 1:T
        y_vals[i, j] = F_p(k_vals[i, j], h_vals[i, j], exp(S_vals[i,1]))
    end
end
    

#y_ss_vals = repeat([y_vals[1]], T)
    
plot(y_vals[2:end, 1], label=col_labels[1], xlabel = "Time", ylabel="y", title="Output deviation")

plot(y_vals[2:end,:], layout = (3, 2), label = ["log_z" "add tau_c" "add tau_h" "add tau_d" "add tau_p" "add log_g"], title=["Output deviation" "" "" "" "" ""])
savefig("figs\\LQ_y_taxes_sim.png")



## Plots LQ 
plot(k_grid, k_primeLQ.(k_grid, 0), title = "k'(k, z) using LQ/Vaughan", xlabel="k", ylabel="k'(k, z)", label = "k' = 0.031 + 0.8971k +0.57*log(z_ss)") 
savefig("figs\\cap_policy_VLQ.png") # upward sloping curve

plot(k_grid, h_primeLQ.(k_grid, 0), title = "h(k, z) using LQ/Vaughan ", xlabel="k", ylabel="h(k, z)", label = "h = 0.328 - 0.0195k + 0.189*log(z_ss)") #downward sloping  curve
savefig("figs\\h_policy_VLQ.png")

## Plots Vaughan 
plot(k_grid, k_primeV.(k_grid, 0), title = "k'(k, z) using Vaughan", xlabel="k", ylabel="k'(k, z)") # upward sloping curve
savefig("figs\\cap_policy_1_c_1.png")

plot(k_grid, h_primeV.(k_grid, 0), title = "h(k, z) using Vaughan ", xlabel="k", ylabel="h(k, z)") #downward sloping  curve
savefig("figs\\h_policy_1_c_1.png")


