# We'll write down matrices T, Z, Q, H for the various process
# State: xₜ = Txₜ₋₁ + ηₜ ∼ N(0, Q) 
# Observation: yₜ = Zxₜ + ϵₜ ∼ N(0, H)

##Loading packages
using Random
using Distributions
using Plots
using Optim
using ForwardDiff

rng=Random.seed!(456)


## Parameters 
n = 1000
ρ = 0.6
σ = 0.5

## Simulate AR(1)
y_AR1 = repeat([reshape([0.0], 1, 1)], n)
[y_AR1[i] = [ρ]*y_AR1[i-1] + σ*randn(rng, 1) for i in 2:n] 
vec_y = [y_AR1[i][1, 1] for i in 1:n]


## Getting Matrices to apply kalman filter
T = reshape([ρ], 1, 1)
Z = reshape([1.0], 1, 1)
Q = reshape([σ^2], 1, 1)
H = reshape([0.0], 1, 1)
x0 = reshape(mean(y_AR1), 1, 1)
P0 = copy(x0)

# Getting P0
#nclude("stationary_P.jl")
#P_st = stationary_P(T, Q)

## Applying kalman_filter
include("stationary_P.jl")
#include("kalman_filter.jl")
include("kalman.jl")

#x_hat, v, y_hat, P, F, G, K  = kalman_filter(T, Z, Q, H, x0, y_AR1, n)
y_hat = kalman_filter(T, Z, Q, H, P0, x0, y_AR1)[1]
vec_y_hat = [y_hat[i][1, 1] for i in 1:length(y_hat)]


## Plotting the y vs y_hat
#vec_y_hat = [y_hat[i][1, 1] for i in 2:n]

plot(vec_y,xlabel="Time", label ="Simulated AR(1)", legend=:bottomright)
plot!(vec_y_hat, label ="Estimate from Kalman Filter", legend=:bottomright)
savefig("figs\\AR1.png")


## Writing the likelhood function
# For the AR1 process we need to estimate two Parameters ρ and σ:


function log_L_AR1(θ::Vector, y)  # θ= [ρ, log_σ]
    T = length(y)

    ρ1 = θ[1]
    σ1 = exp(θ[2])
    T1 = reshape([ρ1], 1, 1)
    Q1 = reshape([σ1^2], 1, 1)

    x_t_tm1 = kalman_filter(T1, Z, Q1, H, P0, x0, y)[1]
    P_t_tm1 = kalman_filter(T1, Z, Q1, H, P0, x0, y)[3]

    v1 = [y[i] - Z*x_t_tm1[i] for i in 1:T]
    vec_v = [v1[i][1,1] for i in 1:T]
    F1 = [Z*P_t_tm1[i]*Z' + H for i in 1:T]
    Ft = abs.(det.(vec(F1)))

    return sum([-0.5*log(Ft[i]) - 0.5*vec_v[i]'*inv(Ft[i])*vec_v[i] for i in 1:T])
end


## Optimization
res = optimize(θ->-log_L_AR1(θ, y_AR1), [0.1, -0.1])
ρ_hat, σ_hat= Optim.minimizer(res)[1], exp(Optim.minimizer(res)[2]) # (0.59, 0.47)
# actual 0.6, 0.5



#------------------------------------------------------------------------------------------#

## Same thing for AR[2] process  

ρ1=0.3
ρ2=0.4
σ = 0.5
n=1000

T = [ρ1 ρ2; 1.0 0.0]
Z = [1.0 0.0]
Q = [σ^2 0.0; 0.0 0.0 ]
H = reshape([0.0], 1, 1)

## Simulate AR(2) process
y_AR2 = repeat([reshape([0.0], 1, 1)], n)
[y_AR2[i] = [ρ1]*y_AR2[i-1] + [ρ2]*y_AR2[i-2] + σ*randn(rng, 1) for i in 3:n] 
vec_yAR2 = [y_AR2[i][1, 1] for i in 1:n]

## Initial State Value
x0 = [mean(y_AR2); mean(y_AR2)]
include("stationary_P.jl")
P0 = stationary_P(T, Q)

## Applying kalman_filter
y_hat = kalman_filter(T, Z, Q, H, P0, x0, y_AR2)[1]
vec_y_hat = [y_hat[i][1, 1] for i in 1:length(y_hat)]



## Plots
plot(vec_yAR2,xlabel="Time", label ="Simulated AR(2)", legend=:bottomright)
plot!(vec_y_hat, label ="Estimate from Kalman Filter", legend=:bottomright)
savefig("figs\\AR2.png")

## MLE estimate



# For the AR2 process we need to estimate three Parameters ρ1, ρ2, and σ:
function log_L_AR2(θ::Vector, y)  # θ= [ρ1, ρ2, log_σ]
    T = length(y)

    ρ1 = θ[1]
    ρ2 = θ[2]
    σ1 = exp(θ[3])
    T1 = [ρ1 ρ2; 1.0 0.0]
    Q1 = [σ1^2 0.0; 0.0 0.0 ]

    x_t_tm1 = kalman_filter(T1, Z, Q1, H, P0, x0, y)[1]
    P_t_tm1 = kalman_filter(T1, Z, Q1, H, P0, x0, y)[3]

    v1 = [y[i] - Z*x_t_tm1[i] for i in 1:T]
    vec_v = [v1[i][1,1] for i in 1:T]
    F1 = [Z*P_t_tm1[i]*Z' + H for i in 1:T]
    Ft = abs.(det.(vec(F1)))

    return sum([-0.5*log(Ft[i]) - 0.5*vec_v[i]'*inv(Ft[i])*vec_v[i] for i in 1:T])

end

## Optimization
res = optimize(θ->-log_L_AR2(θ, y_AR2), [0.1, 0.1, -0.1])
ρ1_hat, ρ2_hat, σ_hat= Optim.minimizer(res)[1], Optim.minimizer(res)[2], exp(Optim.minimizer(res)[3])
# (0.43805774271267905, 0.339321425959066, 0.48782512515169046) with 100 Observation
# (0.29998269723452614, 0.3991193281365929, 0.5071353879851042) with 1000
#  original 0.3, 0.4, 0.5

#-------------------------------------------------------------------------------------#
## MA(1)

ρ = 0.7
σ = 0.8
n = 5000

B = [0.0 0.0; 1.0 0.0]
Z = [1.0 ρ]
Q = [σ^2 0.0; 0.0 0.0 ]
H = reshape([0.0], 1, 1)

## Simlate MA1
ϵ = rand(Normal(0, σ), n)

y_MA1 = repeat([reshape([0.0], 1, 1)], n)
[y_MA1[i] = reshape([ϵ[i]], 1,1) + [ρ]*ϵ[i-1] for i in 2:n] 
vec_yMA1 = [y_MA1[i][1, 1] for i in 1:n]

## Initial State Value
x0 = reshape([0.0; 0.0], 2,1)

include("stationary_P.jl")
P0 = stationary_P(T, Q)

## Applying kalman_filter
y_hat = kalman_filter(T, Z, Q, H, P0, x0, y_MA1)[1]
#vec_y_hat = [y_hat[i][1, 1] for i in 1:length(y_hat)]



## Plots
plot(vec_yMA1,xlabel="Time", label ="Simulated MA1", legend=:bottomright)
plot!(vec_y_hat, label ="Estimate from Kalman Filter", legend=:bottomright)
savefig("figs\\MA1.png")

## MLE estimate
# For the AR2 process we need to estimate two Parameters ρ1, and σ:
function log_L_MA1(θ::Vector, y)  # θ= [ρ1, log_σ]
    T = length(y)

    ρ1 = θ[1]
    σ1 = exp(θ[2])
    Z1 = [1.0 ρ1]
    Q1 = [σ1^2 0.0; 0.0 0.0 ]

    x_t_tm1 = kalman_filter(B, Z1, Q1, H, P0, x0, y)[1]
    P_t_tm1 = kalman_filter(B, Z1, Q1, H, P0, x0, y)[3]

    v1 = [y[i] - Z1*x_t_tm1[i] for i in 1:T]
    vec_v = [v1[i][1,1] for i in 1:T]
    F1 = [Z1*P_t_tm1[i]*Z1' + H for i in 1:T]
    Ft = abs.(det.(vec(F1)))

    return sum([-0.5*log(Ft[i]) - 0.5*vec_v[i]'*inv(Ft[i])*vec_v[i] for i in 1:T])
end



## Optimization
res = optimize(θ->-log_L_MA1(θ, y_MA1), [0.1, -0.1])
ρ_hat, σ_hat= Optim.minimizer(res)[1], exp(Optim.minimizer(res)[2])
# (0.7048434388423279, 0.8217521056921987)  #100
# original 0.7 0.8 
#(0.6938506474885056, 0.8123171786517321) with #5000 obs

#-----------------------------------------------------------------------------------#
## Random Walk

σ_η = 0.6
σ_ϵ = 0.8
n = 5000

B = reshape([1.0], 1, 1)
Z = copy(T)
Q = reshape([σ_η^2], 1, 1)
H = reshape([σ_ϵ^2], 1, 1)


## Simlate Random Walk
η = rand(Normal(0, σ_η), n)
ϵ = rand(Normal(0, σ_ϵ), n)

μ= repeat([reshape([0.0], 1, 1)], n)
[μ[i] = μ[i-1] + reshape([η[i]], 1,1) for i in 2:n]


y_RW = repeat([reshape([0.0], 1, 1)], n)
[y_RW[i] = μ[i] + reshape([ϵ[i]], 1,1) for i in 1:n] 
vec_yRW = [y_RW[i][1, 1] for i in 1:n]

## Initial State Value
x0 = reshape([0.0], 1,1)

include("stationary_P.jl")
P0 = stationary_P(T, Q)

## Applying kalman_filter
y_hat = kalman_filter(T, Z, Q, H, P0, x0, y_RW)[1]


## Plots
plot(vec_yRW,xlabel="Time", label ="Simulated Random Walk", legend=:bottomright)
plot!(vec_y_hat, label ="Estimate from Kalman Filter", legend=:bottomright)
savefig("figs\\RW.png")


## MLE estimate
# For the Random Walk process we need to estimate two Parameters σ_ϵ and σ_η
function log_L_RW(θ::Vector, y)  # θ= [log_σ_ϵ, log_σ_η]
    T = length(y)

    σ_ϵ = exp(θ[1])
    σ_η = exp(θ[2])
    Q1 = reshape([σ_η^2], 1, 1)
    H1 = reshape([σ_ϵ^2], 1, 1)

    x_t_tm1 = kalman_filter(B, Z, Q1, H1, P0, x0, y)[1]
    P_t_tm1 = kalman_filter(B, Z, Q1, H1, P0, x0, y)[3]

    v1 = [y[i] - Z*x_t_tm1[i] for i in 1:T]
    vec_v = [v1[i][1,1] for i in 1:T]
    F1 = [Z*P_t_tm1[i]*Z' + H1 for i in 1:T]
    Ft = abs.(det.(vec(F1)))

    return sum([-0.5*log(Ft[i]) - 0.5*vec_v[i]'*inv(Ft[i])*vec_v[i] for i in 1:T])
end


## Optimization
res = optimize(θ->-log_L_RW(θ, y_RW), [-0.1, -0.1])
σ_η_hat, σ_ϵ_hat= exp(Optim.minimizer(res)[2]), exp(Optim.minimizer(res)[1])
# original 0.6, 0.8
# (0.5799585949271968, 0.8148006904037058)