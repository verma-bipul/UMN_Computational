# PS1 Quant2 :

## Loading Packages
using DataFrames
using CSV
using Plots
using Random
using Distributions
using Optim
using BenchmarkTools

## Calculating GDP growth rate
df = CSV.read("us_gdp_data/gdp_clean.csv", DataFrame) # Loading GDP data
year = df.year   #get years
gdp = df.gdp    #get gdp

g1 = [(gdp[i+1]-gdp[i])/gdp[i] for i in 1:length(gdp)-1]   #obating growth rate
g2 = [log(gdp[i+1]) - log(gdp[i]) for i in 1:length(gdp)-1]

plot(year[2:end], g2, xlabel="year", ylabel="gdp_growth_rate", label="GDP growth rate 1947-2021", legend=:bottomleft)
savefig("figs\\gdp_growth.png")



## 2-(a)
Random.seed!(456)

## Since We are using formula from HW3

σ_ϵ = 0.10
σ_ν = 0.15

n = length(g2)

B = reshape([1.0], 1, 1)
Z = copy(B)
Q = reshape([σ_ν^2], 1, 1)
R = reshape([σ_ϵ^2], 1, 1)

y_gdp = [reshape([g2[i]], 1,1) for i in 1:n]

x0 = reshape([0.02], 1,1)
P0 = copy(x0)


## Obtaining ̂μₜₜ₋₁
include("kalman.jl")
mu_t_tm1 = kalman_filter(B, Z, Q, R, P0, x0, y_gdp)[1]
vec_mu_t_tm1 = [mu_t_tm1[i][1, 1] for i in 1:length(mu_t_tm1)]

plot(year, vec_mu_t_tm1 , xlabel="year", ylabel="mu_hat | y_t-1", label="Estimated value of state", legend=:bottomleft)
savefig("figs\\2_a.png")



## 2(b) Test Cases: Simulations
ν = rand(Normal(0, σ_ν), n)
ϵ = rand(Normal(0, σ_ϵ), n)

μ1 = 0.02
μ= repeat([reshape([μ1], 1, 1)], n)
[μ[i] = μ[i-1] + reshape([ν[i]], 1,1) for i in 2:n]
vec_μ = [μ[i][1,1] for i in 1:n]

y_RW = repeat([reshape([0.0], 1, 1)], n)
[y_RW[i] = μ[i] + reshape([ϵ[i]], 1,1) for i in 1:n] 

## Application Kalman Filter using simulated y values

x_hat_sim = kalman_filter(B, Z, Q, R, P0, x0, y_RW)[1]
vec_x_hat_sim = [x_hat_sim[i][1, 1] for i in 1:length(x_hat_sim)]

σ_μ = sqrt.([kalman_filter(B, Z, Q, R, P0, x0, y_RW)[3][i][1,1] for i in 1:length(vec_x_hat_sim)])  

x_sim_UCI = vec_x_hat_sim + 2*σ_μ
x_sim_LCI = vec_x_hat_sim - 2*σ_μ

## 2-(b) Plots

plot(vec_μ, xlabel="time", ylabel="mu", label="Simulated Values", legend=:bottomleft)
plot!(vec_x_hat_sim, label="Estimated mu Kalman Filter" )
plot!(x_sim_UCI, label="mu + sigma" )
plot!(x_sim_LCI, label="mu - sigma")
savefig("figs\\2_b.png")


## Cheking our likelihood estimate using the simulated case:

function log_L(θ::Vector, y)  # θ= [log_σ_ϵ, log_σ_ν]  # σ_ϵ = 0.10 , σ_ν = 0.15
    T = length(y)

    σ_ϵ = exp(θ[1])
    σ_ν = exp(θ[2])
    Q1 = reshape([σ_ν^2], 1, 1)
    R1 = reshape([σ_ϵ^2], 1, 1)

    x_t_tm1 = kalman_filter(B, Z, Q1, R1, P0, x0, y)[1]
    P_t_tm1 = kalman_filter(B, Z, Q1, R1, P0, x0, y)[3]

    v1 = [y[i] - Z*x_t_tm1[i] for i in 1:T]
    vec_v = [v1[i][1,1] for i in 1:T]
    F1 = [Z*P_t_tm1[i]*Z' + R1 for i in 1:T]
    Ft = abs.(det.(vec(F1)))

    return sum([-0.5*log(Ft[i]) - 0.5*vec_v[i]'*inv(Ft[i])*vec_v[i] for i in 1:T])
end


## MLE estimates for simulated values
res = optimize(θ->-log_L(θ, y_RW), [-2.0, -2.0])
σ_ϵ_hat, σ_ν_hat= exp(Optim.minimizer(res)[1]), exp(Optim.minimizer(res)[2]) 
# est (0.09191757969082172, 0.17214392397722658)
# actual 0.10, 0.15
log_L([log(σ_ϵ_hat), log(σ_ν_hat) ], y_RW)  #105.72
log_L([log(0.10), log(0.15) ], y_RW)  #105.1   

# MLE function seems to work well

## Using other Optimization Method
res = optimize(θ->-log_L(θ, y_RW), [-2.0, -2.0], BFGS())
σ_ϵ_hat, σ_ν_hat= exp(Optim.minimizer(res)[1]), exp(Optim.minimizer(res)[2])  # (0.09191757969082172, 0.17214392397722658)
log_L([log(σ_ϵ_hat), log(σ_ν_hat) ], y_RW)  #105.72
log_L([log(0.10), log(0.15) ], y_RW)  #105.1  
# same results

## Optimization on actual data
res = optimize(θ->-log_L(θ, y_gdp), [-2.0, -3.0], BFGS())
σ_ϵ_hat, σ_ν_hat= exp(Optim.minimizer(res)[1]), exp(Optim.minimizer(res)[2]) # (0.02245809354495799, 0.0016098692615514217)
log_L([log(σ_ϵ_hat), log(σ_ν_hat)], y_gdp) # 238.93978596397653



## Graphing: 
σ_ϵ = 0.0224
σ_ν = 0.00161

n = length(g2)

T = reshape([1.0], 1, 1)
Z = copy(T)
Q = reshape([σ_ν^2], 1, 1)
H = reshape([σ_ϵ^2], 1, 1)

y_gdp = [reshape([g2[i]], 1,1) for i in 1:n]

x0 = reshape([0.02], 1,1)
P0 = copy(x0)


## Obtaining ̂μₜₜ₋₁
include("kalman_filter.jl")
x_hat = kalman_filter(B, Z, Q, H, P0, x0, y_gdp)[1]
vec_x = [x_hat[i][1, 1] for i in 1:n]   # converting to vector for plotting

plot(year[2:end], vec_x , xlabel="year", ylabel="mu_hat | y_t-1", label="Estimated state after MLE", legend=:topright)
savefig("figs\\3_b.png")

##-------------------------------NO CHANGES REQ ABOVE---------------##

## Loading EM
include("EM1.jl")


## Lets run the EM algorithm for simulated case with 100 iterations 
Q_est, R_est = EM_update([0.04, 0.04], y_RW)[1][1,1], EM_update([0.04, 0.04], y_RW)[2][1,1]
σ_ν_em, σ_ϵ_em = sqrt(Q_est), sqrt(R_est) #(0.1444551166732994, 0.10848435716362345)  actual: (0.15, 0.10) 

Q_est, R_est = EM_update2([0.04, 0.04], y_RW)[1][1,1], EM_update2([0.04, 0.04], y_RW)[2][1,1]
σ_ν_em, σ_ϵ_em = sqrt(Q_est), sqrt(R_est) #(0.14711079564979315, 0.10702671839904873)  actual: (0.15, 0.10)
# the EM_update2 give more precise estimates

log_L([log(σ_ϵ_em), log(σ_ν_em)], y_RW)  # the EM log LL is lower #104.8
log_L([log(σ_ϵ_hat), log(σ_ν_hat)], y_RW)  # 105.7


# The estimates are very close are everything works well

## Implementing the EM on actual data 
Q_est, R_est = EM_update([0.005, 0.005], y_gdp, 1000)[1][1,1], EM_update([0.005, 0.005], y_gdp, 1000)[2][1,1]
σ_ν_em, σ_ϵ_em = sqrt(Q_est), sqrt(R_est)   #(0.019560722670778168, 0.01759905203706767)

Q_est, R_est = EM_update2([0.005, 0.005], y_gdp)[1][1,1], EM_update2([0.005, 0.005], y_gdp)[2][1,1]
σ_ν_em, σ_ϵ_em = sqrt(Q_est), sqrt(R_est) #(0.021888946670865543, 0.016820414947110406)



## Lets compare the likelihoods
log_L([log(σ_ϵ_em), log(σ_ν_em)], y_gdp)  # the EM log LL is lower # 227
log_L([log(σ_ϵ_hat), log(σ_ν_hat)], y_gdp) # the MLE is higher # 238  σ_ϵ, σ_ν = 0.0224, 0.00161





##----------------------------------------------------------##

## Checking things for single variable case: 
log_Ls(x) = -log_L([x[1], log(0.15)], y_RW)

res2 = optimize(x->log_Ls(x), [-2.0], BFGS())
exp(Optim.minimizer(res2)[1])   #0.10538683034861751

plot(y->log_Ls(y), xlims=(-4, 0))

## Applying EM to sinlge variable case
sqrt(EM_update2([0.15^2, 0.04], y_RW)[1,1])

## Definig EM that updates only R1

function EM_update2(θ_ini::Vector, y, Maxiter::Integer = 100, Tol::AbstractFloat=1e-8) # θ = [Q, R] are variances
    T = length(y)

    Q_old = reshape([θ_ini[1]], 1, 1)
    R_old = reshape([θ_ini[2]], 1, 1)

    i=1
 
    while i<Maxiter 
        
        x_t_T= kalman_smoother(B, Z, Q_old, R_old, P0, x0, y)[1]
        P_t_T = kalman_smoother(B, Z, Q_old, R_old, P0, x0, y)[2]
        P_t_tm1_T = copy(P_t_T)
        [P_t_tm1_T[i] = J[i-1]*P_t_T[i] for i in 2:T]
        
         α = sum([y[i]*y[i]' for i in 1:T])
         δ = sum([y[i]*x_t_T[i] for i in 1:T])
         γ = sum([P_t_T[i] + x_t_T[i]*x_t_T[i] for i in 1:T])
         β = sum([P_t_tm1_T[i] + x_t_T[i]*x_t_T[i-1] for i in 2:T])
        
         γ1 = γ - P_t_T[end] - x_t_T[end]*x_t_T[end] 
         γ2 = γ - P_t_T[1] - x_t_T[1]*x_t_T[1] 
        
         R_new = reshape([(1/T)*((α - δ*inv(γ)*δ')[1,1])], 1,1)
        
         if maximum([norm(R_new - R_old)]) < Tol
            return R_new
         end
        
         R_old = copy(R_new)

         i=i+1

    end      
    
    println("Method did not converge. The last iteration $i, gives $Q_old,  $R_old") 

end

## -----------------SOME CHECKS ABOVE --------------------------##