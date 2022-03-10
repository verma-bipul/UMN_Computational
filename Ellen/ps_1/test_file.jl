# Test File 
# This is to try out some stuffs

# understanding NLsolve
using NLsolve
using ForwardDiff
using Distributions
using Plots

# We'll try to solve the model for some simple cases now-

## Parameters
β = 0.96     # discount factor
ψ = 0.0     # labor supply coeff
θ = 0.33      # capital share in production
γ_z = 0.0   # productivity growth parameter     
γ_n = 0.0   # labor growth parameter
δ = 0.10    # depreciation
ρ = 0.90      # AR(1) coeff
k_max = 40   # maximum level of capital
σ=0.5    


## Let us generate AR(1) shocks

n = 1000
log_zz = zeros(n)
log_zz[1] = log(1)

for i in 1:n-1
    log_zz[i+1] = ρ*log_zz[i] + rand(Normal(0, σ))
end

##
plot(log_zz)

##
zz = exp.(log_zz)
plot(zz)

### 
z_dis = [0.5, 1, 1.5, 2, 2.5] # value of shocks have been taken keeping in mind that log_z is AR(1)
logz_dis = log.(z_dis)
m = [(logz_dis[i+1]+logz_dis[i])/2 for i in 1:4] # midpoints of shock realizations
P = zeros(5, 5) 
for i in 1:5
    for j in 2:4
        P[i, j] = cdf(Normal(), m[j] - ρ*logz_dis[i]) - cdf(Normal(), m[j-1] - ρ*logz_dis[i] )
    end
    P[i, 1] = cdf(Normal(), m[1] - ρ*logz_dis[i])
    P[i, 5] = 1-cdf(Normal(),m[4] - ρ*logz_dis[i])
end
