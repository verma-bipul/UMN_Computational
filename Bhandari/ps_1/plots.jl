# Plots

# A subroutines that plots the estimates of state for state space in HW1

function plot_est_state(y::Vector, σ_ϵ::Float64, σ_ν::Float64)
    n = length(y)

T = reshape([1.0], 1, 1)
Z = copy(T)
Q = reshape([σ_ν^2], 1, 1)
H = reshape([σ_ϵ^2], 1, 1)


y_mat = [reshape([y[i]], 1,1) for i in 1:n]
x0 = reshape([0.02], 1,1)
P0 = copy(x0)

include("kalman_filter.jl")
x_hat = kalman_filter(T, Z, Q, H, P0, x0, y_mat, n)[1]
vec_x = [x_hat[i][1, 1] for i in 1:n]   # converting to vector for plotting

plot(year[2:end], vec_x , xlabel="year", ylabel="mu_hat | y_t-1", label="Estimated value of state", legend=:bottomleft)
    
end

## 
