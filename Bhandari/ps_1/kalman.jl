# Test file

# We'll try to build improved version of the kalman filter and smoother:
# State: xₜ = Bxₜ₋₁ + νₜ ∼ N(0, Q) 
# Observation: yₜ = Zxₜ + ϵₜ ∼ N(0, R)

using LinearAlgebra

function kalman_filter(B::Matrix, Z::Matrix, Q::Matrix, R::Matrix, P0::Matrix, x0, y)
    T = length(y)

    x_t_tm1 = repeat([zeros(size(x0))], T+1)  #creating empty arrays for x
    x_t_t = repeat([zeros(size(x0))], T)

    P_t_tm1 = repeat([zeros(size(Q))], T+1)  #creating empty arrays for P
    P_t_t = repeat([zeros(size(Q))], T)

    x_t_tm1[1] = x0   #initialization 
    P_t_tm1[1] = P0
    
    G = repeat([zeros(size(Z*Q))], T)
    K = repeat([zeros(size(T*G[1]'*ones(size(H))))], T)  # matrix for storing K

    for i in 1:T
        K[i] = P_t_tm1[i]*Z'*inv(Z*P_t_tm1[i]*Z' + R)  
        x_t_t[i] = x_t_tm1[i] +  K[i]*(y[i] - Z*x_t_tm1[i])
        P_t_t[i] = P_t_tm1[i] -K[i]*Z*P_t_tm1[i]

        x_t_tm1[i+1] = B*x_t_t[i]
        P_t_tm1[i+1] = B*P_t_t[i]*B' + Q     
    end

    return x_t_tm1, x_t_t, P_t_tm1, P_t_t, K

end


function kalman_smoother(B::Matrix, Z::Matrix, Q::Matrix, R::Matrix, P0::Matrix, x0, y)
    T = length(y)

    x_t_tm1, x_t_t, P_t_tm1, P_t_t = kalman_filter2(B, Z, Q, R, P0, x0, y)

    x_t_T = copy(x_t_tm1)   # this automatically ensures initialization: x_t_T[T+1] = x_t_tm1[T+1] 
    P_t_T = copy(P_t_tm1)   # this ensures initialization P_t_T[T+1] = P_t_tm1[T+1]
    J = copy(P_t_t)         # just to enures that J is of size T
    
    for i in T:-1:1
        J[i] = P_t_t[i]*B'*inv(P_t_tm1[i+1])
        x_t_T[i] = x_t_t[i] + J[i]*(x_t_T[i+1] - x_t_tm1[i+1] )
        P_t_T[i] = P_t_t[i] + J[i]*(P_t_T[i+1] - P_t_tm1[i+1])*J[i]'
    end

    return x_t_T, P_t_T, J
    
end



#------------ Testing the code on simulated process ----------------------#

## 2(b) Test Cases: Simulations
σ_ϵ = 0.10
σ_ν = 0.15
n=100
ν = rand(Normal(0, σ_ν), n)
ϵ = rand(Normal(0, σ_ϵ), n)

μ1 = 0.02
x0 = reshape([0.02], 1,1)
P0 = copy(x0)


B = reshape([1.0], 1, 1)
Z = copy(B)
Q = reshape([σ_ν^2], 1, 1)
R = reshape([σ_ϵ^2], 1, 1)

μ= repeat([reshape([μ1], 1, 1)], n)
[μ[i] = μ[i-1] + reshape([ν[i]], 1,1) for i in 2:n]
vec_μ = [μ[i][1,1] for i in 1:n]

y_RW = repeat([reshape([0.0], 1, 1)], n)
[y_RW[i] = μ[i] + reshape([ϵ[i]], 1,1) for i in 1:n] 


## Checking our kalman filter function

# the function kalman_filter2 function works well !
# the kalman smoother function works well!
