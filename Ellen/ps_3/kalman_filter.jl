# This subroutine applies kalman filter to a state space model.
# State: xₜ = Txₜ₋₁ + ηₜ ∼ N(0, Q) 
# Observation: yₜ = Zxₜ + ϵₜ ∼ N(0, H)

using LinearAlgebra

function kalman_filter(T::Matrix, Z::Matrix, Q::Matrix, H::Matrix, x0, y, n::Integer)
    x_hat = repeat([zeros(size(x0))], n)
    x_hat[1] = T*x0

    y_hat = repeat([zeros(size(y[1]))], n)

    v = copy(y_hat)

    P0 = stationary_P(T, Q)
    P = repeat([zeros(size(Q))], n)
    P[1] = T*P0*T' + Q
    
    F = repeat([zeros(size(H))], n)
    G = repeat([zeros(size(Z*Q))], n)
    K = repeat([zeros(size(T*G[1]'*ones(size(H))))], n)

    for i in 1:n-1
        y_hat[i] = Z*x_hat[i]

        v[i] = y[i] - y_hat[i]
        
        F[i] = Z*P[i]*Z' + H
        G[i] = Z*P[i]
        K[i] = T*G[i]'*inv(F[i])
        
        x_hat[i+1] =  T*x_hat[i] + K[i]*v[i]
        P[i+1] = T*(P[i]-G[i]'*inv(F[i])*G[i])*T' + Q
        
    end

    y_hat[n] = Z*x_hat[n]

    v[n] = y[n] - y_hat[n]
        
    F[n] = Z*P[n]*Z' + H
    G[n] = Z*P[n]
    K[n] = T*G[n]'*inv(F[n])

    return x_hat, v, y_hat,  P, F, G, K

end



           



    


