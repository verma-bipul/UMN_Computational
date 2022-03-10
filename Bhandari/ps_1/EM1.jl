## Let us write the code for EM algorithm:


function EM_update(θ_ini::Vector, y, Maxiter::Integer = 100, Tol::AbstractFloat=1e-8) # θ = [Q, R] are variances
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
         δ = sum([y[i]*x_t_T[i]' for i in 1:T])
         γ = sum([P_t_T[i] + x_t_T[i]*x_t_T[i]' for i in 1:T])
         β = sum([P_t_tm1_T[i] + x_t_T[i]*x_t_T[i-1]' for i in 2:T])
        
         γ1 = γ - P_t_T[end] - x_t_T[end]*x_t_T[end]' 
         γ2 = γ - P_t_T[1] - x_t_T[1]*x_t_T[1]' 
        
         Q_new = reshape([(1/(T-1))*((γ2 - β*inv(γ1)*β')[1,1])], 1,1)
         R_new = reshape([(1/T)*((α - δ*inv(γ)*δ')[1,1])], 1,1)
        
         if maximum([norm(Q_new - Q_old), norm(R_new - R_old)]) < Tol
            return Q_new, R_new
         end
        
         Q_old = copy(Q_new)
         R_old = copy(R_new)

         i=i+1

    end      
    
    println("Method did not converge. The last iteration $i, gives $Q_old,  $R_old") 

end



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

        α = [y[i]*y[i]' for i in 1:T]
        δ = [y[i]*x_t_T[i]' for i in 1:T]
        γ = [P_t_T[i] + x_t_T[i]*x_t_T[i]' for i in 1:T]
        β = [P_t_tm1_T[i] + x_t_T[i]*x_t_T[i-1]' for i in 2:T]
        
        
         Q_new = mean([γ[i] - β[i-1]*B - B*β[i-1] + B*γ[i-1]*B' for i in 2:T])
         R_new = mean([α[i] - δ[i]*Z' - Z*δ[i]' + Z*γ[i]*Z' for i in 1:T])
        
         if maximum([norm(Q_new - Q_old), norm(R_new - R_old)]) < Tol
            return Q_new, R_new
         end
        
         Q_old = copy(Q_new)
         R_old = copy(R_new)

         i=i+1

    end      
    
    println("Method did not converge. The last iteration $i, gives $Q_old,  $R_old") 

end

