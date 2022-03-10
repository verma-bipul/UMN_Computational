# This subroutine calculates the staionary P
using LinearAlgebra

function stationary_P(T::Matrix, Q::Matrix)
    P_old = Q*Q
    tol = 1e-6
    maxiter = 1000
    iter = 1
    normdiff = 100.0

    while normdiff > tol && iter <= maxiter
        P_new = T*P_old*T' + Q
        normdiff = opnorm(P_new - P_old)
        P_old = copy(P_new)
        iter = iter + 1
    end
    P = P_old

    #println("The maximum iteration is $iter ", " with normdiff = $normdiff")

    return P 
end