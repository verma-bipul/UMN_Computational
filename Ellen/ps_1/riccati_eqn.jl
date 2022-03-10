# This subroutine/function will take as input matrices A, B, Q, W and β.
# It will return matrix P and F after iteration.

## Loading Relevant Packages
using LinearAlgebra


function Riccati(A::Matrix, B::Matrix, R:: Matrix, Q::Matrix, W:: Matrix, β::Real)
    P_old = zeros(size(R)) # preallocation of matrices
    F_old = zeros(size(W'))

    normdiff = 100.0
    iter = 1
    tolerance = 1e-5
    maxiter = 1000

    while normdiff > tolerance && iter < maxiter
        P_new = R + β*A'*P_old*A -(β*A'*P_old*B + W)*inv(Q + β*B'*P_old*B)*(β*B'*P_old*A + W')
        F_new = inv(Q + β*B'*P_old*B)*(β*B'*P_old*A + W')
        
        normdiff = maximum([opnorm(P_new - P_old), opnorm(F_new - F_old)])

        P_old = copy(P_new)
        F_old = copy(F_new)

        iter = iter+1
    end

    P = P_old
    F = F_old

    #println("The maximum iteration is $iter ", " with normdiff = $normdiff")
    return P, F
end


### Redefinign Riccati

function RiccatiMM(A::Matrix, B::Matrix, R::Matrix, Q::Matrix, W:: Matrix, β::Real)
    A_t = sqrt(β*(1+γ_n))*(A - B*inv(R)*W')
    B_t = sqrt(β*(1+γ_n))*B
    Q_t = Q - W*inv(R)*W'


    P_old = zeros(size(Q_t)) # preallocation of matrices
    F_old = inv(R + B_t'*P_old*B_t)*(B_t'*P_old*A_t)

    normdiff = 100.0
    iter = 1
    tolerance = 1e-5
    maxiter = 1000

    while normdiff > tolerance && iter < maxiter
        P_new = Q_t + A_t'*P_old*A_t -(A_t'*P_old*B_t)*inv(R + B_t'*P_old*B_t)*(B_t'*P_old*A_t)
        
        F_new = inv(R + B_t'*P_new*B_t)*(B_t'*P_new*A_t)
        
        normdiff = maximum([opnorm(P_new - P_old), opnorm(F_new - F_old)])

        P_old = copy(P_new)
        F_old = copy(F_new)
        iter = iter+1
    end

    P = P_old
    F = F_old + inv(R)*W'

    #println("The maximum iteration is $iter ", " with normdiff = $normdiff")
    return P, F
end