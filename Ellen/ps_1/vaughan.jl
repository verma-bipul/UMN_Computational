# This subroutine returns the P matrix using the Vaughan Method

## Loading Packages
using LinearAlgebra

## Calculation of P and F using the Vaughan Method

function Vaughan(A::Matrix, B::Matrix, Q::Matrix, R:: Matrix, W::Matrix, β::Real)
    A_t = sqrt(β)*(A - B*inv(Q)*W')
    B_t = sqrt(β)*B
    R_t = R - W*inv(Q)*W'

    H = [inv(A_t)  inv(A_t)*B_t*inv(Q)*B_t' ; R_t*inv(A_t)  R_t*inv(A_t)*B_t*inv(Q)*B_t'+ A_t']
    V = eigen(H).vectors
    
    L = Int(size(V)[1]/2)
    # We need to split the matrix  V into 4 equal parts
    V12 = V[1:L, L+1:end]
    V22 = V[L+1:end, L+1:end]
    P = V22*inv(V12)
    F = inv(Q + β*B'*P*B)*(β*B'*P*A + W')

    return P, F
end

