# This subroutine calculates the root of a multivar function using Newton Root findinf Algo

using  ForwardDiff
using LinearAlgebra
using Plots

## Note that f:ℛᵐ → ℛⁿ 
function multivar_newt_root(f::Function, x0::Vector, Maxiter::Integer = 1000, Tol::AbstractFloat=1e-10)
    x_old = copy(x0)
    fprime = x -> ForwardDiff.jacobian(f, x)
    x_n = zeros(length(x0))
    i=0

    filler = 0.001*ones(size(fprime(x0)))
    while i<Maxiter
        x_n_temp = x_old - inv(fprime(x_old)+filler)*f(x_old)
        for i in 1:length(x_n)
            x_n[i] = maximum([x_n_temp[i], 1e-10])
        end

        if f(x_n)==zeros(length(x0)) || norm(x_n - x_old) < Tol
            return x_n
        end
        i = i+1
        x_old = copy(x_n)
    end
    fval = f(x_old)
    println("Method did not converge. The last iteration gives
        $x_n", " with fval = $fval") 
end

## Lets check our subroutine
#f1(x) = log.(x) 
#multivar_newt_root(f1, [10, 10])  # the subroutine works well
