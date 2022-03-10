## Newton Root Finding Method
using ForwardDiff
using Plots

## Main Algo 
# Since we'll be working with economic varible which are all positive 
# We'll modify this algorithm to limit search for root only in the postive quadrant.

function Root_Newton(f::Function, x, Maxiter::Integer=1000, Tol::AbstractFloat=1e-10)
fprime = x -> ForwardDiff.derivative(f, x)
x_n=0.0
i=0
while i<Maxiter
x_n = maximum([x - f(x)/fprime(x), 1e-10])
if f(x_n)==0 || abs(x_n -x) < Tol
    return x_n
end
i=i+1
x=copy(x_n)
end
fval = f(x)
println("Method did not converge. The last iteration gives
    $x_n", " with fval = $fval") 
end


## Checks
# f1(x) = log(x) 
# Root_Newton(f1, 10)  # the modified function looks for roots only in the positive domian.


## Implementation on utility function
# c, k, α = 1, 3, 0.35
# Euler2(N) = 0.6*(1-N)/c - (0.65)*(k/N)^(0.35) # Method doesn't converge for these parameters
# Root_Newton(Euler2, 1)