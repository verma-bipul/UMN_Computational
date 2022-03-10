# Date: Sep 16, 2021
# Bipul Verma

## This is a subroutine to do numerical qauadrature using Newton-Cotes Method
# Intput f:: function , interval [a. b], n:: desired no of intervals

function  Num_quad(f::Function, a::Real, b::Real, n::Int)
    h = (b-a)/(n-1)
    x = zeros(n)
    w = zeros(n)
    for i in 1:n
        x[i] = a + (i-1)*h
    end
    for i in 2:(n-1)
        w[i] = h
    end
    w[1] = h/2
    w[n] = h/2

    integral = sum([ w[i]*f(x[i]) for i in 1:n ]) 
    return integral
end


