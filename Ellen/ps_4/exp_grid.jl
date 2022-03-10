## This subroutine gives exponentially spaced grids of N::points (including a and b), b/w points a and b.

function exp_grid(a::Real, b::Real, N::Int)  
    exp_0(x) = exp(x) - exp(1)
    d = exp_0.(LinRange(log(a+exp(1)), log(b+exp(1)), N))
    return d
end