# This program defines the utility function, production function etc. 

include("parameters.jl")

## Defining utility function and production function
function U(c::Real, h::Real)
    if c>0 && h<1 && h>0
        return log(c) + ψ*log(1-h)
    else
        return -Inf
    end    
end

function F_p(k::Real, h::Real, z::Real)
    if k>0 && h<1 && h>0
        return (k^θ)*(z*h)^(1-θ)
    else
        return -Inf
    end             
end


function F_hat(k::Real, h::Real, z::Real)
    if k>0 && h<1 && h>0
        return (k^θ)*(z*h)^(1-θ) + (1-δ)*k
    else
        return -Inf
    end             
end

## Writing a function to obtain C as a 