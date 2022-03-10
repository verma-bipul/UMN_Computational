# This program defines the utility function, production function etc. 


## Defining utility function and production function
function U(c, h)
    if c>0 && h<1 && h>0
        return log(c) + ψ*log(1-h)
    else
        return -Inf
    end    
end

function F_p(k, h, z)
    if k>0 && h<1 && h>0
        return (k^θ)*(z*h)^(1-θ)
    else
        return -Inf
    end             
end


function F_hat(k, h, z)
    if k>0 && h<1 && h>0
        return (k^θ)*(z*h)^(1-θ) + (1-δ)*k
    else
        return -Inf
    end             
end
