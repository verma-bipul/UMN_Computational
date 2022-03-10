# Tauchen
# This submodule takes as input parameters for AR(1) process: μ, σ, m:: no of sd devn, n:: number of state
# The AR(1) process is z_{t+1} = ρ*z_t +  ϵ_{t+1} ; ϵ ∼ N(0, σ)

##
using Distributions

##

function Tauchen(ρ, σ, m=3, n=7)
    σ_z = sqrt(σ^2/(1-ρ^2))

    z_max = m*σ_z
    z_min = -z_max

    z = LinRange(z_min, z_max, n)  # creating a evenly spaced grid of discrete states


    step = (z_max - z_min)/(n - 1)
    half_step = 0.5 * step
    P = zeros(n, n)

    d = Normal(0, σ)

    for i in 1:n
        P[i, 1] = cdf(d, z[1]-ρ*z[i] + half_step)
        P[i, n] = 1 - cdf(d, z[n] - ρ*z[i] - half_step)
        for j in 2:n-1
            r = z[j] - ρ*z[i]
            P[i, j] = cdf(d, r + half_step) - cdf(d, r - half_step)
        end
    end

    return z, P
end

