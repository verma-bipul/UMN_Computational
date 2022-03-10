# Tauchen
# AR(1) Process: x' = (1-ρ)μₓ + ρx + N(0, σ)

##
using Distributions


function Tauchen(μ,  ρ, σ, n, m=3)
    σ_z = sqrt(σ^2/(1-ρ^2))

    z_max = μ+ m*σ_z
    z_min =μ-m*σ_z

    z = LinRange(z_min, z_max, n)  # creating a evenly spaced grid of discrete states


    step = (z_max - z_min)/(n - 1)
    half_step = 0.5 * step
    P = zeros(n, n)

    d = Normal(0, 1)

    for i in 1:n
        P[i, 1] = cdf(d, (z[1]- (1-ρ)*μ - ρ*z[i] + half_step)/σ)
        P[i, n] = 1 - cdf(d, (z[n] - (1-ρ)*μ - ρ*z[i] - half_step)/σ)
        for j in 2:n-1
            r = z[j] - ρ*z[i] - (1-ρ)*μ
            P[i, j] = cdf(d, (r + half_step)/σ) - cdf(d, (r - half_step)/σ)
        end
    end

    return z, P
end



