# This module contains all the necessary function to solve the Aiyagari model
# One needs to define the household instance containing all the parameters before running the functions

# module to get grids

function get_grids(amin, amax, np, power)

    grid = LinRange(amin^(1/power), amax^(1/power), np)
    return grid.^power
end


function get_cmin(ϵmat, τy, w, ψ, frish, μ, T, Na, Ne)
    cmin = zeros(Na, Ne)
    for i in 1:Na, j in 1:Ne
        cmin[i, j] = find_zero(c-> (1-τy)*w*ϵmat[1, 1]*n(c, ϵmat[1, 1], w, τy, ψ, frish, μ) + T -c, 1.0)
    end
    return cmin
end


function get_endo_c(β, r, τy, μ, Pϵ, c_old)  # this function has been optimized for speed
    B = β*(1+r*(1-τy))*Uc(c_old, μ)*Pϵ'  
    return Uc_inv(B, μ) 
end


get_endo_a(Amat, ϵmat, r, τy, ψ, w, frish, μ, T, ct) = @. 1/(1+(1-τy)*r)*(ct + Amat - (1-τy)*w*ϵmat*n(ct, ϵmat, w, τy, ψ, frish, μ) - T)


function get_cbind(Amat, ϵmat, r, τy, w, ψ, frish, μ, T, Na, Ne)
    cbind = zeros(Na, Ne)
    if ψ==0.0
    cbind = (1+r)*Amat + w*ϵmat
    else 
        for i in 1:Na, j in 1:Ne
            cbind[i, j] = find_zero(c-> (1+(1-τy)*r)*Amat[i, j] + (1-τy)*w*ϵmat[i, j]*n(c, ϵmat[i, j], w, τy, ψ, frish, μ) + T -c, 1.0)
        end
    end
    return cbind
end


function get_c_new(r, w, β, Pϵ, τy, ψ, frish, μ, T, c_old, cbind, Amat, ϵmat, Ne)
    
    ct = get_endo_c(β, r, τy, μ, Pϵ, c_old)
    at = get_endo_a(Amat, ϵmat, r, τy, ψ, w, frish, μ, T, ct)

    #c_new = zeros(Na, Ne)
    cnonbind = similar(ct)


    for j in 1:Ne
        cnonbind[:, j] = LinearInterpolation(at[:, j], ct[:, j], extrapolation_bc=Line()).(Amat[:, j])
    end

    
    for j = 1:Ne
        ct[:,j] = (Amat[:,j] .> at[1,j]).*cnonbind[:,j] .+ (Amat[:,j] .<= at[1,j]).*cbind[:,j]
    end

    return ct
end


function get_c_spline(Amat, c_old, Ne)
    c_new = Array{Spline1D}(undef, Ne)

    for j in 1:Ne
        c_new[j] = Spline1D(Amat[:, j], c_old[:, j]; k=1)
    end

    return c_new
end


function get_a_spline(Amat, a_pol, Ne)
    a_new = Array{Spline1D}(undef, Ne)

    for j in 1:Ne
        a_new[j] = Spline1D(Amat[:, j], a_pol[:, j]; k=1)
    end

    return a_new
end


function pol_EGM(hh, r, T, ψ, A; tol=1e-8, maxiter=100_00)

    @unpack β, μ, frish, δ, θ, τy, Pϵ, Amat, AmatL, ϵmat, ϵmatL, Na, NaL, Ne = hh

    w = (1-θ)*A*(((r+δ)/(A*θ))^(-θ/(1-θ)))

    cbind = get_cbind(Amat, ϵmat, r, τy, w, ψ, frish, μ, T, Na, Ne)
    #c_old = get_cmin(ϵmat, τy, w, ψ, frish, μ, T, Na, Ne)
    #c_old = @. r*Amat+w*ϵmat+T
    c_old = copy(cbind)

    iter = 1
    normdiff = 100.0

    while iter < maxiter && normdiff > tol  # this process can be parallelized --- learn this
        c_new = get_c_new(r, w, β, Pϵ, τy, ψ, frish, μ, T, c_old, cbind, Amat, ϵmat, Ne)
    
        normdiff = opnorm(c_new - c_old)
        iter = iter + 1
        c_old = copy(c_new)
    
    end

    c_new = zeros(NaL, Ne)

    cspl = get_c_spline(Amat, c_old, Ne)

    for j in 1:Ne
        c_new[:, j] = @. cspl[j](AmatL[:, j])
    end

    n_new = @. n(c_new, ϵmatL, w, τy, ψ, frish, μ)
    a_new = @. (1+(1-τy)*r)*AmatL + (1-τy)*w*ϵmatL*n_new + T - c_new
    
    return  a_new, c_new, n_new
end


# Everything is correct till this point of time.
# EGM is fast. Convergence take less than 0.3 seconds on i3-7gen 2.4Ghz processor

# We'll define the tranistion matrix and finally the employ the root finding 

## Building the transition matrix

function Qtran(a_grid, a_policy, Pϵ, Na, Ne)
    # for now suppose that a_policy is a matrix with size conforming to a_grid
    # we'll later change this so that a_grid and a_pol can be different

    Q = spzeros(Na*Ne, Na*Ne)

    for j in 1:Ne
        sj = (j-1)*Na

        for i in 1:Na
            k = searchsortedlast(a_grid[:,j], a_policy[i, j]) # return the index of the last value in a_grid less than or equal to a'(a_i, e_j)

            if (0 < k && k <Na) # this is to adjust for first and last grid points
                k = k
            elseif k == Na
                k = Na-1
            else
                k = 1
            end

            #(0 < k && k <Na) ? k = k : (k==Na) ? k = Na-1: k = 1  

            wk = (a_policy[i, j] - a_grid[k, j])/(a_grid[k+1, j] - a_grid[k, j])
            wk = min(max(wk, 0.0), 1.0)

            for m in 1:Ne
                tm = (m-1)*Na
                Q[k+tm, i+sj] = (1-wk)*Pϵ[j, m]
                Q[k+1+tm, i+sj] = wk*Pϵ[j, m]
            end

        end

    end

    return Q

end


# Function to calculate the stationary distribution

function get_st_dis(Q, Na, Ne)
    Ns = Na*Ne

    iter = 1
    normdiff = 100
    λ_old = repeat([1/Ns], Ns, 1)

    while normdiff > 1e-16
       λ_new = Q*λ_old

       normdiff = opnorm(λ_new - λ_old)
       iter = iter + 1

       λ_old = copy(λ_new)

    end
    return λ_old    
end



## Getting the aggregates which will be calirated to meet the targets !!

function get_aggregates(hh, r, T, ψ, A)

    println("The input values are r = $r, T=$T, ψ= $ψ, A=$A")
    # As the root finder will run it'll keep printing the values that its seaching over
    # Its nice to keep track of the progress

    @unpack β, μ, frish, δ, θ, τy, Pϵ, Amat, AmatL, ϵmat, ϵmatL, Na, NaL, Ne, ϕg, ϕb = hh

    w = (1-θ)*A*(((r+δ)/(A*θ))^(-θ/(1-θ)))

    a_new, n_new = pol_EGM(hh, r, T, ψ, A)[[1, 3]] 

    Q = Qtran(AmatL, a_new, Pϵ, NaL, Ne)  #Note that everything is being evaluated at finer grid points
    λ_st = get_st_dis(Q, NaL, Ne)

    A_new = sum(vec(a_new).*λ_st)
    N_new = sum(vec(n_new).*vec(ϵmatL).*λ_st)

    Kdd = N_new*((r+δ)/θ)^(-1/(1-θ))

    Ydd = A*(Kdd^θ)*(N_new^(1-θ))

    G = ϕg*Ydd
    B = ϕb*Ydd

    GBC_resi = τy*(w*N_new + r*A_new) - (G + r*B + T)
    Asset_mkt_resi = Kdd + B - A_new

    println("The Y is $Ydd, N is $N_new, govt buget residual is $GBC_resi, asset market residual is $Asset_mkt_resi")

    return N_new -0.28, Ydd -1.0, GBC_resi, Asset_mkt_resi, A_new
end

function get_st_a(hh, a_new)
    @unpack AmatL, Pϵ, NaL, Ne = hh

    Q = Qtran(AmatL, a_new, Pϵ, NaL, Ne)  #Note that everything is being evaluated at finer grid points
    λ_st = reshape(get_st_dis(Q, NaL, Ne), NaL, Ne)
    λ_a = sum(λ_st, dims=2)

    return λ_a, λ_st    
end


# Function for getting plots
function get_a_plots(Nend, hh, a_new)
    plot(hh.AmatL[1:Nend, 1], a_new[1:Nend,1], label="low_shock", title="Asset Policy Aiyagari with endo L", xlabel="Current Assets", ylabel="Savings")
    plot!(hh.AmatL[1:Nend, 3], a_new[1:Nend,3], label="medium_shock", legend=:topleft)
    plot!(hh.AmatL[1:Nend, 5], a_new[1:Nend,5], label="high_shock")
end


function get_c_plots(Nend, hh, c_new)
    plot(hh.AmatL[1:Nend, 1], c_new[1:Nend,1], label="low_shock", title="Asset Policy Aiyagari with endo L", xlabel="Current Assets", ylabel="Savings")
    plot!(hh.AmatL[1:Nend, 3], c_new[1:Nend,3], label="medium_shock", legend=:topleft)
    plot!(hh.AmatL[1:Nend, 5], c_new[1:Nend,5], label="high_shock")
end

function get_n_plots(Nend, hh, n_new)
    plot(hh.AmatL[1:Nend, 1], n_new[1:Nend,1], label="low_shock", title="Asset Policy Aiyagari with endo L", xlabel="Current Assets", ylabel="Savings")
    plot!(hh.AmatL[1:Nend, 3], n_new[1:Nend,3], label="medium_shock", legend=:topleft)
    plot!(hh.AmatL[1:Nend, 5], n_new[1:Nend,5], label="high_shock")
end

function get_wealth_dis_plot(Nend, hh, a_new)

    λ_a, λ_st = get_st_a(hh, a_new)  #obtaining the row sums
    plot(hh.AmatL[1:Nend, 1], λ_a[1:Nend], label="Density", title="Asset Disbn in Aiyagari endo L", xlabel="Asset Level")
end

