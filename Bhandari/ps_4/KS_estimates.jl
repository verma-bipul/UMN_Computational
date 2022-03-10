#This module contains the main estimates of the KS models
# Reference: https://github.com/QuantEcon/krusell_smith_code/blob/master/KSfunctions.ipynb

## Loading Packages
using Parameters, Dierckx, Interpolations, LinearAlgebra, Plots, GLM

include("KS_functions.jl")
include("MarkovC_Simulation.jl")

Household = @with_kw (
# All these parameters are the same as used in original KS 

β = 0.99,
δ = 0.025,
θ = 0.36,
μ = 1,

Na = 100, #no of grid points for k
NK = 4, #no of grid points for K
Ns = 4, #no of states

a_grid = get_grids_MMV(0.0, 1000, Na, 7),
amat = repeat(a_grid, 1, NK, Ns),  # This creates a matrix where a_grid, K_grid is repeated for each s
# Note: i denotes index for a, j for K, m for s=(z, ϵ)

Kmin = 30,
Kmax = 50,

K_grid = collect(LinRange(Kmin, Kmax , NK)),    
Kmat = [K_grid[j] for i in 1:Na, j in 1:NK, m in 1:Ns],


z_grid = [1.01, 0.99],
zmat = [m==1||m==3 ? z_grid[1] : z_grid[2]  for i in 1:Na, j in 1:NK, m in 1:Ns],

ϵ_grid = [1.0, 0.0],
ϵmat = [m==1||m==2 ? ϵ_grid[1] : ϵ_grid[2]  for i in 1:Na, j in 1:NK, m in 1:Ns],

ug=0.04,
ub=0.1,
zg_ave_dur=8,
zb_ave_dur=8,
ug_ave_dur=1.5,
ub_ave_dur=2.5,
puu_rel_gb2bb=1.25,
puu_rel_bg2gg=0.75,

dg = Categorical([0.96, 0.04]), # this is the denisty of unenmp, emp ig g
db = Categorical([0.90, 0.10]),  # this is the denisty of unenmp, emp ig g

N_bar = 1/(1-ub),   # this is the aggregate labor
Nmat = [m==1||m==3 ? N_bar*(1-ug) : N_bar*(1-ub)  for i in 1:Na, j in 1:NK, m in 1:Ns],

Rmat = 1 .+ get_r(θ, δ, zmat, Kmat, Nmat),
wmat = get_w(θ, zmat, Kmat, Nmat ),  # based in the aggregate labor supply

c_bind = Rmat.*amat .+ wmat.*ϵmat.*N_bar,

Probs = get_transition_matrix(ug, ub, zg_ave_dur, zb_ave_dur, ug_ave_dur, ub_ave_dur, puu_rel_gb2bb, puu_rel_bg2gg)

)

hh = Household()
# At this point check if all the grid matrices have been correctly defined

Uc(c, μ) = c.^(-μ)
Uc_inv(c, μ) = c.^(-1/μ) 


# We'll define the extend the policy function to be defined as a 2D spline interpolation on the discrete grid

function get_c_spline(a_grid, K_grid, c_old, Ns)
    c_new = Array{Spline2D}(undef, Ns)
    
    for m in 1:Ns
        c_new[m] = Spline2D(a_grid, K_grid, c_old[:, :, m])  #input the x-axis, y-axis, and z-axis
    end

    return c_new
end


function get_a_spline(a_grid, K_grid, a_pol, Ns)
    a_new = Array{Spline2D}(undef, Ns)

    for m in 1:Ns
        a_new[m] = Spline2D(a_grid, K_grid, a_pol[:, :, m])
    end

    return a_new
end


# We'll write a function to calculate K' based on b vector
function get_K_prime(b, zmat, Kmin, Kmax, Kmat)
    Kpg = @. clamp(exp(b[1] + b[2]*log(Kmat)), Kmin, Kmax) #clamp restricts the values to be b/w Kmin and Kmax
    Kpb = @. clamp(exp(b[3] + b[4]*log(Kmat)), Kmin, Kmax)

    return @. Kpg*(zmat==1.01) + Kpb*(zmat==0.99) 
end 

Kp = get_K_prime(b_MMV, hh.zmat, hh.Kmin, hh.Kmax, hh.Kmat)

    
# We'll have to evaluate the policy function at a', K', z' vals

function eval_cspline(a_grid, amat, K_grid, Kp, Na, NK, Ns, c_old)
    c_spline =  get_c_spline(a_grid, K_grid, c_old, Ns)
    
    c_evals = zeros(Na, NK, Ns)

    for m in 1:Ns
        c_evals[:, :, m] = @. c_spline[m](amat[:, :, m], Kp[:, :, m]) # notice that we are evaluating our old policy at (a, Kp)
    end

    return c_evals
end


#eval_cspline(hh.a_grid, hh.amat, hh.K_grid, Kp, hh.Ns, hh.c_bind)
"""
The `get_endo_c` function works as follows:       
    - it will take as input c_old i.e the old policy function, and K'
    - it then evaluates the policy at (a, K') for each z,
    - finally, it calculates the endogenous c^* based on the Euler
"""
function get_endo_c(β, δ, θ, μ, zmat, Nmat, amat, a_grid, K_grid, Kp, Ps, NK, Na, Ns, c_old)

    c_temp = eval_cspline(a_grid, amat, K_grid, Kp, Na, NK, Ns, c_old)
    B = @. β*(1+get_r(θ, δ, zmat, Kp, Nmat))*Uc(c_temp, μ)  # the RHS of euler has been evaluated at all grid pts

    for j in 1:NK
        B[:, j, :] = B[:, j, :]*Ps' 
    end  
    return Uc_inv(B, μ) 
end

#c_old = @. hh.c_bind -  hh.Rmat*hh.amat 
#c_old = zeros(100, 4, 4)

ct = get_endo_c(hh.β, hh.δ, hh.θ, hh.μ, hh.zmat, hh.Nmat, hh.amat, hh.a_grid, hh.K_grid, Kp, hh.Probs.Ps, hh.NK, hh.Na, hh.Ns, c_old)


function get_endo_a(ct, amat, ϵmat, Rmat, wmat, N_bar)
    return @. (1/Rmat)*(ct + amat - wmat*ϵmat*N_bar)     
end

at = get_endo_a(ct, hh.amat, hh.ϵmat, hh.Rmat, hh.wmat, hh.N_bar)

function get_c_new(β, δ, θ, μ, zmat, Nmat, amat, ϵmat, Rmat, wmat, a_grid, K_grid, Kp, Ps, NK, Na, Ns, N_bar, cbind, c_old)
    """
    This function gives the updated guess for the policy function c_new using EGM
    """

    ct = get_endo_c(β, δ, θ, μ, zmat, Nmat, amat, a_grid, K_grid, Kp, Ps, NK, Na, Ns, c_old)
    at =  get_endo_a(ct, amat, ϵmat, Rmat, wmat, N_bar)

    cnonbind = similar(ct)

    for j in 1:NK, m in 1:Ns
        cnonbind[:, j, m] = Spline1D(at[:, j, m], ct[:, j, m]).(amat[:, j, m])
        #cnonbind[:, j, m] = LinearInterpolation(at[:, j, m], ct[:, j, m], extrapolation_bc=Line()).(amat[:, j, m])

    end

    for j in 1:NK, m in 1:Ns
        ct[:, j, m] = (amat[:,j, m] .> at[1,j, m]).*cnonbind[:,j, m] .+ (amat[:,j, m] .<= at[1,j, m]).*cbind[:,j, m]
    end

    return ct
end

#get_c_new(hh.β, hh.δ, hh.θ, hh.μ, hh.zmat, hh.Nmat, hh.amat, hh.ϵmat, hh.Rmat, hh.wmat, hh.a_grid, hh.K_grid, Kp, hh.Ps, hh.NK, hh.Na, hh.Ns, hh.N_bar, hh.c_bind, hh.c_bind)



function pol_EGM(hh, b; tol=1e-10, maxiter=100_00)
    @unpack β, μ, δ, θ, Probs, zmat, Nmat, amat, Rmat, wmat, ϵmat, Kmat, a_grid, K_grid, Kmin, Kmax, Na, NK, Ns, N_bar, c_bind = hh

    Kp =  get_K_prime(b, zmat, Kmin, Kmax, Kmat)

    c_old = @. c_bind -  Rmat*amat 
    #c_old = zeros(Na, NK, Ns)

    iter = 1
    normdiff = 100.0

    while iter < maxiter && normdiff > tol  # this process can be parallelized --- learn this
        c_new = get_c_new(β, δ, θ, μ, zmat, Nmat, amat, ϵmat, Rmat, wmat, a_grid, K_grid, Kp, Probs.Ps, NK, Na, Ns, N_bar, c_bind, c_old)
    
        normdiff = norm(c_new - c_old)
        iter = iter + 1
        c_old = copy(c_new)
    
    end

    a_new = @. Rmat*amat + wmat*ϵmat*N_bar - c_old
   #a_new = @. clamp(Rmat*amat + wmat*ϵmat*N_bar - c_old, a_grid[1], a_grid[end])


    println("The normdiff in EGM is $normdiff, iter = $iter")
    
    return  a_new, c_old
end


# Check the policy function at the true b, do they look sensible ???
a_new= @time pol_EGM(hh, b_MMV)[1] # 7 sec to run 1 iteration in i3-7gen, 2.4 Ghz


## Few Plots
plot(hh.a_grid[1:50], a_new[1:50, 1, 3], lab="g0")
plot(hh.a_grid[1:100], a_new[1:100, 3, 1], lab="g1")
plot!(hh.a_grid[1:100], hh.a_grid[1:100], color=:black, linestyle=:dash, lab="45 degree", width=0.5)

a_pol = get_a_spline(hh.a_grid, hh.K_grid, a_new, hh.Ns)


####################################
## MMV #############################
###################################

# carrting out policy function iteration as in MMV

function get_a_new_MMV(a_old, a_grid, K_grid, amat, ϵmat, Kmat, Rmat, wmat, Na, NK, Ns, β, θ, δ, zmat, Kp,Nmat, N_bar,  μ, Ps)

   a_pol = get_a_spline(a_grid, K_grid, a_old, Ns)

   c_next = zeros(Na, NK, Ns)
   ap = zeros(Na, NK, Ns)
   app = zeros(Na, NK, Ns)

   for m in 1:Ns
    ap[:, :, m] = @. a_pol[m](amat[:, :, m], Kmat[:, :, m]) 
   end

   for m in 1:Ns
    app[:, :, m] = @. a_pol[m](ap[:, :, m], Kp[:, :, m]) # notice that we are evaluating our old policy at (a, Kp)
   end

    c_next = @. (1+get_r(θ, δ, zmat, Kp, Nmat))*ap + get_w(θ, zmat, Kp, Nmat)*N_bar*ϵmat - app 

    B = @. β*(1+get_r(θ, δ, zmat, Kp, Nmat))*Uc(c_next, μ)

    for j in 1:NK  # this step calculate the expectation
        B[:, j, :] = B[:, j, :]*Ps' 
    end  

    RHS = Uc_inv(B, μ) 

    return @. clamp(Rmat*amat + wmat*ϵmat*N_bar - RHS, a_grid[1], a_grid[end])       
end


function pol_MMV(hh, b; tol=1e-8, maxiter=100_00)
    @unpack β, μ, δ, θ, Probs, zmat, Nmat, amat, Rmat, wmat, ϵmat, Kmat, a_grid, K_grid, Kmin, Kmax, Na, NK, Ns, N_bar, c_bind = hh

    Kp =  get_K_prime(b, zmat, Kmin, Kmax, Kmat)

    a_old = copy(c_bind)

    iter = 1
    normdiff = 100.0

    while iter < maxiter && normdiff > tol  # this process can be parallelized --- learn this
        a_new = get_a_new_MMV(a_old, a_grid, K_grid, amat, ϵmat, Kmat, Rmat, wmat, Na, NK, Ns, β, θ, δ, zmat, Kp,Nmat, N_bar,  μ, Probs.Ps)

        normdiff = maximum(abs, a_new - a_old)
        iter = iter + 1
        a_old = @. 0.7*a_new + 0.3*a_old
        #a_old = copy(a_new)
    
    end

    println("The normdiff in MMV is $normdiff, iter = $iter")
    
    return  a_old
end

b_MMV = [0.153389, 0.959254, 0.142, 0.960953]
@time a_new_MMV = pol_MMV(hh, b_MMV)


a_polMMV = get_a_spline(hh.a_grid, hh.K_grid, a_new, hh.Ns)


#####################################
#     SIMULATION                    #
#####################################


## Based on the aggregate state in the last period and current aggregate state

draw_eps_shock!(zi::Val{1}, zi_lag::Val{1},  epsi_lag, P) = draw_eps_shock!( epsi_lag, P.Peps_gg)
draw_eps_shock!(zi::Val{1}, zi_lag::Val{2},  epsi_lag, P) = draw_eps_shock!( epsi_lag, P.Peps_gb)
draw_eps_shock!(zi::Val{2}, zi_lag::Val{1},  epsi_lag, P) = draw_eps_shock!( epsi_lag, P.Peps_gb)
draw_eps_shock!(zi::Val{2}, zi_lag::Val{2},  epsi_lag, P) = draw_eps_shock!( epsi_lag, P.Peps_bb)
    
function draw_eps_shock!(epsi_shock_before, Peps)
    N = length(epsi_shock_before)
    eps_shock = zeros(N)

    for i in 1:N  # Looping over the population
        if epsi_shock_before[i] ==1  # employed in the last period
            eps_shock[i] = rand(Categorical(Peps[1, :]))
        else epsi_shock_before[i] ==2
            eps_shock[i] = rand(Categorical(Peps[2, :]))
        end
    end

    return eps_shock
end



# Now we'll use the policy function to carry out simulation and update the guess for b until convergence

T= 11000
N = 5000


function convert_2_s(Z_sim, ϵ_sim, N, T)
    S_sim = zeros(N, T)

    for t in 1:T, i in 1:N
        S_sim[i, t] = Z_sim[t] + 2*(ϵ_sim[i, t]-1)
    end

    return S_sim

end

function simulate_shocks(Probs, T, N, dg, db, ug, ub)

    Z_sim = markov_simulate(Probs.Pz, T )  # This is the simulation for aggregate shock for T periods
    
    ϵ_sim = zeros(N, T)  # preallocation

    # Simulating ϵ shock for 1st period
    if Z_sim[1] == 1   # 1 means G
        ϵ_sim[:, 1] = rand(dg, N) # in good times draw shocks such that ug = 0.04  # 1 means employed
    elseif Z_sim[1] == 2
        ϵ_sim[:, 1] = rand(db, N) # in good times draw shocks such that ug = 0.1
    else
        error("The value of z shock $Z_sim[1] is strange")
    end

    # Simulating shocks for the remaining periods using sim_eps_shock function

    for t in 2:T
        ϵ_sim[:, t] = draw_eps_shock!(Val(Z_sim[t]), Val(Z_sim[t-1]), ϵ_sim[:, t-1], Probs)
    end

    # We need to make sure that the unemployement rate depends only on the aggregate shocks
    # with large N the probs enure this, with small N we need to make some corrections

    # adjustment
    for t=1:T
        n_u = count(ϵ_sim[:,t].==2) # count number of unemployed
        ur_ideal = ifelse(Z_sim[t] == 1, ug, ub)
        gap = ur_ideal*N - n_u
        if gap > 0 # make emp --> unemp
            change_emp_unemp = rand(findall(1 .== ϵ_sim[:,t]), Int(gap))
            ϵ_sim[change_emp_unemp, t] .= 2
        elseif gap < 0  # make unemp --> emp
           change_unemp_emp = rand(findall(2 .== ϵ_sim[:, t]), Int(-gap))
            ϵ_sim[change_unemp_emp, t] .= 1
        end 
    end    

    #S_sim = convert_2_s(Z_sim, ϵ_sim, N, T)
    return Z_sim, ϵ_sim
end

Random.seed!(0)
z_sim, ϵ_sim =  simulate_shocks(hh.Probs, 1100, 10000, hh.dg, hh.db, hh.ug, hh.ub)

@time S_sim = convert_2_s(z_sim, ϵ_sim, 10000, 1100)
    

## Note that we need not simulate the shocks again and again when calculating b'S


## Getting policies for the above shocks
"""
The `Simulate_K` function return the time series of aggregate capital given 

"""
function Simulate_K(a_pol, S_sim)
    N, T = size(S_sim)

    k_pol = fill(40.0, N)  #inital asset distribution

    K_sim = zeros(T)


    for t in 1:T
        K_sim[t] = mean(k_pol)
        for i in 1:N
            k_pol[i] = a_pol[Int(S_sim[i, t])](k_pol[i], K_sim[t])
        end        
    
    end

    return K_sim
 
end

a_pol = get_a_spline(hh.a_grid, hh.K_grid, a_new_MMV, hh.Ns )
K_sim = Simulate_K(a_pol, S_sim)

k_pol = fill(40.0, N)
for i in 1:N
    k_pol[i] = a_pol[Int(S_sim[i, 3])](k_pol[i], mean(k_pol))
end        

mean(k_pol)




## Given time series for K and simulated shocks carry out regression and get the R² coeff
"""
- find the index of good shocks and bad shocks,
- get yg ,Xg, yb, Xb accordingly 
"""
function get_reg_coeff_R2(K_sim, S_sim, b_old)
    index_g = findall(x-> (x==1)|| (x==3), S_sim[1, 101:end-1])  # 1 and 3 are the idices for good aggregate shock
    index_b = findall(x-> (x==2)|| (x==4), S_sim[1, 101:end-1]) 

    Xg = @. log(K_sim[100+index_g])
    yg =@. log(K_sim[100+index_g + 1])

    Xb = @. log(K_sim[100+index_b])
    yb =@. log(K_sim[100+index_b + 1])

    resg = lm([ones(length(Xg)) Xg], yg)
    resb = lm([ones(length(Xb)) Xb], yb)

    b_n = zeros(4)

    R2 = [r2(resg), r2(resb)]
    b_n[1], b_n[2] = coef(resg)
    b_n[3], b_n[4] = coef(resb)
    dif_b = maximum(abs, b_n-b_old)

    println("The reg coeff is $b_n with R2 = $R2, and diff = $dif_b")
    return b_n
end

get_reg_coeff_R2(K_sim, S_sim, [0.0225, 0.995, 0.015, 0.995 ])


## Write a function which iterates over the regression coefficients till convergence

function iterate_KS(hh, S_sim, b_old; maxiter=100, tol=1e-8)

    b_new = zeros(length(b_old))

    iter = 1
    normdiff = 100.0

    while iter < maxiter && normdiff > tol  # this process can be parallelized --- learn this
        a_new = pol_EGM(hh, b_old)[1]
        a_pol = get_a_spline(hh.a_grid, hh.K_grid, a_new, hh.Ns)
        K_sim = Simulate_K(a_pol, S_sim)  # Get the aggregate K for the simulated shocks

        b_new = get_reg_coeff_R2(K_sim, S_sim, b_old)
    
        normdiff = norm(b_new - b_old)
        iter = iter + 1
        b_old = @. 0.2*b_new + 0.8*b_old
        #b_old = copy(b_new)
    
    end

    println("The last iter is $iter, with normdiff = $normdiff")
    return b_old

end

@time iterate_KS(hh, S_sim, [0, 1, 0, 1]; maxiter=100, tol=1e-8)

# b = The reg coeff is [0.02838516420460977, 0.9617536929219482, 0.02060164941278167, 0.9721014406696694] with R2 = [0.9340406489413047, 0.9365387957204148] 

#################################
function iterate_KS_MMV(hh, S_sim, b_old; maxiter=100, tol=1e-8)

    b_new = zeros(length(b_old))

    iter = 1
    normdiff = 100.0

    while iter < maxiter && normdiff > tol  # this process can be parallelized --- learn this
        a_new = pol_MMV(hh, b_old)
        a_pol = get_a_spline(hh.a_grid, hh.K_grid, a_new, hh.Ns)
        K_sim = Simulate_K(a_pol, S_sim)  # Get the aggregate K for the simulated shocks

        b_new = get_reg_coeff_R2(K_sim, S_sim, b_old)
    
        normdiff = norm(b_new - b_old)
        iter = iter + 1
        b_old = @. 0.3*b_new + 0.7*b_old
    
    end

    println("The last iter is $iter, with normdiff = $normdiff")
    return b_old

end

@time iterate_KS_MMV(hh, S_sim, [0,1,0,1]; maxiter=100, tol=1e-8)




### ------------------ Compare the policy function using the two methods

