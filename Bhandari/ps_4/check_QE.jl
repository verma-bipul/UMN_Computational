## This file checks quant econ policy

using Interpolations # to use interpolation 
using Random, LinearAlgebra
using QuantEcon  # to use `gridmake`, `<:AbstractUtility`
#using Optim      # to use minimization routine to maximize RHS of bellman equation
using GLM        # to regress
#using JLD2       # to save the result
#using ProgressMeter # to show progress of iterations
using Parameters # to use type with keyword arguments


struct TransitionMatrix
    P::Matrix{Float64}       # 4x4 
    Pz::Matrix{Float64}      # 2x2 aggregate shock
    Peps_gg::Matrix{Float64} # 2x2 idiosyncratic shock conditional on good to good
    Peps_bb::Matrix{Float64} # 2x2 idiosyncratic shock conditional on bad to bad
    Peps_gb::Matrix{Float64} # 2x2 idiosyncratic shock conditional on good to bad
    Peps_bg::Matrix{Float64} # 2x2 idiosyncratic shock conditional on bad to good
end

abstract type UMPSolutionMethod end

@with_kw struct EulerMethod <: UMPSolutionMethod
    update_k::Float64 = 0.7
end

function create_transition_matrix(ug::Real, ub::Real,
    zg_ave_dur::Real, zb_ave_dur::Real,
    ug_ave_dur::Real, ub_ave_dur::Real,
    puu_rel_gb2bb::Real, puu_rel_bg2gg::Real)

# probability of remaining in good state
pgg = 1-1/zg_ave_dur
# probability of remaining in bad state
pbb = 1-1/zb_ave_dur
# probability of changing from g to b
pgb = 1-pgg
# probability of changing from b to g
pbg = 1-pbb  

# prob. of 0 to 0 cond. on g to g
p00_gg = 1-1/ug_ave_dur
# prob. of 0 to 0 cond. on b to b
p00_bb = 1-1/ub_ave_dur
# prob. of 0 to 1 cond. on g to g
p01_gg = 1-p00_gg
# prob. of 0 to 1 cond. on b to b
p01_bb = 1-p00_bb

# prob. of 0 to 0 cond. on g to b
p00_gb=puu_rel_gb2bb*p00_bb
# prob. of 0 to 0 cond. on b to g
p00_bg=puu_rel_bg2gg*p00_gg
# prob. of 0 to 1 cond. on g to b
p01_gb=1-p00_gb
# prob. of 0 to 1 cond. on b to g
p01_bg=1-p00_bg

    # prob. of 1 to 0 cond. on  g to g
    p10_gg=(ug - ug*p00_gg)/(1-ug)
    # prob. of 1 to 0 cond. on b to b
    p10_bb=(ub - ub*p00_bb)/(1-ub)
    # prob. of 1 to 0 cond. on g to b
    p10_gb=(ub - ug*p00_gb)/(1-ug)
    # prob. of 1 to 0 cond on b to g
    p10_bg=(ug - ub*p00_bg)/(1-ub)
    # prob. of 1 to 1 cond. on  g to g
    p11_gg= 1-p10_gg
    # prob. of 1 to 1 cond. on b to b
    p11_bb= 1-p10_bb
    # prob. of 1 to 1 cond. on g to b
    p11_gb= 1-p10_gb
    # prob. of 1 to 1 cond on b to g
    p11_bg= 1-p10_bg
    
    #   (g1)         (b1)        (g0)       (b0)
    P=[pgg*p11_gg pgb*p11_gb pgg*p10_gg pgb*p10_gb;
       pbg*p11_bg pbb*p11_bb pbg*p10_bg pbb*p10_bb;
       pgg*p01_gg pgb*p01_gb pgg*p00_gg pgb*p00_gb;
       pbg*p01_bg pbb*p01_bb pbg*p00_bg pbb*p00_bb]
    Pz=[pgg pgb;
        pbg pbb]
    Peps_gg=[p11_gg p10_gg
             p01_gg p00_gg]
    Peps_bb=[p11_bb p10_bb
             p01_bb p00_bb]
    Peps_gb=[p11_gb p10_gb
             p01_gb p00_gb]
    Peps_bg=[p11_bg p10_bg
             p01_bg p00_bg]
    transmat=TransitionMatrix(P, Pz, Peps_gg, Peps_bb, Peps_gb, Peps_bg)
    return transmat
end


function KSParameter(;
    beta::AbstractFloat=0.99,
    alpha::AbstractFloat=0.36,
    delta::Real=0.025,
    theta::Real=1,
    k_min::Real=0,
    k_max::Real=1000,
    k_size::Integer=100,
    K_min::Real=30,
    K_max::Real=50,
    K_size::Integer=4,
    z_min::Real=0.99,
    z_max::Real=1.01,
    z_size::Integer=2,
    eps_min::Real=0.0,
    eps_max::Real=1.0,
    eps_size::Integer=2,
    ug::AbstractFloat=0.04,
    ub::AbstractFloat=0.1,
    zg_ave_dur::Real=8,
    zb_ave_dur::Real=8,
    ug_ave_dur::Real=1.5,
    ub_ave_dur::Real=2.5,
    puu_rel_gb2bb::Real=1.25,
    puu_rel_bg2gg::Real=0.75,
    mu::Real=0, 
    degree::Real=7)
if theta == 1
u = LogUtility()
else
u = CRRAUtility(theta)
end

l_bar=1/(1-ub)
    # individual capital grid
    k_grid=
        (range(0, stop=k_size-1, length=k_size)/(k_size-1)).^degree*(k_max-k_min).+k_min   
    k_grid[1] = k_min; k_grid[end] = k_max; # adjust numerical error
    # aggregate capital grid
    K_grid=range(K_min, stop=K_max, length=K_size)
    # aggregate technology shock
    z_grid=range(z_max, stop=z_min, length=z_size)
    # idiosyncratic employment shock grid
    eps_grid=range(eps_max, stop=eps_min, length=eps_size)
    s_grid=gridmake(z_grid, eps_grid)               # shock grid
    # collection of transition matrices
    transmat=create_transition_matrix(ug,ub,
        zg_ave_dur,zb_ave_dur,
        ug_ave_dur,ub_ave_dur,
        puu_rel_gb2bb,puu_rel_bg2gg)

    ksp=(u=u, beta=beta, alpha=alpha, delta=delta, theta=theta,
         l_bar=l_bar, k_min=k_min, k_max=k_max, k_grid=k_grid,
         K_min=K_min, K_max=K_max, K_grid=K_grid, z_grid=z_grid,
         eps_grid=eps_grid, s_grid=s_grid, k_size=k_size, K_size=K_size,
         z_size=z_size, eps_size=eps_size, s_size=z_size*eps_size, 
         ug=ug, ub=ub, transmat=transmat, mu=mu)

    return ksp
end

r(alpha::Real, z::Real, K::Real, L::Real)=alpha*z*K^(alpha-1)*L^(1-alpha)
w(alpha::Real,z::Real,K::Real,L::Real)=(1-alpha)*z*K^(alpha)*L^(-alpha)

mutable struct KSSolution
    k_opt::Array{Float64,3}
    value::Array{Float64,3}
    B::Vector{Float64}
    R2::Vector{Float64}
end

function KSSolution(ksp::NamedTuple;
    load_value::Bool=false,
    load_B::Bool=false,
    filename::String="result.jld2")
if load_value || load_B
result=load(filename)
kss_temp=result["kss"]
end
if load_value
k_opt=kss_temp.k_opt
value=kss_temp.value
else
k_opt=ksp.beta*repeat(ksp.k_grid,outer=[1,ksp.K_size,ksp.s_size])
k_opt=0.9*repeat(ksp.k_grid,outer=[1,ksp.K_size,ksp.s_size])
k_opt .= clamp.(k_opt, ksp.k_min, ksp.k_max)
value=ksp.u.(0.1/0.9*k_opt)/(1-ksp.beta)
end
if load_B
B = kss_temp.B
else
B = b_MMV
end
kss = KSSolution(k_opt, value, B, [0.0, 0.0])
return kss
end



function solve_ump!(umpsm::EulerMethod, 
    ksp::NamedTuple, kss::KSSolution;
    max_iter::Integer=10000,
    tol::AbstractFloat=1e-8)
alpha, beta, delta, theta, l_bar, mu = 
ksp.alpha, ksp.beta, ksp.delta, ksp.theta, ksp.l_bar, ksp.mu
k_grid, k_size = ksp.k_grid, ksp.k_size
K_grid, K_size = ksp.K_grid, ksp.K_size
s_grid, s_size = ksp.s_grid, ksp.s_size
k_min, k_max = ksp.k_min, ksp.k_max
global counter = 0
k_opt_n = similar(kss.k_opt)
#prog = ProgressThresh(tol, "Solving individual UMP by Euler method: ")
while true
global counter += 1
for s_i = 1:s_size
z, eps = s_grid[s_i, 1], s_grid[s_i, 2]
for (K_i, K) = enumerate(K_grid)
Kp, L = compute_Kp_L(K,s_i,kss.B,ksp)
for (k_i, k) = enumerate(k_grid)
    wealth = (r(alpha,z,K,L)+1-delta)*k+
                w(alpha,z,K,L)*(eps*l_bar + mu*(1-eps))
    expec=compute_expectation_FOC(kss.k_opt[k_i, K_i, s_i], Kp, s_i, ksp)
    cn = (beta*expec)^(-1.0/theta)
    k_opt_n[k_i, K_i, s_i] = wealth-cn
end
end
end
k_opt_n .= clamp.(k_opt_n, k_min, k_max)
dif_k = maximum(abs, k_opt_n - kss.k_opt)
#ProgressMeter.update!(prog, dif_k)
if dif_k < tol
    break
end
if counter >= max_iter
    @warn "Euler method failed to converge with $counter iterations (dif = $dif_k)"
    break
end
kss.k_opt .= umpsm.update_k*k_opt_n .+ (1-umpsm.update_k)*kss.k_opt
end
return kss.k_opt
end


function compute_expectation_FOC(kp::Real,
    Kp::Real,
    s_i::Integer,
    ksp::NamedTuple)
alpha, theta, delta, l_bar, mu, P =
ksp.alpha, ksp.theta, ksp.delta, ksp.l_bar, ksp.mu, ksp.transmat.P
global expec = 0.0
for s_n_i = 1:ksp.s_size
zp, epsp = ksp.s_grid[s_n_i, 1], ksp.s_grid[s_n_i, 2]
Kpp, Lp = compute_Kp_L(Kp, s_n_i, kss.B, ksp)
rn = r(alpha, zp, Kp, Lp)
kpp = interpolate((ksp.k_grid, ksp.K_grid), kss.k_opt[:, :, s_n_i], Gridded(Linear()))
cp = (rn+1-delta)*kp + w(alpha, zp ,Kp, Lp)*(epsp*l_bar+mu*(1.0-epsp))-kpp(kp, Kp)
global expec = expec + P[s_i, s_n_i]*(cp)^(-theta)*(1-delta+rn)
end 
return expec
end


function compute_Kp_L(K::Real, s_i::Integer,
    B::AbstractVector, ksp::NamedTuple)
Kp, L=ifelse(s_i%ksp.eps_size == 1,
(exp(B[1]+B[2]*log(K)), ksp.l_bar*(1-ksp.ug)), # if good
(exp(B[3]+B[4]*log(K)), ksp.l_bar*(1-ksp.ub))) # if bad
Kp = clamp(Kp, ksp.K_min, ksp.K_max)
return Kp, L
end


# instance of KSParameter
ksp = KSParameter()
# instance of KSSolution
kss = KSSolution(ksp, load_value=false, load_B=false)


@time solve_ump!(EulerMethod(), ksp, kss)

## Everthing is stored in kss.k_opt 
# Lets us try to understand why the two solutions differ

### Shock Generation:
function generate_shocks(ksp::NamedTuple;
    z_shock_size::Integer = 1100,
    population::Integer = 10000)

# unpack parameters
Peps_gg = ksp.transmat.Peps_gg
Peps_bg = ksp.transmat.Peps_bg
Peps_gb = ksp.transmat.Peps_gb
Peps_bb = ksp.transmat.Peps_bb

# draw aggregate shock
zi_shock = simulate(MarkovChain(ksp.transmat.Pz), z_shock_size)

### Let's draw individual shock ###
epsi_shock = Array{Int}(undef, z_shock_size, population) # preallocation

# first period
rand_draw=rand(population)
# recall: index 1 of eps is employed, index 2 of eps is unemployed
if zi_shock[1] == 1 # if good
epsi_shock[1, :] .= (rand_draw .< ksp.ug) .+ 1 # if draw is higher, become employed 
elseif zi_shock[1] == 2 # if bad
epsi_shock[1, :] .= (rand_draw .< ksp.ub) .+ 1 # if draw is higher, become employed
else
error("the value of z_shocks[1] (=$(z_shocks[1])) is strange")
end

# from second period ...   
for t = 2:z_shock_size
draw_eps_shock!(Val(zi_shock[t]), Val(zi_shock[t-1]),
   view(epsi_shock, t, :), epsi_shock[t-1, :], ksp.transmat)
end

# adjustment
for t=1:z_shock_size
n_e = count(epsi_shock[t,:].==1) # count number of employed
empl_rate_ideal = ifelse(zi_shock[t] == 1, 1.0-ksp.ug, 1.0-ksp.ub)
gap = round(Int, empl_rate_ideal*population) - n_e
if gap > 0
become_employed_i = rand(findall(2 .== epsi_shock[t,:]), gap)
epsi_shock[t, become_employed_i] .= 1
elseif gap < 0
become_unemployed_i = rand(findall(1 .== epsi_shock[t, :]), -gap)
epsi_shock[t,become_unemployed_i] .= 2
end 
end
    
return zi_shock, epsi_shock    
end


draw_eps_shock!(zi::Val{1}, zi_lag::Val{1}, epsi, 
                epsi_lag::AbstractVector, transmat::TransitionMatrix) = 
    draw_eps_shock!(epsi, epsi_lag, transmat.Peps_gg)
draw_eps_shock!(zi::Val{1}, zi_lag::Val{2}, epsi, 
                epsi_lag::AbstractVector, transmat) = 
    draw_eps_shock!(epsi, epsi_lag, transmat.Peps_bg)
draw_eps_shock!(zi::Val{2}, zi_lag::Val{1}, epsi, 
                epsi_lag::AbstractVector, transmat) = 
    draw_eps_shock!(epsi, epsi_lag, transmat.Peps_gb)
draw_eps_shock!(zi::Val{2}, zi_lag::Val{2}, epsi, 
                epsi_lag::AbstractVector, transmat) = 
    draw_eps_shock!(epsi, epsi_lag, transmat.Peps_bb)


function draw_eps_shock!(epsi_shocks,
        epsi_shock_before,
        Peps::AbstractMatrix)
# loop over entire population
for i=1:length(epsi_shocks)
rand_draw=rand()
epsi_shocks[i]=ifelse(epsi_shock_before[i] == 1,
             (Peps[1, 1] < rand_draw)+1,  # if employed before
             (Peps[2, 1] < rand_draw)+1)  # if unemployed before
end
return nothing
end

## Let us start by comparing if the simulation are the same
# generate shocks
Random.seed!(0) # for reproducability
@time zi_shocks, epsi_shocks =generate_shocks(ksp;
        z_shock_size=1100, population=10000);

## Comparing the simulations

abstract type SimulationMethod end

struct Stochastic <: SimulationMethod
    epsi_shocks::Matrix{Int}
    k_population::Vector{Float64}
end

Stochastic(epsi_shocks::Matrix{Int}) = 
    Stochastic(epsi_shocks, fill(40, size(epsi_shocks, 2)))



function simulate_aggregate_path!(ksp::NamedTuple, kss::KSSolution,
    zi_shocks::AbstractVector, K_ts::Vector, sm::Stochastic)
epsi_shocks, k_population = sm.epsi_shocks, sm.k_population

T = length(zi_shocks)   # simulated duration
N=size(epsi_shocks, 2) # number of agents
# loop over T periods
for (t, z_i) = enumerate(zi_shocks)
    K_ts[t] = mean(k_population) # current aggrgate capital
    
    # loop over individuals
    for (i, k) in enumerate(k_population)
        eps_i = epsi_shocks[t, i]   # idiosyncratic shock
        s_i = epsi_zi_to_si(eps_i, z_i, ksp.z_size) # transform (z_i, eps_i) to s_i
        # obtain next capital holding by interpolation
        itp_pol = interpolate((ksp.k_grid, ksp.K_grid), kss.k_opt[:, :, s_i], Gridded(Linear()))
        k_population[i] = itp_pol(k, K_ts[t])
    end
end
return K_ts
end
epsi_zi_to_si(eps_i::Integer, z_i::Integer, z_size::Integer) = z_i + ksp.z_size*(eps_i-1)


K_ts = Vector{Float64}(undef, length(zi_shocks))
sm = Stochastic(epsi_shocks)
K_ts = simulate_aggregate_path!(ksp, kss, zi_shocks, K_ts, sm)

