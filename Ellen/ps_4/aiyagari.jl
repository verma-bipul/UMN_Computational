# This file contains estimation for the basic Aiyagari model with stop

## Loading Packges
using LinearAlgebra
using Plots
using NLsolve

## Setting Parameters (Taken from Fran)
α = 0.3
β = 0.9
δ = 0.05
μ = 2

lL = 0.4  # low labor ss
lH = 1    # high labor ss

ζ = 30000
n = 20   # total number of grid points
# p = 2

r = 0.04
w = (((r+δ)/α)^(-α/(1-α)))*(1-α)

## Loading Modules
include("linear_basis.jl")
include("exp_grid.jl")

## a_grid = [exp_grid(0.01, 0.2, 5); exp_grid(0.25, 1, 4)]  # We can adjust the grid later as well
a_min, a_max = 0.0, 20
x = LinRange(a_min^0.2, a_max^0.2, n)
a_grid = x.^5
##
#a_pL(a::Real, θL::Vector) = sum([θL[i]*ψ(a_grid, a, i) for i in 1:n])
#a_pH(a::Real, θH::Vector) = sum([θH[i]*ψ(a_grid, a, i) for i in 1:n])

a_p(a::Real, θ::Vector) = sum([θ[i]*ψ(a_grid, a, i) for i in 1:n])

## Writing out the residual equation for l = L and l = H

function RL(assets::Real, θ::Vector)
    θL, θH = θ[1:n], θ[n+1:2*n] 
    cL(a) = w*lL +(1+r)*a - a_p(a, θL)
    cH(a) = w*lH + (1+r)*a - a_p(a, θH)  
    c_pLL(a) = w*lL + (1+r)*a_p(a, θL) - a_p(a_p(a, θL), θL) 
    c_pLH(a) = w*lH + (1+r)*a_p(a, θL) - a_p(a_p(a, θL), θH) 
    c_pHL(a) = w*lL + (1+r)*a_p(a, θH) - a_p(a_p(a, θH), θL) 
    c_pHH(a) = w*lH + (1+r)*a_p(a, θH) - a_p(a_p(a, θH), θH)
    RL = β*(1+r)*(0.5*c_pLH(assets)^(-μ) + 0.5*c_pLL(assets)^(-μ)) + β*ζ*minimum([a_p(assets, θL), 0.0])^2 - cL(assets)^(-μ)
    return RL
end

function RH(assets::Real, θ::Vector)
    θL, θH = θ[1:n], θ[n+1:2*n] 
    cL(a) = w*lL +(1+r)*a - a_p(a, θL)
    cH(a) = w*lH + (1+r)*a - a_p(a, θH)  
    c_pLL(a) = w*lL + (1+r)*a_p(a, θL) - a_p(a_p(a, θL), θL) 
    c_pLH(a) = w*lH + (1+r)*a_p(a, θL) - a_p(a_p(a, θL), θH) 
    c_pHL(a) = w*lL + (1+r)*a_p(a, θH) - a_p(a_p(a, θH), θL) 
    c_pHH(a) = w*lH + (1+r)*a_p(a, θH) - a_p(a_p(a, θH), θH)
    RH = β*(1+r)*(0.5*c_pHH(assets)^(-μ) + 0.5*c_pHL(assets)^(-μ)) + β*ζ*minimum([a_p(assets, θH), 0.0])^2 - cH(assets)^(-μ)
    return RH
end


## Calculating weighted residuals
# first we need to obtain weighted residual for each i 
# collect all the weigthed residulas together to form a system of equations.
include("num_quadrature.jl")

function w_res_RL(θ::Vector, i::Int)
    if i==1
        return Num_quad(a ->ψ(a_grid, a, i)*RL(a, θ), a_grid[i], a_grid[i+1], 4)
    elseif i==n
        return Num_quad(a ->ψ(a_grid, a, i)*RL(a, θ), a_grid[i-1], a_grid[i], 4)
    else
        return Num_quad(a ->ψ(a_grid, a, i)*RL(a, θ), a_grid[i-1], a_grid[i+1], 4)
    end    
end

function w_res_RH(θ::Vector, i::Int)
    if i==1
        return Num_quad(a ->ψ(a_grid, a, i)*RH(a, θ), a_grid[i], a_grid[i+1], 4)
    elseif i==n
        return Num_quad(a ->ψ(a_grid, a, i)*RH(a, θ), a_grid[i-1], a_grid[i], 4)
    else
        return Num_quad(a ->ψ(a_grid, a, i)*RH(a, θ), a_grid[i-1], a_grid[i+1], 4)
    end    
end

## Now we just need to find the roots of w_aiya function
# For inital Guess we'll use the a_grid points itself (this assumes that the policy fn is 45 deg line)
θ_inital = [a_grid; a_grid .+ 0.4]

## Trying out build in  multivar newt root
# We'll use the multivar newton root
include("multivar_newt_root.jl")
w_aiya(θ::Vector) = vcat([w_res_RL(θ, i) for i in 1:n], [w_res_RH(θ, i) for i in 1:n])

## Applying multivar newton root  
θ_est =  multivar_newt_root(w_aiya,θ_inital, 1000)
θ_est = [0.1710016450030469, 0.0863632329975809, 0.061912694563855813, 0.01796595928386001, 1.0e-10, 1.0e-10, 1.0e-10, 1.0e-10, 1.0e-10, 1.0e-10, 1.0e-10, 1.0e-10, 1.0e-10, 1.0e-10, 1.0e-10, 1.0e-10, 1.0e-10, 1.0e-10, 1.0e-10, 1.0e-10, 1.0e-10, 205.22335078294628, 1.0e-10, 1.0e-10, 1.0e-10, 1.0e-10, 1.0e-10, 1.0e-10, 1.0e-10, 1.0e-10, 1.0e-10, 1.0e-10, 1.0e-10, 1.0e-10, 1.0e-10, 1.0e-10, 1.0e-10, 1.0e-10, 1.0e-10, 1.0e-10, 1.0e-10, 0.016294526391468885, 0.06872630070317515, 0.203066939129922, 0.5066952145939823, 1.1377300125947498, 2.4101135064556662, 4.957323901191395, 10.032821755247816, 20.38297337795326, 1.4796957935789679, 0.3047540390793688, 0.5829865001825337, 0.5083117900210115, 0.44751210835763167, 0.4062647213926861, 0.38176540005451354, 0.3683433774461575, 0.36130925556823196, 0.3577074408803629, 0.35588472834993623, 0.35496789930915684, 0.354508103051148, 0.3542778374497006, 0.3541626292495543, 0.3541050041170709, 0.3540761843384214, 0.3540617760545484, 0.35405457478753616, 0.3540509810615347, 0.35404919807883256, 0.3540483342759534, 0.3540479577379705, 0.3540478802801006, 0.35404806315829973, 0.3540485978009761, 0.3540497515357817, 0.35405210122920366, 0.35405682172912817, 0.3540662732839862, 0.3540851816725491, 0.35412300109828376, 0.354198641261343, 0.3543499224516518, 0.3546524848006956, 0.3552576141136122, 0.35646786113129697, 0.35888845594631863, 0.3637293099978458, 0.37341337458087154, 0.39277233919885524, 0.43154724726674626, 0.5088476959579182, 0.6649117029998757, 0.9812477999363586, 1.6164969295458704, 2.889766322814488, 5.435672241114238, 10.509608844016823, 20.874077819297632] 


## Let us try plotting policy function

a_pol_L = [a_p(a_grid[i], θ_est[1:n]) for i in 1:length(a_grid)]
a_pol_H = [a_p(a_grid[i], θ_est[n+1:2*n]) for i in 1:length(a_grid)]

plot(a_grid[1:45], a_pol_L[1:45], label="a'(a, 0.4) :Low L", legend=:bottomright )
plot!(a_grid[1:45], a_pol_H[1:45], label="a'(a, 1) :Low H", legend=:bottomright )
savefig("figs\\a_policy_aiyagari_new.png")


## Using Root finiding Algo from the Packges
function f!(F, θ)
    for i in 1:n
        F[i] = w_res_RL(θ, i)   
    end
    for i in 1:n
        F[n+i] = w_res_RH(θ, i)
    end
end

est_θ = nlsolve(f!,θ_inital).zero

##
for i in 1:2*n
    if est_θ[i] < 0.0
        est_θ[i] = 0.0
    end
end

## Let us try plotting policy function

a_pol_L = [a_p(a_grid[i], est_θ[1:n]) for i in 1:length(a_grid)]
a_pol_H = [a_p(a_grid[i], est_θ[n+1:2*n]) for i in 1:length(a_grid)]

plot(a_grid[1:10], a_pol_L[1:10], label="a'(a, 0.4) :Low L shock", legend=:bottomright )
plot!(a_grid[1:10], a_pol_H[1:10], label="a'(a, 1) :High H shock", legend=:bottomright )
savefig("figs\\a_policy_aiyagari.png")


plot(a_grid, a_pol_L, label="a'(a, 0.4) :Low L Shock", legend=:bottomright )
plot!(a_grid, a_pol_H, label="a'(a, 1) :Low H Shock", legend=:bottomright )
savefig("figs\\a_policy_aiyagari_new.png")





## Plots limited iterations
plot(a->a_p(a, theta_1k_iter[1:n]), xlim=(0.0, 1), label="a'(a, 0.4) :Low L", legend=:bottomright)
plot!(a->a_p(a, theta_1k_iter[n+1:2*n]), xlim=(0.0, 1), label = "a'(a, 1) : High L", legend=:bottomright)