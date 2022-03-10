## This file contains parameter
using LinearAlgebra


β = 0.962   # discount factor
ψ = 1.25   # leisure pref parameter
θ = 0.35  # capital share
δ = 0.05  # depreciation rate
γ_z = 0.0172 # productivity growth 
γ_n = 0.01  # population growth


ρ_z =  0.75
ρ_c =  0.75 
ρ_h =  0.75
ρ_d =  0.75
ρ_p =  0.75
ρ_g =  0.75

ρ_P = Diagonal(repeat([0.75], 6))

σ_Q = Diagonal(repeat([0.01], 6))
σ_Q[1,1], σ_Q[6,6] =0.02, 0.02

g_share = 0.17 # share of govt expenditure of total output.

## To calculate the steady state we'll need the steady state values of taxes as well
τ_css = 0.065
τ_hss = 0.38
τ_dss = 0.133
τ_pss = 0.36
z_ss = 1.0

