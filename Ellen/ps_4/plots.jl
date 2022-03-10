# This file contains codes for plotting all the results

## Plotting the basis function for the grid X = [0, 1, 2, 6]
inlcude("linear_basis.jl")

plot(x -> ψ([0, 1, 3, 6], x, 1), xlim=(0, 6), label = "phi-1(x)")
plot!(x -> ψ([0, 1, 3, 6], x, 2), xlim=(0, 6), label = "phi-2(x)")
plot!(x -> ψ([0, 1, 3, 6], x, 3), xlim=(0, 6), label = "phi-3(x)")
plot!(x -> ψ([0, 1, 3, 6], x, 4), xlim=(0, 6), label = "phi-4(x)")
savefig("figs\\linear_basisMM.png")



## Plots for basic growth model
include("FE_test.jl")

plot(k->(1-β*α)*A*(k)^α, xlim=(0.01,6), label ="Actual c(k)", legend=:bottomright) 
plot!(k->c_est(k), xlim =(0.01, 6), label = "Estimated c(k)", legend=:bottomright)
savefig("figs\\c_policy_5grid.png")