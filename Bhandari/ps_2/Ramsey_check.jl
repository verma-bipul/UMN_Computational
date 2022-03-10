# This module solves the Ramsey problem for the quasi linear case
# We'll try to replicate figures from quant econ https://python-advanced.quantecon.org/opt_tax_recur.html#top

σ = 2
γ = 2
β = 0.9

## Utility functions
u(x::Real) = x^(1-σ)/(1-σ)
v(x::Real) = x^(1+γ)/(1+γ)

U(c::Real, n::Real) = u(c) - v(n) # defining the utility function

## Derivates
Uc(c) = ForwardDiff.derivative(u, c)
Ucc(c) = ForwardDiff.derivative(Uc, c)

Un(n) = -ForwardDiff.derivative(v, n)
Unn(n) = ForwardDiff.derivative(Un, n)

##
g = [0.1, 0.1, 0.1, 0.1, 0.2, 0.1]
θ = ones(6)
s = [(1.0, g[i]) for i in 1:6]
Ps = [0 1 0 0 0 0; 0 0 1 0 0 0; 0 0 0 0.5 0.5 0; 0 0 0 0 0 1; 0 0 0 0 0 1; 0 0 0 0 0 1]

##

## Ramsey for a given Φ
include("newton_julia.jl")

function Ramsey_policy(Φ::Real, s0::Tuple)

    # for a given value of multiplier Φ and initial state s0, this function calculates Ramsey policies
    
    function res_tg1(s_tuple::Tuple)
        θ, g = s_tuple[1], s_tuple[2]
        n(c::Real) = (g + c)/θ
        res1(c::Real) = (1+Φ)*(Uc(c)*θ + Un(n(c))) + Φ*(c*Ucc(c)*θ + n(c)*Unn(n(c)))
        c = Root_Newton(c->res1(c), 0.5)
        return c , n(c)
    end

    c_tg1 = [res_tg1(s[i])[1] for i in 1:length(s)]  # note that s has been defined outside the function
    n_tg1 = [res_tg1(s[i])[2] for i in 1:length(s)] 

    function res_t0(s_tuple::Tuple)  #condition for t=0
        θ, g = s_tuple[1], s_tuple[2]
        n(c::Real) = (g + c)/θ
        b0(c::Real) = 1.0
        res1(c::Real) = (1+Φ)*(Uc(c)*θ + Un(n(c))) + Φ*(c*Ucc(c)*θ + n(c)*Unn(n(c))) - Φ*Ucc(c)*b0(c)*θ
        c = Root_Newton(c->res1(c), 0.5)
        return c , n(c)
    end

    c0, n0 = res_t0(s0)[1], res_t0(s0)[2]

    return c_tg1, n_tg1, c0, n0

end


## Updating the value of Φ:
c_tg1, n_tg1, c0, n0 = Ramsey_policy(0.06175628494006813, s[1])

sum_int = inv(1.0I - β*Ps)*(Uc.(c_tg1).*c_tg1 .+ Un.(n_tg1).*n_tg1)

ResImC = Uc(c0) - Uc(c0)*c0 - Un(n0)*n0 - β*sum([Ps[1,i]*sum_int[i] for i in 1:length(sum_int)])

b_tg1 = sum_int./Uc.(c_tg1)
b0 = 1.0

τ_tg1 = [1.0 + Un(n_tg1[i])/Uc(c_tg1[i]) for i in 1:length(s)]

## Trying the root finding method to get Φ

function search_Φ(Φ::Real, s0::Tuple)
    c_tg1, n_tg1, c0, n0 = Ramsey_policy(Φ, s0)
    
    i_s0 = findfirst(isequal(s0), s)

    sum_int = inv(1.0I - β*Ps)*(Uc.(c_tg1).*c_tg1 .+ Un.(n_tg1).*n_tg1) #this sum is independent of s0

    ResImC = Uc(c0) - Uc(c0)*c0 - Un(n0)*n0 - β*sum([Ps[i_s0,i]*sum_int[i] for i in 1:length(sum_int)])

    return ResImC

end

Φ_est = Root_Newton(x->search_Φ(x, s[1]), 0.1)  # 0.06175628494006813

τ0 = 0.0

## Lets plot
sHist_h = [1, 2, 3, 4, 6, 6, 6]
sHist_l = [1, 2, 3, 5, 6, 6, 6]

c_sim_h = [c_tg1[i] for i in sHist_h]
c_sim_l = [c_tg1[i] for i in sHist_l]

n_sim_h = [n_tg1[i] for i in sHist_h]
n_sim_l = [n_tg1[i] for i in sHist_l]

b_sim_h = [b_tg1[i] for i in sHist_h]
b_sim_l = [b_tg1[i] for i in sHist_l]

τ_sim_h = [τ_tg1[i] for i in sHist_h]
τ_sim_l = [τ_tg1[i] for i in sHist_l]

c_plot_l = vcat(c0, c_sim_l)
c_plot_h = vcat(c0, c_sim_h)

n_plot_l = vcat(n0, n_sim_l)
n_plot_h = vcat(n0, n_sim_h)

b_plot_l = vcat(b0, b_sim_l)
b_plot_h = vcat(b0, b_sim_h)

τ_plot_l = vcat(τ0, τ_sim_l)
τ_plot_h = vcat(τ0, τ_sim_h)

plot(c_plot_l, label="g low")
plot!(c_plot_h, label = "g high")
savefig("figs\\QE_C.png")

plot(n_plot_l, label="g low")
plot!(n_plot_h, label = "g high")
savefig("figs\\QE_N.png")

plot(b_plot_l, label="g low")
plot!(b_plot_h, label = "g high")
savefig("figs\\QE_B.png")

plot(τ_plot_l, label="g low")
plot!(τ_plot_h, label = "g high")
savefig("figs\\QE_tau.png")

## ALL THE PLOTS ARE REPLICATED EXACTLY


# Lets try to find the N period bind prices for this case:

q = zeros(length(s), length(s))
q[:,1] .= 1.0
[q[i, 2] = β*sum([Ps[i, j]*Uc(c_tg1[j]) for j in 1:length(s)])*(1/Uc(c_tg1[i])) for i in 1:length(s)]

for col in 3:length(s)
    for i in 1:length(s)
     [q[i, col] = β*sum([Ps[i, j]*Uc(c_tg1[j])*q[j,col-1] for j in 1:length(s)])*(1/Uc(c_tg1[i])) for i in 1:length(s)]
    end
end 

Z = sum_int./Uc.(c_tg1)

bN = inv(q)*Z