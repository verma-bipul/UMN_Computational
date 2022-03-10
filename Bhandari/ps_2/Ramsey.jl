# Optimal Ramsey Policies

## Loading Packages    
using LinearAlgebra
using Plots
using Random
using Distributions
using Kronecker
using ForwardDiff
using Roots
using NLsolve
using BenchmarkTools

## Parameters 

β = 0.99
σ = 2
γ = 1


## Utility functions
u(x::Real) = x^(1-σ)/(1-σ)
v(x::Real) = x^(1+1/γ)/(1+1/γ)

U(c::Real, n::Real) = u(c) - v(n) # defining the utility function


## Getting all the differentials to be used later
#note that in the present case Ucn(.) = 0 as utilty is spearable in (c, n)

Uc(c) = ForwardDiff.derivative(u, c)
Ucc(c) = ForwardDiff.derivative(Uc, c)

Un(n) = -ForwardDiff.derivative(v, n)
Unn(n) = ForwardDiff.derivative(Un, n)




## Discrete markov g and theta
include("Tauchen.jl")  
log_g, Pg = Tauchen(log(0.15), 0.95, 1.2/15, 5)  # get grid on g and transn prob
log_θ, Pθ = Tauchen(log(1), 0.95, 2/400, 5)  # get grid on θ and transn prob

g, θ = exp.(collect(log_g)), exp.(collect(log_θ))

s = vec(collect(Iterators.product(θ, g)))  # matrix of s , use vec(s) to get a vector of states, vec is by columns

Ps = Pg⊗Pθ  # Kronecker product gets the transition matrix on state s 

## Ramsey for a given Φ
include("newton_julia.jl")

function Ramsey_policy(Φ::Real, s0::Tuple)

    i_s0 = findfirst(isequal(s0), s)

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
        b0(c::Real) = 4*θ*n(c)
        res1(c::Real) = (1+Φ)*(Uc(c)*θ + Un(n(c))) + Φ*(c*Ucc(c)*θ + n(c)*Unn(n(c))) - Φ*Ucc(c)*b0(c)*θ
        c = Root_Newton(c->res1(c), 0.5)
        return c , n(c)
    end

    c0, n0 = res_t0(s0)[1], res_t0(s0)[2]
    b0 = 4*s0[1]*n0

    # Evaluating the implementation constraint isnide the function
    sum_int = inv(1.0I - β*Ps)*(Uc.(c_tg1).*c_tg1 .+ Un.(n_tg1).*n_tg1)
    ResImC = Uc(c0)*b0 - Uc(c0)*c0 - Un(n0)*n0 - β*sum([Ps[i_s0,i]*sum_int[i] for i in 1:length(sum_int)])

    b_tg1 = sum_int./Uc.(c_tg1)
    

    τ_tg1 = [1.0 + Un(n_tg1[i])/(Uc(c_tg1[i])*s[i][1]) for i in 1:length(s)]

    println("The redidual of Implementation constraint is $ResImC")

    return c_tg1, n_tg1, b_tg1, τ_tg1, c0, n0, b0

end


## Updating the value of Φ:
c_tg1, n_tg1, b_tg1, τ_tg1, c0, n0, b0 = Ramsey_policy(0.05602874910330168, s[1])


## Trying the root finding method to get Φ

function search_Φ(Φ::Real, s0::Tuple)
    c_tg1, n_tg1, b_tg1, τ_tg1, c0, n0, b0 = Ramsey_policy(Φ, s0)
    
    i_s0 = findfirst(isequal(s0), s)

    sum_int = inv(1.0I - β*Ps)*(Uc.(c_tg1).*c_tg1 .+ Un.(n_tg1).*n_tg1) #this sum is independent of s0

    ResImC = Uc(c0)*b0 - Uc(c0)*c0 - Un(n0)*n0 - β*sum([Ps[i_s0,i]*sum_int[i] for i in 1:length(sum_int)])

    return ResImC

end

Φ_est = Root_Newton(x->search_Φ(x, s[1]), 0.05)  # 0.05602874910330168
# root finding gives the exact value of Φ such that the implementation constraint binds
# runtime 11.3 seconds

## ? does the value of multiplier Φ depend on the inital state s0: lets check 
# there are 25 values that the initial state can take which one to choose?
# policy with mean initial state
Root_Newton(x->search_Φ(x, s[13]), 0.1)  # 0.07154370941749508


# Does the policies c(s) n(s) for t ≥ 1 depend on s0  -- yes because they depend on Φ which depends on s0

## Running some more checks on the Ramsey 
#check if the govt constraint holds independently
function check_budget()
    res = zeros(size(s))

    for i in 1:length(s)
        res[i] = Uc(c_tg1[i])*c_tg1[i] + Un(n_tg1[i])*n_tg1[i] + β*sum([Ps[i, j]*Uc(c_tg1[j])*b_tg1[j] for j in 1:length(s)]) - Uc(c_tg1[i])*b_tg1[i]
    end
    
    return res  
end

check_budget()   # at each state the govt budget holds


#----------------- SIMULATING RAMSEY POLICY WITH STATE CONTINGENT DEBT ----------------------#
include("MarkovC_Simulation.jl")
state_tran = markov_simulate(Ps, 1, 1000)

c_sim = [c_tg1[i] for i in state_tran]
n_sim = [n_tg1[i] for i in state_tran]
b_sim = [b_tg1[i] for i in state_tran]
τ_sim = [round(τ_tg1[i], digits=5) for i in state_tran]

plot(c_sim, label="Simulated path C")
savefig("figs\\ramsey_C.png")
plot(n_sim, label="Simulated path N")
savefig("figs\\ramsey_N.png")
plot(b_sim, label="Simulated path b")
savefig("figs\\ramsey_B.png")
plot(τ_sim, ylim=(0.0, 0.4), label="Simulated path taxes")
savefig("figs\\ramsey_Tau.png")


# We'll plot all the things at one place together. For this we'll need a mtrix 
simulated_policy = reshape([c_sim; n_sim; b_sim; τ_sim], 1000, 4)

titles =["Simulated path C" "Simulated path N" "Simulated path b" "Simulated path taxes"]
plot(simulated_policy, layout = (2,2), title=titles, label=["c" "n" "b" "tax"] )

layout = grid(4, 1, heights=[0.1 ,0.4, 0.4, 0.1])

#----------PORTFOLIO OF RISK FREE BONDS-------------------------------#

## Getting bond prices
# we need q1, q2, ...., qn for each of the 25 states
# we will represent the prices in matrix, where each row will correspond to prices for that state
q = zeros(length(s), length(s))
q[:,1] .= 1.0
[q[i, 2] = β*sum([Ps[i, j]*Uc(c_tg1[j]) for j in 1:length(s)])*(1/Uc(c_tg1[i])) for i in 1:length(s)]

for col in 3:length(s)
    for i in 1:length(s)
     [q[i, col] = β*sum([Ps[i, j]*Uc(c_tg1[j])*q[j,col-1] for j in 1:length(s)])*(1/Uc(c_tg1[i])) for i in 1:length(s)]
    end
end 

sum_int = inv(1.0I - β*Ps)*(Uc.(c_tg1).*c_tg1 .+ Un.(n_tg1).*n_tg1)
Z = sum_int./Uc.(c_tg1)

bN = inv(q)*Z
bN_potfolio = bN./sum(bN)

# the answer seems to be weird and incorrect and i can't figure out why


#----------------------- WITHOUT STATE CONTINGENT DEBT --------------------------------------#


