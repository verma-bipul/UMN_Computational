# Date: Sep, 2021
# Bipul Verma 

# This program calculates the steady state of the model
# We'll later take the second order Taylor approxiamtion around the shocks

## Loading Programs
using NLsolve
using ForwardDiff
 
## Defining MRS, Euler, Budget
∂U_∂c(c, h) = ForwardDiff.derivative(c->U(c, h), c)
∂U_∂h(c, h) = ForwardDiff.derivative(h->U(c,h), h)
∂F_∂k(k, h) = ForwardDiff.derivative(k->F_hat(k, h, 1), k)
∂F_∂h(k ,h) = ForwardDiff.derivative(h->F_hat(k, h, 1), h)  
MRS(c, k, h) = ∂F_∂h(k ,h) + ∂U_∂h(c, h)/∂U_∂c(c, h)
Euler(k, h) = (1+γ_z) - β*∂F_∂k(k, h)
Resource(c, k, h) = c + (1+γ_z)*(1+γ_n)*k - F_hat(k, h, 1)

## Creating a system of non-linear equations
function f!(F, x)
    F[1] = MRS(x[1], x[2], x[3])  # x[1] is c, x[2] is k, x[3] is h
    F[2] = Euler(x[2], x[3])
    F[3] = Resource(x[1], x[2], x[3])
end

## Getting steady_state values
ss_vals = nlsolve(f!, [0.5, 1.0, 0.5]).zero  # The result is very sensetive to the initial value

## Collecting the SS values 
bar_c, bar_k, bar_h = ss_vals[1], ss_vals[2], ss_vals[3] 
                                                  


##----END NOTES ---------##
#This file is very robust and can be well replicated.