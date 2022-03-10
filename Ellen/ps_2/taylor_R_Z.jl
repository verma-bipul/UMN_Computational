# Bipul Verma, Sep 22, 2021
# This module take a  function r(Z), bar_Z as input and returns
# the second order taylor expansion around bar_Z in the form hat_r(Z) = Z'MZ

## Loading Packages
using ForwardDiff

# Vector theoretically are column vectors and transpose is a row vecor
# Julia follows this convention, so we can directly use formula for 
# M as given in LS book.
# r(z) = x'Rx + u'Qu + 2x'Wu ::NOTE: Q and R are interchnaged in the Prof. Ellens notes compared to LS

## Defining the taylor expansion
function Taylor_rZ(r::Function, bar_X::Vector, bar_u::Vector)
    R = zeros(length(bar_X), length(bar_X))  # preallocation of matrices to be filled later
    Q = zeros(length(bar_u), length(bar_u))
    W = zeros(length(bar_X), length(bar_u))

    bar_Z = [bar_X; bar_u]  # stacked vector 

    index_one = findfirst(isequal(1), bar_Z)
    e = zeros(length(bar_Z))
    e[index_one] = 1.0

    Δr = ForwardDiff.gradient(r, bar_Z) # Getting the gradient and Hessian
    Hr = ForwardDiff.hessian(r, bar_Z)

    M = e*(r(bar_Z) - Δr'*bar_Z + 0.5*bar_Z'*Hr*bar_Z)*e' + 
         0.5*(Δr*e' -e*bar_Z'*Hr -Hr*bar_Z*e' +e*Δr') + 0.5*Hr  # this is the formula in LS book

    r_hat(Z::Vector) = Z'*M*Z  # Our quadratic approximation

    R = M[1:length(bar_X), 1:length(bar_X)]
    Q = M[length(bar_X)+1:end, length(bar_X)+1:end]
    W = M[1:length(bar_X), length(bar_X)+1:end]
    return r_hat, R, Q, W   # one can later access all these elements outside the function
end

