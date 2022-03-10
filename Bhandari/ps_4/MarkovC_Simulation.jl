# This submodule simultes a finite state markov chain given transition matix P and initial state number
using Distributions

"""
The `markov_simulate` fucntion simulates a markov chain given transition matrix P, for T periods   
 
 - index_s0 is an optional argument, which specifies the index of starting point.
"""
function markov_simulate(P, T::Int64; index_s0 = nothing)


    N = size(P)[1]
    condn_dist_vec = [Categorical(P[i, :]) for i in 1:N]  #Categorical fn calculates gives the dicrete pdf

    S = zeros(Int64, T)

    if index_s0 == nothing
        S[1] = rand(collect(1:N))
    else
        s[1] = index_s0
    end

    for t in 2:T
        S[t] = rand(condn_dist_vec[S[t-1]])
    end     

    return S
end


# check markov markov_simulate
#G = [0.1 0.9; 0.8 0.2]

#markov_simulate(G, 1, 100)
