# This submodule simultes a finite state markov chain given transition matix P and initial state number

function markov_simulate(P, index_s0::Int, T::Int64)
    N = size(P)[1]
    condn_dist_vec = [Categorical(P[i, :]) for i in 1:N]  #Categorical fn calculates gives the dicrete pdf

    S = zeros(Int64, T)
    S[1] = index_s0

    for t in 2:T
        S[t] = rand(condn_dist_vec[S[t-1]])
    end     

    return S
end


# check markov markov_simulate
#G = [0.1 0.9; 0.8 0.2]

#markov_simulate(G, 1, 100)
