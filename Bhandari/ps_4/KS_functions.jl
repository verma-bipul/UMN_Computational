# This module contains the necessary functions to estimate the KS models

function get_grids(amin, amax, np, power)

    grid = LinRange(amin^(1/power), amax^(1/power), np)
    return grid.^power
end

function get_grids_MMV(amin, amax, np, power)
    J = np-1
    grid = [(j/J)^power*amax for j in 0:J]

    return grid
end



struct TransMat
    Ps::Matrix{Float64}       # 4x4 
    Pz::Matrix{Float64}      # 2x2 aggregate shock
    Peps_gg::Matrix{Float64} # 2x2 idiosyncratic shock conditional on good to good
    Peps_bb::Matrix{Float64} # 2x2 idiosyncratic shock conditional on bad to bad
    Peps_gb::Matrix{Float64} # 2x2 idiosyncratic shock conditional on good to bad
    Peps_bg::Matrix{Float64} # 2x2 idiosyncratic shock conditional on bad to good
end

function get_transition_matrix(ug, ub, zg_ave_dur, zb_ave_dur, ug_ave_dur, ub_ave_dur, puu_rel_gb2bb, puu_rel_bg2gg)

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
 # prob. of 1 to 1 cond. on g to g
 p11_gg= 1-p10_gg
 # prob. of 1 to 1 cond. on b to b
 p11_bb= 1-p10_bb
 # prob. of 1 to 1 cond. on g to b
 p11_gb= 1-p10_gb
 # prob. of 1 to 1 cond on b to g
 p11_bg= 1-p10_bg
    
 #   (g1)         (b1)        (g0)       (b0)
 Ps=[pgg*p11_gg pgb*p11_gb pgg*p10_gg pgb*p10_gb;
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

    transmat=TransMat(Ps, Pz, Peps_gg, Peps_bb, Peps_gb, Peps_bg)
    return transmat
end


## Functions to calculate the wage rate and interest rate

get_r(θ, δ, z, K, N) = @. θ*z*K^(θ-1)*N^(1-θ) - δ  
get_w(θ, z, K, N) = @. (1-θ)*z*K^(θ)*N^(-θ)
