## This replicates the policy iteration in MarkovC_Simulation
function compute_Kp_L(K, s_i, B, Kmin, Kmax, Nbar, ug, ub, Ns)
    Kp, L=ifelse(s_i%Ns == 1,
    (exp(B[1]+B[2]*log(K)), Nbar*(1-ug)), # if good
    (exp(B[3]+B[4]*log(K)), Nbar*(1-ub))) # if bad
    Kp = clamp(Kp, Kmin, Kmax)
    return Kp, L
end



function compute_expectation_FOC(kp, Kp, s_i, hh, B, k_opt)
@unpack β, μ, δ, θ, Probs, zmat, Nmat, amat, Rmat, wmat, ϵmat, Kmat, a_grid, K_grid, Kmin, Kmax, Na, NK, Ns, N_bar, c_bind, ug, ub = hh

global expec = 0.0
for s_n_i = 1:Ns
zp, epsp = zmat[1,1, s_n_i], ϵmat[1, 1,  s_n_i]
Kpp, Lp = compute_Kp_L(Kp, s_n_i, B, Kmin, Kmax, N_bar, ug, ub, Ns)
rn = get_r(θ, δ, zp, Kp, Lp)
kpp = interpolate((a_grid, K_grid), k_opt[:, :, s_n_i], Gridded(Linear()))
cp = (rn+1)*kp + get_w(θ, zp, Kp, Lp)*(epsp*N_bar)-kpp(kp, Kp)
global expec = expec + Probs.Ps[s_i, s_n_i]*(cp)^(-μ)*(1+rn)
end 
return expec
end

function solve_ump(hh, k_opt, B ; max_iter::Integer=10000, tol::AbstractFloat=1e-8)
 @unpack β, μ, δ, θ, Probs, zmat, Nmat, amat, Rmat, wmat, ϵmat, Kmat, a_grid, K_grid, Kmin, Kmax, Na, NK, Ns, N_bar, c_bind, ug, ub = hh

 global counter = 0
k_opt_n = similar(k_opt)
while true
global counter += 1
for s_i = 1:Ns
z, eps = zmat[1,1, s_i], ϵmat[1,1,s_i]
for (K_i, K) = enumerate(K_grid)
Kp, L = compute_Kp_L(K, s_i, B, Kmin, Kmax, N_bar, ug, ub, Ns)
for (k_i, k) = enumerate(a_grid)
    wealth = (get_r(θ, δ, z, K, L)+1)*k+ get_w(θ, z, K, L)*(eps*N_bar)
    expec=compute_expectation_FOC(k_opt[k_i, K_i, s_i], Kp, s_i, hh, B, k_opt)
    cn = (β*expec)^(-1.0/μ)
    k_opt_n[k_i, K_i, s_i] = wealth-cn
end
end
end
k_opt_n .= clamp.(k_opt_n, a_grid[1], a_grid[end])
dif_k = maximum(abs, k_opt_n - k_opt)
if dif_k < tol
break
end
if counter >= max_iter
@warn "Euler method failed to converge with $counter iterations (dif = $dif_k)"
break
end
k_opt .= 0.7*k_opt_n .+ (0.3)*k_opt
end
return k_opt
end

k_opt = 0.9*hh.amat

@time ammv = solve_ump(hh, k_opt, b_MMV )


function plot_Fig2(hh, k_opt, K_eval_point)
    k_lim = range(0, stop=80, length=1000)
    itp_e = interpolate((hh.a_grid, hh.K_grid), k_opt[:, :, 1], Gridded(Linear()))
    itp_u = interpolate((hh.a_grid, hh.K_grid), k_opt[:, :, 3], Gridded(Linear()))
    
    kp_e(k) = itp_e(k, K_eval_point)
    kp_u(k) = itp_u(k, K_eval_point)
    
    p = plot(k_lim, kp_e.(k_lim), linestyle=:solid, lab="employed")
    plot!(p, k_lim, kp_u.(k_lim), linestyle=:solid, lab="unemployed")
    plot!(p, k_lim, k_lim, color=:black, linestyle=:dash, lab="45 degree", width=0.5)
    title!(p, "FIG2: Individual policy function \n at K=$K_eval_point when good state")
    return p
end

plot_Fig2(hh, ammv, 40)
plot_Fig2(hh, a_new_MMV, 40)
plot_Fig2(hh, a_new, 40)
savefig("figs\\fig2_KS_egm.png")

plot_Fig2(hh, kss.k_opt, 40)


## Since all the policy function seems indistinguisable, we can well work with endogenous grid