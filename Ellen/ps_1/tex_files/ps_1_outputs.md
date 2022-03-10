### Econ 8185 Homework 1  

Bipul Verma

Sept 2021

---

In this problem set we compute the equilibrium of the following model -

$$\max_{\hat{c}_t, h_t , \hat{k}_{t+1}} \hat{\beta}^t U(\hat{c}_t, h_t) $$

subject to 

$$\hat{c}_t + (1+\gamma_z)(1+\gamma_n)\hat{k}_{t+1} = \hat{F}(\hat{k}_t, h_t, z_t)$$

$$log(z_{t+1}) = \rho log(z_t) + \epsilon_{t+1}$$ 

----



## 1 (a) Iteration on Bellman Equation

**Parameters:  ψ = 0, δ = 1, $\gamma_n$ = 0.0, $\gamma_z$ = 0.0,**



<p float="left">
<img src="figs/value_fn_1_1.png" alt="drawing" width="420"/>  
<img src="figs/cap_policy_1_1.png" alt="drawing" width="420"/>
<img src="figs/lab_policy_1_1.png" alt="drawing" width="420"/>
</p>



**Parameters:  ψ = 0.65, $\gamma_n$ = 0.02, $\gamma_z$ = 0.02, $\delta = 0.05, \rho = 0.8, \theta = 0.34$**  

Value function is way too slow with 200*100 k,h grid. I did the iteration for 40 mins on intel-i3 but then gave up.
Will try this after I'm done with the other parts.

<p float="left">
<img src="figs/value_fn_1_f.png" alt="drawing" width="420"/>  
<img src="figs/cap_policy_1_f.png" alt="drawing" width="420"/>
<img src="figs/lab_policy_1_f.png" alt="drawing" width="420"/>
</p>
**Algorithm**:

1. Create a 2 dimensional grid of capital $(k)$ and labor supply $(h)$.
2. Use subroutine Tauchen.jl to get an evenly spaced grid of shocks $(z)$ along with the Transition Matrix $(P)$ for the AR(1) process for $log(z)$.
3. Create 2 dimensional empty grids for storing value functions and policy functions. 
4. Create a return function $G(k, k', h, z, V_n) = U(c, h) + \hat{\beta}E_z[V_n(k', z')]$ where $c = F(k, h , z) + (1-\delta)k - (1+\gamma_n)(1+\gamma_z)k'$. The expected value is calculated using the transition matrix obtained above.
5. Create a grid search loop to look for the maximum value and the maximisers, i.e for each $(k, z)$, $V_{n+1}(k, z) = \max_{k', h} G(k, k', h, z, V_n)$;   $k'(k, z) , h(k, z) = argmax_{k', h} G(k, k', h, z, V_n)$ 
6. Iterate on the loop until convergence: $||V_{n+1} - V_n|| < tolerance$. 


----

## 1 (b) LQ Approximation

For ease of notation define:

$U(c, h) = log(c) + \psi log(1-h) $

 $\hat{F}(k, h, z) = k^{\theta}(zh)^{1-\theta} + (1-\delta)k$ 

$c(k, k', h, z) =  \hat{F}(k, h, z) - (1+\gamma_n)(1+\gamma_z)k' $



**(ii)** Return function depends on $(k, k', h)$:

The model in this case is described as follows:

$\max_u\sum \beta^t r(X_t, u_t)$ subject to $X_{t+1} = A X_t + Bu_t + C\epsilon_{t+1}$ where 

$X_t = [1,  k_{t} , log(z_t)]'$ $u_t = [k_{t+1}, h_t]'$  and $r(X_t, u_t) = U(c(k_t, k_{t+1}, h_t, z_t), h_t)$. Here matrices $A, B, C$ according to state space representation are:

$A= \begin{bmatrix}
1 & 0 & 0\\ 0 & 0 & 0 \\ 0 & 0 & \rho \end{bmatrix}, 
B = \begin{bmatrix} 0 & 0\\ 1 & 0 \\ 0 & 0 \end{bmatrix},
C = \begin{bmatrix} 0 \\ 0 \\  1 \end{bmatrix} $



<p float="left">  
<img src="figs/cap_policy_1_b_1.png" alt="drawing" width="430"/>
<img src="figs/h_policy_1_b_1.png" alt="drawing" width="430"/>
</p>
**(iii)** Return function depends on $(k, k')$:

We use the first order MRS condition along with the resource constraint to solve for $h(k, k' z)$ as implicit solution of the following equation:

 $$\frac{U_h(c(k, k', h, z), h)}{U_c(c(k, k', h, z), h)} = -\hat{F}(k, h, z)$$ 

In the present case $X_t = \begin{bmatrix} 1 \\ k_t \\ log(z_t)\end{bmatrix}, u_t = \begin{bmatrix}k_{t+1} \end{bmatrix}$. The state space representation is as follows:

$\underbrace{\begin{bmatrix} 1 \\ k_{t+1} \\ log(z_{t+1})\end{bmatrix}}_{X_{t+1}} = \underbrace{\begin{bmatrix}
1 & 0 & 0\\ 0 & 0 & 0 \\ 0 & 0 & \rho \end{bmatrix}}_{A} \underbrace{\begin{bmatrix} 1 \\ k_t \\ log(z_t)\end{bmatrix}}_{X_t} + \underbrace{\begin{bmatrix} 0 \\ 1 \\ 0 \end{bmatrix}}_{B} \underbrace{\begin{bmatrix}k_{t+1} \end{bmatrix}}_{u_t} + \underbrace{\begin{bmatrix} 0 \\ 0 \\ 1 \end{bmatrix}}_{C} \begin{bmatrix} \epsilon_{t+1} \end{bmatrix}  $ 

The results from change in the return function is exactly the same as before. We report the linearized policy function below: 

$$k'(k, z) = 0.0483869  + (0.914605)*k  + (0.837721)*z$$

The policy function is increasing in both $(k, z).$ 


----

## 1 (c) Vaughan Method


<p float="left">  
<img src="figs/cap_policy_1_c_1.png" alt="drawing" width="430"/>
<img src="figs/h_policy_1_c_1.png" alt="drawing" width="430"/>
</p>




----

## 2 Properties of Solution 

- When $\psi = 0$ , $h(k, z)$ is a constant for all the cases. This is evident from the VFI, LQ as well as the Vaughan method for such parameter value.
- When $\delta \neq 1$, the steady state value of capital is lower. The value function is shifted downwards.



---

## 3 Modified Preferences



(b)

The preference are $U(c, h) = \frac{(c(1-h)^{\psi})^{1-\gamma}}{1-\gamma}$  with $\gamma = 5$

**Plots from VFI**



<p float="left">
<img src="figs/value_fn_3_b.png" alt="drawing" width="420"/>  
<img src="figs/cap_policy_3_b.png" alt="drawing" width="420"/>
<img src="figs/lab_policy_3_b.png" alt="drawing" width="420"/>
</p>
**Note:** *The above VFI graphs are from a limited number of iterations and grid points and **not** from entire convergence since convergence was taking too much time. But the shape of policy functions are indicative of the final results.*

**Plots from LQ and Vaughan**

<p float="left">  
<img src="figs/cap_policy_3_VLQ.png" alt="drawing" width="430"/>
<img src="figs/h_policy_3_VLQ.png" alt="drawing" width="430"/>
</p>
