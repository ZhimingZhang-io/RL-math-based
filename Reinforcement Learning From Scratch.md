[TOC]

# Reinforcement Learning From Scratch

> Fundamental idea: When model is unavailable, we can use data. When data is unavailable, we can use model. When both are unavailable, we can do nothing ! ! !
>

<img src="C:\Users\27430\AppData\Roaming\Typora\typora-user-images\image-20251013103858322.png" alt="Outline" style="zoom:50%;" />

## Chapter 1

Key elements of MDP (Markov decision process)

- Sets:

  - State: the set of states
  - Action: the set of actions which is associated for state
  - Reward: the set of rewards

- Probability distribution:

  - State transition probability: at state *s*, take action *a*, the probability to transit to state *s'* is $p(s'|s,a)$ 
  - Reward probability: at state *s*, taking action *a*, the probability to get reward r is $p(r|s,a)$

- Policy: at state *s*, the probability to choose action *a* is $\pi(a|s)$ 

- *Markov property*: memoryless property
  $$
  p(s_{t+1}|a_{t+1},s_{t},...,a_1,s_0)=p(s_{t+1}|a_{t+1},s_t)\\
  p(r_{t+1}|a_{t+1},s_{t},...,a_1,s_0)=p(r_{t+1}|a_{t+1},s_t)
  $$

## Chapter 2

**Bootstrapping**: The returns rely on each other. The following $v$ denotes the *return*.

<img src="C:\Users\27430\AppData\Roaming\Typora\typora-user-images\image-20251013162850687.png" alt="image-20251013162850687" style="zoom:50%;" />
$$
v_n = r_n + \gamma v_{n+1}
$$
***Bellman equation***:

Write the above in the following matrix-vector form:
$$
\begin{bmatrix}v_1\\v_2\\v_3\\v_4\end{bmatrix}=\begin{bmatrix}r_1\\r_2\\r_3\\r_4\end{bmatrix}+\begin{bmatrix}\gamma v_2\\\gamma v_3\\\gamma v_4\\\gamma v_1\end{bmatrix}=\begin{bmatrix}r_1\\r_2\\r_3\\r_4\end{bmatrix}+\gamma \begin{bmatrix}0&1&0&0\\0&0&1&0\\0&0&0&1\\1&0&0&0\end{bmatrix}\begin{bmatrix}v_1\\v_2\\v_3\\v_4\end{bmatrix}
$$
 which can be written as
$$
\mathbf{v}=\mathbf{r}+\gamma\mathbf{P}\mathbf{v}
$$
This is the Bellman equation (for the specific deterministic problem)



The expectation of $G_t$ (the discounted return of different trajectories) is defined as the *state-value function* or simply *state value*:
$$
v_{\pi}(s)=\mathbf{E}[G_t|S_t=s]
$$
Q: What is the relationship between return and state value?

A: The state value is the mean of all possible returns that can be obtained starting from a state.



**Use Bellman equation to calculate it**:
$$
\begin{aligned}v_{\pi}(s)&=\mathbf{E}[G_t|S_t=s]\\&=\mathbf{E}[R_{t+1}+\gamma G_{t+1}|S_t=s]\\&=\mathbf{E}[R_{t+1}|S_t=s]+\gamma\mathbf{E}[G_{t+1}|S_t=s]\\&=\sum_a\pi(a|s)\sum_rp(r|s,a)r +\gamma\sum_{s'}v_\pi(s')\sum_ap(s'|s,a)\pi(a|s)\end{aligned}
$$

Given a little bit transformation, it can be written as
$$
v_\pi(s)=\sum_a\pi(a|s)[\sum_rp(r|s,a)r+\gamma\sum_{s'}p(s'|s,a)v_\pi(s')]
$$
**Highlights**: symbols in this equation

- $v_\pi(s)$ and $v_\pi(s')$ are state values to be calculated. Bootstrapping!
- $\pi(a|s)$ is a given policy. Solving the equation is called policy evaluation
- $p(r|s,a)$ and $p(s'|s,a)$ represent the dynamic model

It is more clear and elegant if written in the vector form:
$$
\mathbf{v}_\pi=\mathbf{r}_\pi+\gamma\mathbf{P}_\pi\mathbf{v}_\pi
$$
For example:
$$
\begin{bmatrix}v_\pi(s_1)\\v_\pi(s_2)\\v_\pi(s_3)\\v_\pi(s_4)\end{bmatrix}=\begin{bmatrix}r_\pi(s_1)\\r_\pi(s_2)\\r_\pi(s_3)\\r_\pi(s_4)\end{bmatrix}+\gamma\begin{bmatrix}p_\pi(s_1|s_1)&p_\pi(s_2|s_1)&p_\pi(s_3|s_1)&p_\pi(s_4|s_1)\\p_\pi(s_1|s_2)&p_\pi(s_2|s_2)&p_\pi(s_3|s_2)&p_\pi(s_4|s_2)\\p_\pi(s_1|s_3)&p_\pi(s_2|s_3)&p_\pi(s_3|s_3)&p_\pi(s_4|s_3)\\p_\pi(s_1|s_4)&p_\pi(s_2|s_4)&p_\pi(s_3|s_4)&p_\pi(s_4|s_4)\end{bmatrix}\begin{bmatrix}v_\pi(s_1)\\v_\pi(s_2)\\v_\pi(s_3)\\v_\pi(s_4)\end{bmatrix}
$$
Which is quite ELEGANT ! ! !



**Action value**: the average return the agent can get starting from a state and taking an action.

Definition:
$$
q_\pi(s,a)=\mathbf{E}[G_t|S_t=s|A_t=a]
$$
Compared to the bellman equation, we can get
$$
v_\pi(s)=\sum_{a}\pi(a|s)q_\pi(s,a)
$$

## Chapter 3

- Core concepts: optimal state value and optimal policy
- A fundamental tool: the Bellman optimality equation (BOE) 

**Bellman optimality equation**:
$$
\begin{align}v_\pi(s)&=\max_{\pi}\sum_a\pi(a|s)[\sum_rp(r|s,a)r+\gamma\sum_{s'}p(s'|s,a)v_\pi(s')]\\&=\max_{\pi}\sum_a\pi(a|s)q(s,a)\end{align}
$$
Considering that $\sum_{a}\pi(a|s)=1$, we have
$$
\max_\pi\sum_a\pi(a|s)q(s,a)=\max_{a\in A(s)}q(s,a)
$$
where the optimality is achieved when
$$
\pi(a|s)=\begin{cases}1&a=a^{*}\\0&a\neq a^{*}\end{cases}
$$
where $a^{*}=arg\max_aq(s,a)$



Some concepts: 

- Fixed points: $x\in X$ is a fixed point of $f:X\to X$ is

$$
f(x)=x
$$

- Contraction mapping: $f$ is a contraction mapping if

$$
||f(x_1)-f(x_2)||\le\gamma||x_1-x_2||
$$

where $\gamma\in(0,1)$

> [!NOTE]
>
> **Theorem (Contraction Mapping Theorem)**:
>
> For any equation that has the form of $x=f(x)$, if $f$ is a contraction mapping, then
>
> - Existence: there exists a fixed point $x^{*}$ satisfying $f(x^{*})=x^{*}$.
> - Uniqueness: The fixed point $x^{*}$ is unique.
> - Algorithm: Consider a sequence ${x_k}$ where $x_{k+1}=f(x_k)$, then $x_k\to x^{*}$ as $k \to \infty$. Moreover, the convergence rate is exponentially fast.

It is proved that BOE
$$
v=f(v)=\max_{\pi}(r_\pi+\gamma P_\pi v)
$$
satisfies the above theorem, so there always exists a solution $v^{*}$ and the solution is unique. The solution could be solved iteratively by
$$
v_{k+1}=f(v_k)=\max_\pi(r_\pi+\gamma P_\pi v_k)
$$
where $v_0$ could be set randomly.

## Chapter 4

**Value iteration algorithm**:

The algorithm
$$
v_{k+1}=f(v_k)=\max_\pi(r_\pi+\gamma P_\pi v_k),\quad k=1,2,3 ...
$$
can be decomposed to two steps.

- Step 1: policy update (PU): This step is to solve

$$
\pi_{k+1}=arg\max_{\pi}(r_\pi+\gamma P_\pi v_k)
$$

​	where $v_k$ is given

- Step 2: value update (VU)

$$
v_{k+1}=r_{\pi_{k+1}}+\gamma P_{\pi_{k+1}}v_k
$$

**Procedure summary**:

$v_k(s) \to q_k(s,a)\to$ greedy policy $\pi_{k+1}(a|s) \to$ new value $v_{k+1} = \max_aq_k(s,a)$



**Policy iteration**:

Given a random initial policy $\pi_0$

- Step 1: policy evaluation (PE)

  This step is to calculate the state value of $\pi_k$:
  $$
  v_{\pi_k}=r_{\pi_k}+\gamma P_{\pi_k}v_{\pi_k}
  $$
  Note that $v_{\pi_k}$ is a state value function.

- Step 2: policy improvement (PI)
  $$
  \pi_{k+1}=arg\max_{\pi}(r_\pi+\gamma P_{\pi}v_{\pi_k})
  $$
  The maximization is component-wise.

When one iteration is done, the policy is changed as follows which could lead to the change of $r$ (immediate reward), so the next value iteration which is aimed to solve $v_{\pi_{k+1}}$ is different.



**The comparison between value and policy iteration**:

![image-20251014202650480](C:\Users\27430\AppData\Roaming\Typora\typora-user-images\image-20251014202650480.png)



**Truncated policy iteration** is an even algorithm between value and policy where the steps of BE iteration (find the value through iteration) is limited.  

## Chapter 5

**Monte Carlo Estimation**: a broad class of techniques that rely on repeated random sampling to love approximation problems.



**MC Exploring Starts**: 

![image-20251016095805757](C:\Users\27430\AppData\Roaming\Typora\typora-user-images\image-20251016095805757.png)

Efficiently use one episode to estimate as many $q_{\pi}$ as possible.



**MC $\epsilon$-greedy**:

Since MC Exploring Starts needs to fully cover every single starts, it becomes less elegant. So here comes MC $\epsilon$-greedy which can use only one episode within sufficient steps to explore all the state-action pairs.
$$
\pi(a|s)=\begin{cases}1-\frac{\epsilon}{|A(s)|}(A(s)-1),\quad a = a^*\\\frac{\epsilon}{|A(s)|},\quad a \ne a^*\end{cases}
$$
 This $\epsilon$-greedy can balance between exploitation and exploration.    

## Chapter 6

**Better way to calculate mean estimation**:
$$
\omega_{k+1}=\frac{1}{k}\sum_{i=1}^{k}x_{i},\quad k=1,2,...
$$
Then, $\omega_{k+1}$ can be expressed in terms of $\omega_K$ as
$$
\begin{align}\omega_{k+1}=\frac{1}{k}\sum_{i=1}^{k}x_{i}&=\frac{1}{k}(\sum_{i=1}^{k-1}x_{i}+x_k)\\&=\frac{1}{k}((k-1)\omega_k+x_k)=w_k-\frac{1}{k}(\omega_k-x_k)\end{align}
$$
Which is more capable in the real situation as this does not need full sampling to obtain the estimation

**Robbins-Monro Algorithm**:

The RM algorithm is to solve $g(\omega)=0$, it can be written as
$$
\omega_{k+1}=\omega_{k}-\alpha_{k}\tilde{g}(\omega_k,\eta_{k})
$$
where $\tilde{g}(\omega,\eta)=g(\omega)+\eta$

**Stochastic gradient descent**:
$$
\omega_{k+1}=\omega_{k}-\alpha_{k}\nabla_{\omega}f(\omega_{k},x_k)
$$

## Chapter 7

**TD (Temporal Difference) learning of state value**:
$$
v_{t+1}(s_t)=v_{t}(s_t)-\alpha_t(s_t)[v_t(s_t)-(r_{t+1}+\gamma v_t(s_{t+1})]
$$
Here,
$$
\bar{v_t}\dot{=}r_{r+1}+\gamma v(s+1)
$$
is called the TD target.
$$
\delta_t\dot{=}v_{t}(s_t)-(r_{t+1}+\gamma v_t(s_{t+1})
$$
is called the TD error.

*The TD error can be interpreted as innovation, which means new information obtained from the experience $(s_{t},r_{t+1},s_{t+1})$*



**TD learning of action values: Sarsa**:
$$
q_{t+1}(s_t,a_t)=q_{t}(s_t,a_t)-\alpha_t(s_t,a_t)[q_t(s_t,a_t)-(r_{t+1}+\gamma q_t(s_{t+1},a_{t+1})]
$$
When combined with policy improvement, Sarsa could solve the optimal problem.



**Q-learning**:
$$
q_{t+1}(s_t,a_t)=q_{t}(s_t,a_t)-\alpha_t(s_t,a_t)[q_t(s_t,a_t)-(r_{t+1}+\gamma \max_{a\in\Alpha}q_t(s_{t+1},a)]
$$
There exists two policies in TD learning task:

- The behavior policy is used to generate experience samples.
- The target policy is constantly updated toward an optimal policy.

On-policy vs off-policy

- When the behavior policy is the same as the target policy, such kind of learning is call on-policy
- When they are different, the learning is called off-policy.

> [!NOTE]
>
> Pseudocode： Optimal policy search by Q-learning (off-policy version)
>
> For each episode $\{s_0,a_0,r_1,s_1,a_1,r_1,...\}$ generated by $\pi_b$ do
>
> ​	For each step $t=0,1,2,...$ of the episode, do
>
> ​		Update q-value: 
> $$
> q_{t+1}(s_t,a_t)=q_{t}(s_t,a_t)-\alpha_t(s_t,a_t)[q_t(s_t,a_t)-(r_{t+1}+\gamma \max_{a\in\Alpha}q_t(s_{t+1},a)]
> $$
> ​		Update target policy:
>
> ​			$\pi_{T,t+1}(a|s_t)=1$ if $a=arg\max_aq_{t+1}(s_t,a)$

**A unified point of view**

All the algorithms above can be expressed in a unified expression:
$$
q_{t+1}(s_t,a_t)=q_t(s_t,a_t)-\alpha_t(s_t,a_t)[q_t(s_t,a_t)-\bar{q_t}]
$$
where $\bar{q_t}$ is the TD target.

![image-20251021221859953](C:\Users\27430\AppData\Roaming\Typora\typora-user-images\image-20251021221859953.png)

## Chapter 8

**Sarsa with function approximation**:
$$
w_{t+1}=w_t+\alpha_t[r_{t+1}+\gamma\hat{q}(s_{t+1},a_{t+1},w_t)-\hat{q}(s_t,a_t,w_t)]\nabla_{w}\hat{q}(s_t,a_t,w_t)
$$
**Q-learning with function approximation**:
$$
w_{t+1}=w_t+\alpha_t[r_{t+1}+\gamma\max_{a\in A}\hat{q}(s_{t+1},a_{t},w_t)-\hat{q}(s_t,a_t,w_t)]\nabla_{w}\hat{q}(s_t,a_t,w_t)
$$
**Deep Q-learning (DQN) **:

The objective function in the case is 
$$
J = E[(R+\gamma\max_{a\in A}\hat{q}(S',a,w_T)-\hat{q}(S,A,w))^2]
$$
where $w_T$ is the target network parameter.



When $w_T$ is fixed, the gradient of $J$ can be easily obtained as
$$
\nabla_wJ = E[(R+\gamma\max_{a\in A}\hat{q}(S',a,w_T)-\hat{q}(S,A,w))\nabla_w\hat{q}(S,A,w)]
$$

> [!NOTE]
>
> Pseudocode: Deep Q-learning (off-policy version)
>
> Aim: Learn an optimal target network to approximate the optimal action values from the experience samples generated by a behavior policy.
>
> Store the experience samples generated by $\pi_b$ in a replay buffer $B=\{(s,a,r,s')\}$
>
> ​	For each iteration, do
>
> ​		Uniformly draw a mini-batch of samples from B
>
> ​		For each sample $(s,a,rs')$, calculate the target value of 	$ y_T=r+\gamma\max_{a\in A}\hat{q}(s',a,w_{T})$, where $w_T$ is the parameter of the target network
>
> ​		Update the main network to minimize $(y_T-\hat{q}(s,a,w))^2$ using the mini-batch 
>
> ​		$\{(s,a,y_T)\}$
>
> ​	Set $w_T=w$ every C iteration

## Chapter 9

Policies can be represented by parameterized functions:
$$
\pi(a|s,\theta)
$$
where $\theta$ is a parameter vector.



gradient-based optimization algorithms to search for optimal policies:
$$
\theta_{t+1}=\theta_t+\alpha\nabla_{\theta}J(\theta_t)
$$
where $J$ can define optimal policies.



**A compact and useful form of the gradient**:
$$
\nabla_{\theta}J(\theta)=\sum_{s\in S}\eta(s)\sum_{s\in A}\nabla_{\theta}\pi(a|s,\theta)q_{\pi}(s,a)\\=E[\nabla_{\theta}ln\pi(A|S,\theta)q_{\pi}(S,A)]
$$

$$
\theta_{t+1}=\theta_t+\alpha(\frac{q_t(s_t,a_t)}{\pi(a_t|s_t,\theta_t)}\nabla_{\theta}\pi(a_t|s_t,\theta_t)
$$
The coefficient can well balance exploration and exploitation.

- If $q_t(s_t,a_t)$ is great, then the coefficient is great.
- Therefore, the algorithm intends to enhance actions actions with greater values.
- If $\pi(a_t|s_t,\theta)$ is small, then the coefficient is large.
- Therefore, the algorithm intends to explore actions that have low probabilities.

## Chapter 10

**actor-critic**:
$$
\theta_{t+1}=\theta_t+\alpha\nabla_{\theta}ln\pi(a_t|s_t,\theta_t)q_t(s_t,a_t)
$$

- This algorithm corresponds to actor
- The algorithm estimating $q_t(s,a)$ corresponds to critic

 

Property: the policy gradient is invariant to an additional baseline:
$$
\theta_{t+1}=\theta_t+\alpha\nabla_{\theta}ln\pi(a_t|s_t,\theta_t)(q_t(s_t,a_t)-v_{t}(s_t))
$$
Using this policy gradient, we can make stochastic samples more close to the ESTEMATION.



**Importance sampling**:

When we want to use off-policy, we can add the importance $\beta=\frac{p_0(x)}{p_1(s)}$ to fully use other data. 

> [!note]
>
> Deterministic actor-critic alorithm
>
> Initialization: A given behavior policy $\beta(a|s)$. A deterministic target policy $\mu(s,\theta_0)$ where $\theta_0$  is the initial parameter vector. A value function $v(s,w_0)$ where $w_0$ is the initial parameter vector
>
> Aim: Search for an optimal policy by maximizing $J(\theta)$
>
> At time step t in each episode, do
>
> ​	Generate $a_t$ following $\beta$ and then observe $r_{t+1},s_{t+1}$.
>
> ​	TD error:
>
> ​		$\delta_t=r_{t+1}+\gamma q(s_{t+1},u(s_{t+1},\theta_t),w_t)-q(s_t,a_t,w_t)$
>
> ​	Critic (value update):
>
> ​		$w_{t+1}=w_t+a_w\delta_t\nabla_w q(s_t,a_t,w_t)$
>
> ​	Actor (policy update):
>
> ​		$\theta_{t+1}=\theta_t+a_\theta\nabla_\theta\mu(s_t,\theta_t)(\nabla_aq(s_t,a,w_{t+1}))|_{a=\mu(s_t)}$ 







THE END ! ! !

Move on to the next step !

