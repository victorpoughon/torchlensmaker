# Parametric Raytracing

This section describes the math behind the parametric raytracing algorithm in Torch Lens Maker.

A parametric surface is a function $S$ of two *parameters* $(u, v) \in \mathbb{R}^2$ :

$$
S: \mathbb{R}^2 \longrightarrow \mathbb{R}^3
$$

We define $Q: \mathbb{R}^3 \longrightarrow \mathbb{R}^3$ and $\theta = (t, u, v)$:

$$
Q(\theta) = P + t V - S(u, v)
$$

## First order

In the first order model, raytracing is finding the triplet of parameter values $\theta^{*} = (t, u, v)$ such that:

$$
Q(\theta) = 0
$$

Assuming that an initial guess $\theta_0$ is available, we can use multidimensional Newton's method. 

$$
\boxed{J(\theta) \times (\theta_{n+1} - \theta_{n}) = - Q(\theta_n)}
$$

where $J(\theta)$ is the jacobian of $Q(\theta)$:

$$
J(\theta) = \begin{bmatrix}
\nabla_t Q(\theta)_x & \nabla_u Q(\theta)_x & \nabla_v Q(\theta)_x \\
\nabla_t Q(\theta)_y & \nabla_u Q(\theta)_y & \nabla_v Q(\theta)_y \\
\nabla_t Q(\theta)_z & \nabla_u Q(\theta)_z & \nabla_v Q(\theta)_z
\end{bmatrix}
$$

which simplifies to:

$$
J(\theta) = \Bigr[
V \quad -\nabla_u S(u,v) \quad -\nabla_v S(u,v)
\Bigr]
$$

## Second order

In the second order model, raytracing is finding the triplet of parameter values $\theta^{*} = (t, u, v)$ that **minimizes** $|| Q(\theta) ||^2$:

$$
|| Q(\theta) ||^2 = || P + t V - S(u, v) ||^2
$$

We define $D: \mathbb{R}^3 \longrightarrow \mathbb{R}^+$:

$$
D(\theta) = || Q(\theta) ||^2 = Q(\theta) \cdot Q(\theta)
$$

To minimize $D(\theta)$ starting from an initial guess $\theta_0$ we are looking to solve $\nabla D(\theta) = 0$. The Newton update step is:

$$
\boxed{
H(\theta) \times (\theta_{n+1} - \theta_{n}) = - \nabla D(\theta)
}
$$

Where $H(\theta)$ is the Hessian matrix of $D$.


**Derivation of $H$:**

Starting from $\nabla D = 2 J^T Q$ and differentiating with respect to $\theta$ using the product rule:

$$
H(\theta) = 2 \bigl( J(\theta)^T J(\theta) + Q(\theta)^T \nabla^2 Q(\theta) \bigr)
$$

where $Q(\theta)^T \nabla^2 Q(\theta)$ denotes the contraction of $Q$ with the third-order tensor of second derivatives of $Q$.

Since $Q(\theta) = P + tV - S(u, v)$ is linear in $t$ and only non-linear through $S(u,v)$, the second derivatives of $Q$ are exactly the (negated) second derivatives of $S$. Denoting $S_{uu} = \frac{\partial^2 S}{\partial u^2}$, $S_{uv} = \frac{\partial^2 S}{\partial u \partial v}$, $S_{vv} = \frac{\partial^2 S}{\partial v^2}$ (each a vector in $\mathbb{R}^3$), the contraction is:

$$
Q^T \nabla^2 Q = -\begin{bmatrix}
0 & 0 & 0 \\
0 & Q \cdot S_{uu} & Q \cdot S_{uv} \\
0 & Q \cdot S_{uv} & Q \cdot S_{vv}
\end{bmatrix}
$$

Therefore the exact Hessian is:

$$
\boxed{H = 2 \; J^T J - 2\begin{bmatrix}
0 & 0 & 0 \\
0 & Q \cdot S_{uu} & Q \cdot S_{uv} \\
0 & Q \cdot S_{uv} & Q \cdot S_{vv}
\end{bmatrix}}
$$
