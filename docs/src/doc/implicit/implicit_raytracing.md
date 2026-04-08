[⮜ Implicit Functions](./index.md)

# Implicit Raytracing

## Variant: 'newton'

Given a signed scalar field $Q(t) = F(P+tV)$, find $t^\star$ such that:

$$\boxed{Q(t^\star) = 0}$$

This yields the standard Newton update step:

$$t \longleftarrow t - \frac{Q(t)}{Q'(t)}$$

With:

$$
\begin{align}
Q(t) &= F(P+tV) \\
Q'(t) &= V \cdot \nabla F(P + tV) \\
\end{align}
$$

Pros:
* Doesn't need second order derivatives

Cons:
* Doesn't converge when rays don't intersect the surface
* Doesn't compute ray surface minimum

## Variant: 'newton2'

Given:

* a **signed** scalar field $F$ equal to zero on the surface and non zero elsewhere
* a ray $P + tV$

Define $Q(t) = F(P + tV)$, and find $t^\star$ such that:

$$\boxed{t^\star = \text{argmin}_t{(Q(t))^2}}$$

The minimum of the objective is reached when the derivative is zero:

$$Q'(t) Q(t) = 0$$

Which yields the "second order" Newton update step:

$$t \longleftarrow t - \frac{Q'(t)Q(t)}{(Q'(t))^2 + Q''(t)Q(t)}$$

With:

$$
\begin{align}
Q(t) &= F(P+tV) \\
Q'(t) &= V \cdot \nabla F(P + tV) \\
Q''(t) &= (V \cdot \nabla \nabla F(P + tV)) V
\end{align}
$$

Pros:
* Converges to the intersection or the ray surface minimum

Cons:
* Requires second order derivatives

## Variant: 'newton_min'

Given:

* a **positive** scalar field $F$ that is zero on the surface, and strictly positive elsewhere. Typically this is a smooth squared distance like quantity.
* a ray $P + tV$

We define $\phi(t) = F(P + tV)$, and find $t^\star$ such that:

$$\boxed{t^\star = \text{argmin}_t{\phi(t)}}$$

This minimum of the objective is reached when the derivative is zero: $\phi'(t) = 0$,
which yields the Newton minimization step:

$$t \longleftarrow t - \frac{\phi'(t)}{\phi''(t)}$$

With:

$$
\begin{align}
\phi(t) &= F(P+tV) \\
\phi'(t) &= V \cdot \nabla F(P + tV) \\
\phi''(t) &= (V \cdot \nabla \nabla F(P + tV)) V
\end{align}
$$

Pros:
* Converges to the intersection or the ray surface minimum

Cons:
* Requires second order derivatives
