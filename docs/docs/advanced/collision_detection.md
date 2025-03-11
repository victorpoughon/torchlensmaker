# Collision detection

A detailed description of how differentiable collision detection works in Torch
Lens Maker.

## Definitions

### Parametric ray

A ray is represented parametrically as the set of all points $P + tV$ with $t
\in \mathbb{R}$, where:

* $P$ is an origin point
* $V$ is a unit direction vector

Therefore a ray is a infinite line in space. The ray can be 2D or 3D depending
on the number of dimensions being simulated. The only change is to the dimension
of P and V.

Technically, the ray origin $P$ can be anywhere on the ray, and the $t$ value is
arbitrary. In practice $P$ is often the surface collision point that creates the
ray (in the case of a refraction or reflection), or the source position (if it
comes from a light source).

### Implicit surface

An implicit surface is defined with a function: $F(X) = 0$. $X$ is either
a 2D or 3D point depending on the number of dimensions being simulated. To
clarify we use lower case $f$ for 2D, and upper case $F$ for 3D:

* In 2D, $f(x,r) = 0$
* In 3D, $F(x,y,z) = 0$

::: warning Mixed 2D / 3D
Note how the 2D axis variables are $x$ and $r$, not $x$ and $y$. This is
intended. In Torch Lens Maker, 2D mode is not a flattened version of a 3D
simulation, but an abstract 2D space that represents an arbitrary meridional
plane (which is not necessarily the XY plane, nor any specific plane of the 3D world). In
rotationally symmetric systems, $r^2 = y^2 + z^2$. In non rotationally
symmetric systems, $r$ is undefined.
:::

For collision detection, a surface must also define the gradient of F:

* In 2D, $\nabla f(x,r) = (\frac{df}{dx}, \frac{df}{dr})$
* In 3D, $\nabla F(x,y,z) = (\frac{dF}{dx}, \frac{dF}{dy}, \frac{dF}{dz})$

$F$ and $\nabla F$ must be defined everywhere, not just on the surface or a
region around the surface.

### Sag functions

As long as the above conditions are met, the meaning of the values of $F$ can be
anything. It can be a proper sign distance function (SDF) like used in shader
graphics, or a "surface sag function" as often used in optics, which is just the
distance to the surface along the X axis. Given a sag function $x = g(y)$, the
associated implicit function is $f(x,r) = g(r) - x$.

This is only defined within the domain of the sag function, which is typically
the diameter of a lens. Outside of that domain, another distance function must
be used.

> TODO document the band distance outline function here.

## Newton's method

To compute the intersection of a ray with a surface, we are looking for the
unknown $t$ such that a point is on the surface and on the ray:

$$ F(P + tV) = 0 $$

We call the quantity on the left side $Q(t)$, and finding the intersections is
finding the roots of $Q$.

In reality there can be zero, one or more roots. In practice we are going to
solve it iteratively and either:

* converge to a single solution
* fail to converge (indicating no collision)

We can solve the intersection equation using Newton's method with the update
step:

$$ t_{n+1} = t_n - \frac{Q(t_n)}{Q'(t_n)} $$

Developing with the multivariate chain rule we get:

$$ t_{n+1} = t_n - \frac{F(P + t_nV)}{V \cdot \nabla F(P + t_n V)} $$

where "$\cdot$" in the denominator is the dot product.

If $Q'(t) = V \cdot \nabla F(P + t_n V)$ is zero the update step is undefined,
when the ray and the surface derivative are parallel, or when the norm of the
derivative is zero (which should never be the case).

In practice, even when there is a unique root, this method can oscillate
strongly instead of converging, in cases where there is a non differentiable
point in F (like in the band distance outline function). To mitigate this, a
damping parameter $\alpha \in [0, 1]$ can be added to reduce the step size:

$$ t_{n+1} = t_n - \alpha \frac{F(P + t_nV)}{V \cdot \nabla F(P + t_n V)} $$

> TODO document the max_delta version of Newton's method

## Gradient descent

A problem with the Newton's Raphson formulation above arises in cases where
there is no root. For collision detection, it is necessary to detect the absence
of collision to be able to know if rays should be refracted and sent to the next
optical element in the stack, or dropped as "non colliding" rays. On top of
collision points, we must also produce a mask indicating which ray do not
collide with the surface.

Thankfully, using the implicit surface function $F$ it is easy to check if the
resulting points after iteration are indeed on the surface. The mask computation
is therefore checking if $F(x,y,z) = 0$ for the points $P+tV$ with the final
values of $t$ after iteration.

However, a big problem is that the algorithm produces an undefined update step
at any minimum where $Q'(t) = 0$. When the gradient is zero, or close to zero,
the update step will be extremely large leading to poor behavior in those cases.

A better formulation instead, is to attempt to minimize the distance to the
root, which can be expressed by defining a new function $H$:

$$ H(t) = \frac{1}{2} Q(t)^2$$

Developing:

$$H(t) = \frac{1}{2} \Big( F(P+tV) \Big)^2$$

The new problem formulation is that we want to find the value of $t$ that
minimizes $H(t)$. We now have a minimization problem instead of a root finding
problem.

The straight forward gradient descent solution to that minization problem is to
take a step in the direction of the gradient of H, which is:

$$ \frac{dH(t)}{dt} = Q'(t) Q(t) = (V \cdot \nabla F(P + t_n V)) F(P+tV) $$

Leading to the update step:

$$ t_{n+1} = t_n - \beta (V \cdot \nabla F(P + t_n V)) F(P+tV) $$

where $\beta$ is a fixed step size. Any existing more advanced forms of gradient
descent can also be used here.

## Gauss-Newton

The formulation above is essentially a least square problem with a single
residual term. We can also use the Gauss-Newton method to solve it.

First, we assume that $Q$ can be linearly approximated by:

$$ Q(t + \delta) = Q(t) + \delta Q'(t) $$

Plugin this into the equation for $H$ we get

$$ H(t+\delta) = \frac{1}{2} ( Q(t)^2 + \delta^2Q'(t)^2 + 2 \delta Q(t) Q'(t) )
$$

For our update step, we want to find the value of $\delta$ that minimizes
$G(t+\delta)$, therefore setting the derivative with respect to $\delta$ equal
to zero yields:

$$ Q'(t)^2 \delta + Q(t)Q'(t) = 0 $$

Solving for $\delta$ shows that we end up back on the first Newton's method
formulation. However this formulation reveals an important assumption. A linear
approximation for $Q$ everywhere implies that $Q(t) = 0$ has a solution (the
intersection with the axis) and that $Q'(t)$ is never zero, otherwise $Q(t)$
would be constant everywhere. The linear assumption only holds in some small
region around the current estimate for $t$, and gets more inaccurate as the step
size $\delta$ increases.

## Levenberg–Marquardt

Starting from the previous equation for $\delta$, instead of simplifying we can
perform a trick and add a constant term $\lambda$ to introduce "damping". This
is known as the Levenberg–Marquardt algorithm:

$$ (Q'(t)^2 + \lambda) \delta + Q(t)Q'(t) = 0 $$

which yields the update step:

$$ t_{n+1} = t_n - \frac{Q(t)Q'(t)}{Q'(t)^2 + \lambda} = t_n - \frac{V \cdot
\nabla F(P + t_n V) F(P+tV)}{(V \cdot \nabla F(P + t_n V))^2 + \lambda} $$

When $\lambda$ is close to zero, this update step is close to the full Newton
update step. When $\lambda$ is large, the update is closer to a gradient descent
step, with step size close to $\frac{1}{\lambda}$. Various techniques exist for
selecting a value for $\lambda$ and even changing it during the iterations.

Having a stricly non zero minimum value on $\lambda$ also helps mitigate the
divide by zero issue.

## Beam search

> TODO document boundary radius domain initialization

Every optimization method above requires an initial guess for the iteration over
t values.

- initial guess strategies: t=0, t=collision with X/Y axis, t=collision with
  bounding sphere
- bounding sphere diameter = bound on step size

differentiable last step: can use different method than main steps

## Differentiable collision detection

> TODO Wang 2022

## Surface of revolution

### 2D shape definition

A 2D shape is defined in the $(x, r)$ plane with the implicit equation: $f(x,
r) = 0$.

X is the optical axis, and R is the meridional axis, aka the perpendiular to the
optical axis.

Additionally, for working with lenses, we always have $f(0, 0) = 0$ (the curve
crosses the origin) and $f'_r(0, r) = 0$ (the curve is vertical at the origin).

### Rotational symmetry

We want to create the corresponding 3D surface by rotation around the X axis.
This means that the R axis from before, is now really the axis of the meridional
plane. (A meridional plane is a plane that contains the optical axis X).

So $r$ is the distance from the X axis to any point on the surface. In 3D, we
have for any meridional plane $r = \sqrt{y^2 + z^2}$.

The definition of the 3D surface of revolution is that the intersection of every
meridional plane with it is the 2D surface:

$$ F(x,y,z) = f \left(x,  \sqrt{y^2 + z^2} \right) = 0 $$

Often the form $f(x, \sqrt{y^2 + z^2})$ can be simplified analytically to
provide an efficient implementation of $F$.

## Generic form of $\nabla F$ for surfaces of revolution

We have:

$$ F(x, y, z) = f \left(x,  \sqrt{y^2 + z^2} \right) $$

Therefore:

$$ F'_x(x, y, z) = f'_x(x, \sqrt{y^2 + z^2}) $$

$$ F'_y(x,y,z) =
\frac{y}{\sqrt{y^2 + z^2}} f_r' \left(x, \sqrt{y^2 + z^2} \right)
$$

$$
F'_z(x,y,z) = \frac{z}{\sqrt{y^2 + z^2}} f_r' \left(x, \sqrt{y^2 + z^2} \right)
$$

However, for some curves $f$ this expression simplifies a lot and therefore
shapes can provide an optimized version of $\nabla F(x, y, z)$, or even $\nabla
F(x,y,z) \cdot V$.



## Sag surface and diameter band surface

In optics, it is convenient to define a surface with what is sometimes called a
*sag function*. This is a function that defines the offset from the meridional
axis, as a function of the radial coordinate: $x = g(r)$.

This suffers a problem however, when using a function that's undefined outside
of a fixed domain, typically the diameter of the surface (or the diameter of the
supporting surface in the case of a half-sphere). For example, when trying to
define a half circle of diameter $D$ with a sag function like this:

> TODO

the resulting implicit function $f(x,r) = g(r) - x$ that can be defined by
moving all terms to one side of the equation is undefined outside of the domain
given by the diameter. This makes collision detection impossible outside of the
diameter domain.

A related problem happens even with surfaces where the sag function is well
defined beyond the domain of the diameter. For example, consider the parabola
function $x = a y^2$. This can be used outside of the diameter domain to define
an implicit form $f(x,r) = a y^2 - x$. However, when performing collision
detection using this implicit form, there will exist a region outside the
surface domain where $f(x, r) = 0$. This causes trouble during optimization
because it creates undesirable local minimums.

To address both theses issues, a partial implicit surface is defined, called a
**diameter band surface**. This is essentially the function:

$$ f(P) = || P - A ||^2 $$

or in meridional coordinates form:

$$ f(x,r) = (x-A_x)^2 + (r - A_r)^2 $$

where $A$ is the *extent point*, i.e the extreme point on the surface.

To simplify computation to only the positive half plane, it's useful to
introduce an absolute value:

$$ f(x,r) = (x-A_x)^2 + (|r| - A_r)^2 $$

The derivatives are:

$$ \nabla_x f(x,r) = 2(X - A_x) $$

$$ \nabla_r f(x,r) = 2 (|r| - A_r) \, \text{sgn}(r) $$

This can be generalized to 3D and yield:

$$ F(x,y,z) = (x - A_x)^2 + (\sqrt{y^2 + z^2} - A_r)^2 $$

with derivatives:

$$ F'_x(x, y, z) = 2(x - A_x) $$

$$ F'_y(x,y,z) = 2y - \frac{A_r}{\sqrt{y^2 + z^2}} $$

$$ F'_z(x,y,z) = 2z - \frac{A_r}{\sqrt{y^2 + z^2}} $$

The gradient is undefined at $Y = Z = 0$, but that's typically not a problem
because we're interested in that function outside of the diameter region.

This surface can be combined with any other sag function to create an implicit
surface that's well defined everywhere and behaves nicely for optimization, even
when starting from outside of the diameter.

The implicit form of a surface defined by a sag function is easily derived:

$$ f(x,r) = g(r) - x $$

$$ \nabla_x f(x,r) = -1 $$

$$ \nabla_r f(x,r) = g'(r) $$

And in 3D, applying rotational symmetry around the X axis:

$$ F(x,y,z) = f(x, \sqrt{y^2 + z^2}) = g(\sqrt{y^2 + z^2}) - x $$

$$ \nabla_x F(x,y,z) = -1 $$

$$ \nabla_y F(x,y,z) = \frac{y}{\sqrt{y^2 + z^2}} g'(\sqrt{y^2 + z^2}) $$

$$ \nabla_z F(x,y,z) = \frac{z}{\sqrt{y^2 + z^2}} g'(\sqrt{y^2 + z^2}) $$

However it's typical for the expressions for $F$ and its derivatives to simplify
further than this, therefore it's best if implementations of sag functions
actually provide all of $g(r)$, $g'(r)$, $G(y,z)$ and $\nabla G(y,z)$, where
$G(y,z) = g(\sqrt{y^2 + z^2})$.

## Collision detection with a 3D transform

Surfaces are defined on a local reference frame so that $F(0, 0, 0) = 0$. But
what if we want to apply a transform to move it in 3D space? Can we apply
scaling, rotation, translation?

Let's assume our 3D transform $T$ is affine invertible and produces points $X'$
given input points $X$ such that $T: X' = AX + B$.

Let's consider some points $X'$ on the new transformed surface, by definition
undoing the transform would put them back on the original surface:

$$ F(T^{-1}(X')) = F(A^{-1}(X' - B)) = 0 $$

Given a parametric 3D ray: $P + tV$, finding the intersection with a transformed
3D surface is therefore solving:

$$ F( A^{-1}(P-B) + tA^{-1}V ) = 0 $$

which is useful because we can use any local solver by
applying the inverse transform to the rays, and using the $F$ function defined
locally:

$$ \begin{cases} P' = A^{-1}(P - B)\\
V' = A^{-1}V \end{cases} $$

In the common case, $A^{-1}$ can be computed without matrix inversion because
it's the product of a rotation and a scaling, each can be easily inverted.

Note that this applies even if the surface is not defined implicitly: we can
find collisions with the transformed surface by applying the above inverse
transform to the rays and calling the local collision detection code.

Another thing we need to do is convert vectors from the surface local frame, to
the global frame, typically surface normals.

A vector $\overrightarrow{N}$ is the difference between its end point $E$ and
start point $S$:

$$ \overrightarrow{N} = E - S $$

So to transform the vector under the affine transformation, we can take the
difference of its transformed endpoints:

$$ T(\overrightarrow{N}) = T(E) - T(S)$$

So after simplifying we get:

$$ T(\overrightarrow{N}) = A(E - S) = A\overrightarrow{N} $$

## Adding anchors

Similarly as above, it can be useful to add a translation step before the
rotation, to model an "anchor". The anchor is the point on the shape that
attaches to the global frame. So, our full transform is now four steps:

1. A translation $-A$ to account for the anchor
2. A scale $S$
3. A rotation $R$
4. A translation $T$ to position the shape in the global frame

$$ X' = RS(X - A) + T $$

The inverse transform is:

$$ X = S^{-1}R^{-1}(X' - T) + A $$

When $X'$ (the points on the transformed surface) and also the collision point
with parametric rays $P + tV$ we can substitue and get:

$$ S^{-1} R^{-1} (P-T) + A + t S^{-1}R^{-1}V $$

And so we can compute "inverse transformed" rays:

$$ \begin{cases} P' = S^{-1} R^{-1}(P-T) + A\\
V' = S^{-1}R^{-1}V \end{cases} $$

Direct transform of vectors is:

$$ T(\overrightarrow{N}) = RS\overrightarrow{N}$$

## Forward kinematic chain

> TODO document using sucessive transforms to make a forward kinematic chain

