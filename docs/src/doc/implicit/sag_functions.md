[⮜ Implicit Functions](./index.md)

# Sag Functions

A sag function models a surface as a deviation from a reference plane. In 2D
that plane is the meridional axis, in 3D it is the YZ plane.

A sag function takes transverse coordinates as input and returns the axial
deviation $g$ (or $G$ in 3D) along with its gradient.

**In 2D:**

$$
g: r \mapsto \bigl(g(r),\; g'(r)\bigr)
$$

**In 3D:**

$$
G: (y, z) \mapsto \bigl(G(y,z),\; \nabla G(y,z)\bigr)
$$

## Spherical

Parameterized by curvature $C$ (with radius of curvature $R = 1/C$).

**In 2D:**

$$\boxed{g(r) = \frac{C r^2}{1 + \sqrt{1 - r^2C^2}}}$$

Derivatives:

$$
\begin{align}
g'(r) &= \frac{C r}{\sqrt{s}}\\
g''(r) &= \frac{C(s + C^2r^2)}{s\sqrt{s}} \\
\end{align}
$$

With $s = 1 - r^2C^2$.

**In 3D** with $r^2 = y^2 + z^2$

$$\boxed{G(y, z) = \frac{C r^2}{1 + \sqrt{1 - r^2C^2}}}$$

Derivatives:

$$
\begin{align}
\nabla_y G(y,z) &= \frac{C y}{\sqrt{s}} \\
\nabla_z G(y,z) &= \frac{C z}{\sqrt{s}} \\
\\
\nabla_{yy} G(y,z) &= \frac{C(s+C^2y^2)}{s\sqrt{s}} \\
\nabla_{yz} G(y,z) &=  \frac{C^3yz}{s\sqrt{s}}\\
\nabla_{zz} G(y,z) &= \frac{C(s+C^2z^2)}{s\sqrt{s}}
\end{align}
$$

With $s = 1 - r^2C^2$.

## Parabolic

Parameterized by coefficient $A$.

**In 2D:**

$$
g(r) = A r^2
$$

$$
g'(r) = 2 A r
$$

**In 3D** with $r^2 = y^2 + z^2$:

$$
\begin{align}
G(y, z) &= A r^2 \\
\\
\nabla_y G(y,z) &= 2 A y \\
\nabla_z G(y,z) &= 2 A z \\
\\
\nabla_{yy} G(y,z) &= 2A \\
\nabla_{xy} G(y,z) &= 0 \\
\nabla_{zz} G(y,z) &= 2A \\
\end{align}
$$



## Conical

Generalization of the spherical surface, parameterized by curvature $C$ and
conic constant $K$.

| $K$ | Surface type |
|--|-|
| $K < -1$ | Hyperbola |
| $K = -1$ | Parabola |
| $-1 < K < 0$ | Prolate ellipse |
| $K = 0$ | Sphere |
| $K > 0$ | Oblate ellipse |

**In 2D:**

$$
g(r) = \frac{C r^2}{1 + \sqrt{1 - (1+K) C^2 r^2}}
$$

$$
g'(r) = \frac{C r}{\sqrt{1 - (1+K) C^2 r^2}}
$$

**In 3D** with $r^2 = y^2 + z^2$:

$$
\begin{align}
G(y, z) &= \frac{C r^2}{1 + \sqrt{1 - (1+K) C^2 r^2}} \\
\\
\nabla_y G(y,z) &= \frac{C y}{\sqrt{1 - (1+K) C^2 r^2}} \\
\nabla_z G(y,z) &= \frac{C z}{\sqrt{1 - (1+K) C^2 r^2}} \\
\\

\end{align}
$$



## Aspheric (Even Polynomial)

An even polynomial in $r$ starting at degree 4, parameterized by coefficients
$\alpha_0, \alpha_1, \dots, \alpha_{N-1}$.

**In 2D:**

$$
g(r) = \sum_{i=0}^{N-1} \alpha_i \, r^{4 + 2i}
$$

$$
g'(r) = \sum_{i=0}^{N-1} \alpha_i \,(4 + 2i)\, r^{3 + 2i}
$$

**In 3D** with $r^2 = y^2 + z^2$:

$$
G(y, z) = \sum_{i=0}^{N-1} \alpha_i \, r^{2(2+i)}
$$

$$
\begin{align}
\nabla_y G(y,z) &= y \sum_{i=0}^{N-1} \alpha_i\,(4+2i)\,r^{2(1+i)} \\
\nabla_z G(y,z) &= z \sum_{i=0}^{N-1} \alpha_i\,(4+2i)\,r^{2(1+i)}
\end{align}
$$



## XY Polynomial

A general bivariate polynomial in $y$ and $z$, parameterized by a coefficient
matrix $C_{p,q}$ of shape $P \times Q$.

**In 3D only:**

$$
G(y, z) = \sum_{p=0}^{P-1} \sum_{q=0}^{Q-1} C_{p,q}\, y^p z^q
$$

$$
\begin{align}
\nabla_y G(y,z) &= \sum_{p=1}^{P-1} \sum_{q=0}^{Q-1} p\, C_{p,q}\, y^{p-1} z^q \\
\nabla_z G(y,z) &= \sum_{p=0}^{P-1} \sum_{q=1}^{Q-1} q\, C_{p,q}\, y^p z^{q-1}
\end{align}
$$


## Sag Sum

A sum of multiple sag functions. Given sag functions $g_1, g_2, \dots, g_M$:

$$
g(r) = \sum_{k=1}^{M} g_k(r), \qquad g'(r) = \sum_{k=1}^{M} g_k'(r)
$$

And equivalently in 3D. This is used to compose surface models, for example a
conical base combined with aspheric correction terms.


## Sag Function Normalization

A sag curve is defined as all points $(x, r)$ such that:

$$x = g(r) \quad \text{where } r \in [-\tau, \tau]$$

where $g: [-\tau, \tau] \to \mathbb{R}$ is the sag function.

A sag curve is typically parametrized by a vector $\theta$, so it's more
complete to write:

$$x = g(r, \theta)$$

where $\theta$ defines the curve's shape.

**Normalizing** means evaluating $g$ at $\frac{r}{\tau}$ instead of $r$, scaling
the result by $\tau$:

$$x = \tau \cdot g\!\left(\frac{r}{\tau}, \theta\right)$$

The gradient then becomes:

$$\nabla_x = g'(\frac{r}{\tau}, \theta)$$

Importantly, this results in a different curve for the same $\theta$.

**Identification** is asking if there is a $\tilde{\theta}$ such that for all $r$:

$$\tau \cdot g\!\left(\frac{r}{\tau}, \theta\right) = g(r, \tilde{\theta})$$


## Lift to Implicit Function

Sag functions are converted to implicit functions via a *lift function*. Two
lift variants exist, `raw` and `euclid`, differing in how they handle the region
outside the lens domain $[-\tau, \tau]$.

Both take a normalization factor $\eta$ (typically either $1$ or $\tau$). The
sag function is always evaluated at normalized coordinates $r / \eta$.

### Raw (2D)

The implicit function is defined everywhere, with no special boundary treatment:

$$
F(x, r) = \eta \cdot g\!\left(\frac{r}{\eta}\right) - x
$$

$$
\nabla F(x, r) = \left(-1,\; g'\!\left(\frac{r}{\eta}\right)\right)
$$

### Euclid (2D)

Inside the domain $|r| \le \tau$, same as raw. Outside the domain $|r| > \tau$,
the implicit function becomes the Euclidean distance to the boundary point $A$:

$$
A = \Bigl(\eta \cdot g\!\left(\tfrac{\tau}{\eta}\right),\; \tau\Bigr)
$$

$$
F(x, r) = \begin{cases}
\eta \cdot g\!\left(\dfrac{r}{\eta}\right) - x & |r| \le \tau \\[6pt]
\left\|(x,\, |r|) - A\right\|_2 & |r| > \tau
\end{cases}
$$

### Raw (3D)

$$
F(x, y, z) = \eta \cdot G\!\left(\frac{y}{\eta}, \frac{z}{\eta}\right) - x
$$

$$
\nabla F(x, y, z) = \left(-1,\; \nabla_y G,\; \nabla_z G\right)
$$
