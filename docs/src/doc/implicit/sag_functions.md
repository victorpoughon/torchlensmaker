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

$$\boxed{g(r) = A r^2}$$

Derivatives:

$$
\begin{align}
g'(r) &= 2 A r \\
g''(r) &= 2A \\
\end{align}
$$

**In 3D** with $r^2 = y^2 + z^2$:

$$\boxed{G(y, z) = A r^2}$$

$$
\begin{align}
\nabla_y G(y,z) &= 2 A y \\
\nabla_z G(y,z) &= 2 A z \\
\\
\nabla_{yy} G(y,z) &= 2A \\
\nabla_{yz} G(y,z) &= 0 \\
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

$$\boxed{g(r) = \frac{C r^2}{1 + \sqrt{1 - (1+K) C^2 r^2}}}$$

Derivatives:

$$
\begin{align}
g'(r) &= \frac{C r}{\sqrt{s}} \\
g''(r) &= \frac{C}{s\sqrt{s}} 
\end{align}
$$

With $s = 1 - (1+K) C^2 r^2$.

**In 3D** with $r^2 = y^2 + z^2$:

$$\boxed{G(y, z) = \frac{C r^2}{1 + \sqrt{1 - (1+K) C^2 r^2}}}$$

Derivatives:
$$
\begin{align}
\\
\nabla_y G(y,z) &= \frac{C y}{\sqrt{s}} \\
\nabla_z G(y,z) &= \frac{C z}{\sqrt{s}} \\
\\
\nabla_{yy} G(y,z) &= \frac{C(s+C^2y^2(1+K))}{s\sqrt{s}} \\
\nabla_{yz} G(y,z) &=  \frac{C^3yz(1+K)}{s\sqrt{s}}\\
\nabla_{zz} G(y,z) &= \frac{C(s+C^2z^2(1+K))}{s\sqrt{s}}
\end{align}
$$

With $s = 1 - (1+K) C^2 r^2$.



## Aspheric (Even Polynomial)

An even polynomial in $r$ starting at degree 4, parameterized by coefficients
$\alpha_0, \alpha_1, \dots, \alpha_{N-1}$.

**In 2D:**

$$\boxed{g(r) = \sum_{i=0}^{N-1} \alpha_i \, r^{4 + 2i}}$$

Derivatives:

$$
\begin{align}
g'(r) &= \sum_{i=0}^{N-1} \alpha_i \,(4 + 2i)\, r^{3 + 2i} \\
g''(r) &= \sum_{i=0}^{N-1} \alpha_i \,(4 + 2i)(3 + 2i)\, r^{2 + 2i}
\end{align}
$$

**In 3D** with $r^2 = y^2 + z^2$:

$$\boxed{G(y, z) = \sum_{i=0}^{N-1} \alpha_i \, r^{4+2i}}$$

Define:

$$
\begin{align}
T &= \sum_{i=0}^{N-1} \alpha_i\,(4+2i)\,(r^2)^{1+i} \\
T' &= \sum_{i=0}^{N-1} \alpha_i\,(4+2i)(1+i)\,(r^2)^{i}
\end{align}
$$

Derivatives:

$$
\begin{align}
\nabla_y G(y,z) &= y \cdot T \\
\nabla_z G(y,z) &= z \cdot T \\
\\
\nabla_{yy} G(y,z) &= T + 2y^2 T' \\
\nabla_{yz} G(y,z) &= 2yz\, T' \\
\nabla_{zz} G(y,z) &= T + 2z^2 T'
\end{align}
$$



## XY Polynomial

A general bivariate polynomial in $y$ and $z$, parameterized by a coefficient
matrix $C_{p,q}$ of shape $P \times Q$.

**In 3D only:**

$$\boxed{G(y, z) = \sum_{p=0}^{P-1} \sum_{q=0}^{Q-1} C_{p,q}\, y^p z^q}$$

Derivatives:

$$
\begin{align}
\nabla_y G(y,z) &= \sum_{p=1}^{P-1} \sum_{q=0}^{Q-1} p\, C_{p,q}\, y^{p-1} z^q \\
\nabla_z G(y,z) &= \sum_{p=0}^{P-1} \sum_{q=1}^{Q-1} q\, C_{p,q}\, y^p z^{q-1} \\
\\
\nabla_{yy} G(y,z) &= \sum_{p=2}^{P-1} \sum_{q=0}^{Q-1} p(p-1)\, C_{p,q}\, y^{p-2} z^q \\
\nabla_{yz} G(y,z) &= \sum_{p=1}^{P-1} \sum_{q=1}^{Q-1} p\,q\, C_{p,q}\, y^{p-1} z^{q-1} \\
\nabla_{zz} G(y,z) &= \sum_{p=0}^{P-1} \sum_{q=2}^{Q-1} q(q-1)\, C_{p,q}\, y^p z^{q-2}
\end{align}
$$


## Sag Sum

A sum of multiple sag functions. Given sag functions $g_1, g_2, \dots, g_M$:

$$
g(r) = \sum_{k=1}^{M} g_k(r)
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

$$
\nabla \nabla F(x, r) = \begin{pmatrix} 0 & 0 \\ 0 & \dfrac{1}{\eta}\, g''\!\left(\dfrac{r}{\eta}\right) \end{pmatrix}
$$

### Absolute (2D)

The sag function is assumed defined everywhere. The associated implicit function is defined as:

$$
F(x, r) = |\eta \cdot g(r / \eta) - x|
$$

The derivatives are:

$$
\begin{align}
\nabla_x F(x,r) &= -s \\
\nabla_r F(x,r) &= g'(r / \eta) \cdot s \\
\\
\nabla_{xx} F(x, r) &= 0 \\
\nabla_{xr} F(x, r) &= 0 \\
\nabla_{rr} F(x, r) &= \dfrac{1}{\eta}\, g''(r / \eta) \cdot s \\
\end{align}
$$

Where $s = \text{sgn}(\eta \cdot g(r / \eta) - x)$.

### Euclid (2D)

Inside the domain $|r| \le \tau$, same as raw. Outside the domain $|r| > \tau$,
the implicit function becomes the Euclidean distance to the boundary point $A$:

$$
A = \Bigl(\eta \cdot g\!\left(\tfrac{\tau}{\eta}\right),\; \tau\Bigr)
$$

Define $P = (x,\, |r|)$. Then:

$$
F(x, r) = \left\|P - A\right\|
$$

**Gradient**:

$$
\nabla F(x, r) = \frac{P - A}{\left\|P - A\right\|}
$$

**Hessian**:

$$
\nabla \nabla F(x, r) = \frac{F(x,r)^2\, I \;-\; (P - A)(P - A)^\top}{F(x, r)^3}
$$

### Raw (3D)

$$
F(x, y, z) = \eta \cdot G\!\left(\frac{y}{\eta}, \frac{z}{\eta}\right) - x
$$

$$
\nabla F(x, y, z) = \left(-1,\; \nabla_y G,\; \nabla_z G\right)
$$

$$
\nabla \nabla F(x, y, z) = \frac{1}{\eta}\begin{pmatrix} 0 & 0 & 0 \\ 0 & \nabla_{yy} G & \nabla_{yz} G \\ 0 & \nabla_{yz} G & \nabla_{zz} G \end{pmatrix}
$$

where $\nabla_{*} G$ are the sag gradient and hessian entries evaluated at $(y/\eta,\, z/\eta)$.
