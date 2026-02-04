# Wavelength to RGB

This is taken from https://stackoverflow.com/a/39446403/565840 and converted to pytorch.


```python
import torch
from torchlensmaker.core.interp1d import interp1d

from torchlensmaker.testing.basic_transform import basic_transform


CIE_X = torch.tensor([
    0.000160, 0.000662, 0.002362, 0.007242, 0.019110, 0.043400, 0.084736, 0.140638, 0.204492, 0.264737,
    0.314679, 0.357719, 0.383734, 0.386726, 0.370702, 0.342957, 0.302273, 0.254085, 0.195618, 0.132349,
    0.080507, 0.041072, 0.016172, 0.005132, 0.003816, 0.015444, 0.037465, 0.071358, 0.117749, 0.172953,
    0.236491, 0.304213, 0.376772, 0.451584, 0.529826, 0.616053, 0.705224, 0.793832, 0.878655, 0.951162,
    1.014160, 1.074300, 1.118520, 1.134300, 1.123990, 1.089100, 1.030480, 0.950740, 0.856297, 0.754930,
    0.647467, 0.535110, 0.431567, 0.343690, 0.268329, 0.204300, 0.152568, 0.112210, 0.081261, 0.057930,
    0.040851, 0.028623, 0.019941, 0.013842, 0.009577, 0.006605, 0.004553, 0.003145, 0.002175, 0.001506,
    0.001045, 0.000727, 0.000508, 0.000356, 0.000251, 0.000178, 0.000126, 0.000090, 0.000065, 0.000046,
    0.000033
])

CIE_Y = torch.tensor([
    0.000017, 0.000072, 0.000253, 0.000769, 0.002004, 0.004509, 0.008756, 0.014456, 0.021391, 0.029497,
    0.038676, 0.049602, 0.062077, 0.074704, 0.089456, 0.106256, 0.128201, 0.152761, 0.185190, 0.219940,
    0.253589, 0.297665, 0.339133, 0.395379, 0.460777, 0.531360, 0.606741, 0.685660, 0.761757, 0.823330,
    0.875211, 0.923810, 0.961988, 0.982200, 0.991761, 0.999110, 0.997340, 0.982380, 0.955552, 0.915175,
    0.868934, 0.825623, 0.777405, 0.720353, 0.658341, 0.593878, 0.527963, 0.461834, 0.398057, 0.339554,
    0.283493, 0.228254, 0.179828, 0.140211, 0.107633, 0.081187, 0.060281, 0.044096, 0.031800, 0.022602,
    0.015905, 0.011130, 0.007749, 0.005375, 0.003718, 0.002565, 0.001768, 0.001222, 0.000846, 0.000586,
    0.000407, 0.000284, 0.000199, 0.000140, 0.000098, 0.000070, 0.000050, 0.000036, 0.000025, 0.000018,
    0.000013
])

CIE_Z = torch.tensor([
    0.000705, 0.002928, 0.010482, 0.032344, 0.086011, 0.197120, 0.389366, 0.656760, 0.972542, 1.282500,
    1.553480, 1.798500, 1.967280, 2.027300, 1.994800, 1.900700, 1.745370, 1.554900, 1.317560, 1.030200,
    0.772125, 0.570060, 0.415254, 0.302356, 0.218502, 0.159249, 0.112044, 0.082248, 0.060709, 0.043050,
    0.030451, 0.020584, 0.013676, 0.007918, 0.003988, 0.001091, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000
])

MATRIX_SRGB_D65 = torch.tensor([
    [ 3.2404542, -1.5371385, -0.4985314],
    [-0.9692660,  1.8760108,  0.0415560],
    [ 0.0556434, -0.2040259,  1.0572252]
])

def gamma_correct_srgb(c):
    threshold = 0.0031308
    a = 0.055
    gamma = 1 / 2.4

    mask = c <= threshold
    result = torch.where(mask, 
                         12.92 * c, 
                         (1 + a) * torch.pow(c, gamma) - a)

    return result

def wavelength_to_rgb(v):
    "Convert a tensor of wavelengths of shape (N,) to a tensor of RGB colors of shape (N, 3)"

    assert(v.dim() == 1)

    LEN_MIN=380
    LEN_MAX=780
    LEN_STEP=5

    N = v.shape[0]
    rgb = torch.zeros((N, 3))

    visible_space = torch.linspace(LEN_MIN, LEN_MAX, CIE_X.shape[0])
    
    X = interp1d(X=visible_space, Y=CIE_X, newX=v)
    Y = interp1d(X=visible_space, Y=CIE_Y, newX=v)
    Z = interp1d(X=visible_space, Y=CIE_Z, newX=v)

    RGB = (MATRIX_SRGB_D65 @ torch.column_stack((X, Y, Z)).T).T

    RGB = gamma_correct_srgb(RGB)
    RGB = torch.clamp(RGB, 0., 1.)

    return RGB

```


```python
from IPython.display import display, HTML

for rgb in wavelength_to_rgb(torch.linspace(400, 700, 200)):
    r, g, b = map(int, (255*rgb).tolist())
    display(HTML(f'<div style="margin: 0; width: 300px; height: 2px; background: rgb({r} {g} {b})"></div>'))
```


<div style="margin: 0; width: 300px; height: 2px; background: rgb(33 0 85)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(40 0 99)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(46 0 112)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(51 0 123)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(57 0 137)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(64 0 151)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(69 0 164)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(75 0 178)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(81 0 192)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(86 0 206)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(91 0 218)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(96 0 231)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(100 0 244)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(105 0 255)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(108 0 255)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(111 0 255)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(113 0 255)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(116 0 255)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(117 0 255)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(118 0 255)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(119 0 255)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(119 0 255)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(119 0 255)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(119 0 255)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(118 0 255)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(116 0 255)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(114 0 255)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(111 0 255)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(107 0 255)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(103 0 255)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(99 0 255)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(92 0 255)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(84 0 255)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(75 0 255)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(64 0 255)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(49 0 255)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(26 0 255)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(0 0 255)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(0 0 255)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(0 7 255)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(0 43 255)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(0 63 255)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(0 78 255)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(0 89 255)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(0 102 255)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(0 113 255)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(0 123 255)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(0 132 255)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(0 141 255)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(0 149 255)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(0 156 255)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(0 162 248)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(0 169 238)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(0 174 227)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(0 180 218)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(0 186 209)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(0 192 198)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(0 197 189)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(0 201 180)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(0 206 171)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(0 210 161)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(0 215 153)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(0 219 144)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(0 224 134)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(0 229 126)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(0 233 117)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(0 238 107)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(0 242 97)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(0 246 88)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(0 251 77)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(0 255 65)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(0 255 52)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(0 255 33)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(0 255 0)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(0 255 0)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(0 255 0)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(0 255 0)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(0 255 0)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(0 255 0)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(0 255 0)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(0 255 0)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(0 255 0)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(0 255 0)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(0 255 0)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(0 255 0)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(0 255 0)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(0 255 0)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(0 255 0)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(0 255 0)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(0 255 0)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(0 255 0)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(0 255 0)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(0 255 0)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(0 255 0)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(0 255 0)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(0 255 0)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(0 255 0)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(23 255 0)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(80 255 0)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(109 255 0)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(132 255 0)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(151 255 0)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(168 255 0)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(183 255 0)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(197 255 0)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(211 255 0)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(223 255 0)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(235 255 0)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(246 255 0)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(255 255 0)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(255 255 0)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(255 255 0)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(255 251 0)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(255 247 0)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(255 241 0)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(255 236 0)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(255 230 0)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(255 225 0)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(255 219 0)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(255 212 0)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(255 206 0)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(255 200 0)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(255 193 0)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(255 186 0)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(255 179 0)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(255 172 0)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(255 164 0)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(255 157 0)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(255 149 0)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(255 140 0)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(255 132 0)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(255 123 0)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(255 113 0)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(255 103 0)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(255 93 0)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(255 81 0)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(255 68 0)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(255 54 0)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(255 36 0)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(255 0 0)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(255 0 0)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(255 0 0)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(255 0 0)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(255 0 0)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(255 0 0)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(255 0 0)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(255 0 0)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(255 0 0)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(255 0 0)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(255 0 0)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(255 0 0)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(255 0 0)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(255 0 0)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(255 0 0)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(255 0 0)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(255 0 0)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(255 0 0)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(255 0 0)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(255 0 0)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(255 0 0)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(255 0 0)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(255 0 0)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(247 0 0)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(239 0 0)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(232 0 0)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(224 0 0)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(217 0 0)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(210 0 0)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(202 0 0)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(194 0 0)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(187 0 0)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(180 0 0)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(173 0 0)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(166 0 0)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(160 0 0)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(153 0 0)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(146 0 0)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(140 0 0)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(134 0 0)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(128 0 0)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(122 0 0)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(117 0 0)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(111 0 0)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(106 0 0)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(101 0 0)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(96 0 0)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(91 0 0)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(87 0 0)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(82 0 0)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(77 0 0)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(74 0 0)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(70 0 0)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(66 0 0)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(62 0 0)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(59 0 0)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(56 0 0)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(52 0 0)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(50 0 0)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(47 0 0)"></div>



<div style="margin: 0; width: 300px; height: 2px; background: rgb(44 0 0)"></div>



```python
import torchlensmaker as tlm
import torch

def rgb_to_hex(r, g, b):
    return '#{:02x}{:02x}{:02x}'.format(int(r*255), int(g*255), int(b*255))

def demo_dispersion(incident, n1, n2, rgb_outcident):
    # colliding with the X=0 plane
    normals = torch.tensor([-1, 0, 0]).expand_as(incident)

    # compute reflection / refraction
    outcident, _ = tlm.refraction(incident, normals, n1=n1, n2=n2, critical_angle="reflect")
    
    # verity unit norm
    assert torch.all(torch.le(torch.abs(torch.linalg.norm(outcident, dim=1) - 1.0), 1e-5))

    surface = tlm.SquarePlane(10.)
    hom, _ = basic_transform(1.0, "origin", [0, 0, 0], [0, 0, 0])(surface)

    # rays to display vectors
    incident_display = torch.column_stack((torch.zeros_like(incident), -incident))
    outcident_display = torch.column_stack((torch.zeros_like(outcident), outcident))

    scene = tlm.new_scene("3D")
    scene["data"].append(tlm.render_surface(surface, hom, dim=3))
    
    scene["data"].extend(tlm.render_collisions(points=torch.zeros((1, 3)), normals=[normals[0, :]]))

    scene["data"].append(tlm.render_rays(
        incident_display[:, :3],
        incident_display[:, :3] + 50*incident_display[:, 3:6], 0))
    
    # TODO update to use ray color instead of default_color
    for rgb, ray in zip(rgb_outcident, outcident_display):
        r, g, b = rgb.tolist()
        ray_start = ray.unsqueeze(0)[:, :3]
        ray_end = ray_start + 50*ray.unsqueeze(0)[:, 3:6]
        scene["data"].append(tlm.render_rays(ray_start, ray_end, 0, default_color=rgb_to_hex(r, g, b)))
        

    scene["show_optical_axis"] = False
    scene["show_axes"] = False

    tlm.display_scene(scene)

# number of rays
N = 50

# white light with all wavelengths from 400 to 700
wavelengths = torch.linspace(400, 700, N)

# Cauchy's equation for "Dense flint glass SF10"
n_material = 1.7280 + 0.01342  / ((wavelengths/1000)**2)

# incident rays unit vectors
#alpha, beta = torch.deg2rad(torch.tensor([-20, 60]))
alpha, beta = torch.deg2rad(torch.tensor([-15, 65]))
incident = torch.tensor([torch.sin(beta) * torch.cos(alpha), torch.sin(beta) * torch.sin(alpha), torch.cos(beta)]).expand((N, -1))

# rgb color of outcident rays
rgb_outcident = wavelength_to_rgb(wavelengths)

demo_dispersion(incident, 1.0, n_material, rgb_outcident)
demo_dispersion(incident, n_material, 1.0, rgb_outcident)
```


<TLMViewer src="./demo_dispersion_files/demo_dispersion_0.json?url" />



<TLMViewer src="./demo_dispersion_files/demo_dispersion_1.json?url" />

