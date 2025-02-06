# Wavelength to RGB

This is taken from https://stackoverflow.com/a/39446403/565840 and converted to pytorch.


```python
import torch
from torchlensmaker.interp1d import interp1d

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

    surface = tlm.surfaces.SquarePlane(10.)
    transform = basic_transform(1.0, "origin", [0, 0, 0], [0, 0, 0])(surface)

    # rays to display vectors
    incident_display = torch.column_stack((torch.zeros_like(incident), -incident))
    outcident_display = torch.column_stack((torch.zeros_like(outcident), outcident))

    scene = tlm.viewer.new_scene("3D")
    scene["data"].append(tlm.viewer.render_surfaces([surface], [transform], dim=3))
    
    scene["data"].extend(tlm.viewer.render_collisions(points=torch.zeros((1, 3)), normals=[normals[0, :]]))

    scene["data"].append(tlm.viewer.render_rays(
        incident_display[:, :3],
        incident_display[:, :3] + 50*incident_display[:, 3:6]),)
    
    # TODO update to use ray color instead of default_color
    for rgb, ray in zip(rgb_outcident, outcident_display):
        r, g, b = rgb.tolist()
        ray_start = ray.unsqueeze(0)[:, :3]
        ray_end = ray_start + 50*ray.unsqueeze(0)[:, 3:6]
        scene["data"].append(tlm.viewer.render_rays(ray_start, ray_end, default_color=rgb_to_hex(r, g, b)))
        

    scene["show_optical_axis"] = False
    scene["show_axes"] = False

    tlm.viewer.ipython_display(scene)

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


<div data-jp-suppress-context-menu id='tlmviewer-d387c75d' class='tlmviewer' style='width: 100%; aspect-ratio: 16 / 9;'></div><script type='module'>async function importtlm() {
    try {
        return await import("/tlmviewer.js");
    } catch (error) {
        console.log("error", error);
        return await import("/files/test_notebooks/tlmviewer.js");
    }
}

const module = await importtlm();
const tlmviewer = module.tlmviewer;

const data = '{"mode": "3D", "camera": "orthographic", "data": [{"type": "surfaces", "data": [{"matrix": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], "samples": [[0.0, 0.0], [0.0, 0.0714249238371849], [0.0, 0.1428498476743698], [0.0, 0.21427476406097412], [0.0, 0.2856996953487396], [0.0, 0.3571246266365051], [0.0, 0.42854952812194824], [0.0, 0.49997445940971375], [0.0, 0.5713993906974792], [0.0, 0.6428242921829224], [0.0, 0.7142492532730103], [0.0, 0.7856741547584534], [0.0, 0.8570990562438965], [0.0, 0.9285240173339844], [0.0, 0.9999489188194275], [0.0, 1.0713738203048706], [0.0, 1.1427987813949585], [0.0, 1.2142237424850464], [0.0, 1.2856485843658447], [0.0, 1.3570735454559326], [0.0, 1.4284985065460205], [0.0, 1.4999233484268188], [0.0, 1.5713483095169067], [0.0, 1.6427732706069946], [0.0, 1.714198112487793], [0.0, 1.7856230735778809], [0.0, 1.8570480346679688], [0.0, 1.9284729957580566], [0.0, 1.999897837638855], [0.0, 2.0713226795196533], [0.0, 2.142747640609741], [0.0, 2.214172601699829], [0.0, 2.285597562789917], [0.0, 2.357022523880005], [0.0, 2.4284474849700928], [0.0, 2.4998724460601807], [0.0, 2.5712971687316895], [0.0, 2.6427221298217773], [0.0, 2.7141470909118652], [0.0, 2.785572052001953], [0.0, 2.856997013092041], [0.0, 2.928421974182129], [0.0, 2.9998466968536377], [0.0, 3.0712716579437256], [0.0, 3.1426966190338135], [0.0, 3.2141215801239014], [0.0, 3.2855465412139893], [0.0, 3.356971502304077], [0.0, 3.428396224975586], [0.0, 3.499821186065674], [0.0, 3.57124662399292], [0.0, 3.642671585083008], [0.0, 3.7140963077545166], [0.0, 3.7855212688446045], [0.0, 3.8569462299346924], [0.0, 3.9283711910247803], [0.0, 3.999796152114868], [0.0, 4.071220874786377], [0.0, 4.142645835876465], [0.0, 4.214070796966553], [0.0, 4.285495758056641], [0.0, 4.3569207191467285], [0.0, 4.428345680236816], [0.0, 4.499770641326904], [0.0, 4.571195602416992], [0.0, 4.64262056350708], [0.0, 4.714045524597168], [0.0, 4.785470008850098], [0.0, 4.8568949699401855], [0.0, 4.928319931030273], [0.0, 4.999744892120361], [0.0, 5.071169853210449], [0.0, 5.142594814300537], [0.0, 5.214019775390625], [0.0, 5.285444736480713], [0.0, 5.356869697570801], [0.0, 5.428294658660889], [0.0, 5.499719619750977], [0.0, 5.5711445808410645], [0.0, 5.642569541931152], [0.0, 5.713994026184082], [0.0, 5.78541898727417], [0.0, 5.856843948364258], [0.0, 5.928268909454346], [0.0, 5.999693870544434], [0.0, 6.0711188316345215], [0.0, 6.142543792724609], [0.0, 6.213968753814697], [0.0, 6.285393714904785], [0.0, 6.356818675994873], [0.0, 6.428243637084961], [0.0, 6.499668598175049], [0.0, 6.571093559265137], [0.0, 6.642518043518066], [0.0, 6.713943004608154], [0.0, 6.785367965698242], [0.0, 6.85679292678833], [0.0, 6.928217887878418], [0.0, 6.999642848968506], [0.0, 7.071067810058594]], "clip_planes": [[0.0, -1.0, 0.0, 5.0], [0.0, 1.0, 0.0, 5.0], [0.0, 0.0, -1.0, 5.0], [0.0, 0.0, 1.0, 5.0]]}]}, {"type": "points", "data": [[0.0, 0.0, 0.0]], "color": "#ff0000"}, {"type": "arrows", "data": [[-1, 0, 0, 0.0, 0.0, 0.0, 1.0]]}, {"type": "rays", "points": [[0.0, 0.0, 0.0, -43.771305084228516, 11.728486061096191, -21.13091278076172], [0.0, 0.0, 0.0, -43.771305084228516, 11.728486061096191, -21.13091278076172], [0.0, 0.0, 0.0, -43.771305084228516, 11.728486061096191, -21.13091278076172], [0.0, 0.0, 0.0, -43.771305084228516, 11.728486061096191, -21.13091278076172], [0.0, 0.0, 0.0, -43.771305084228516, 11.728486061096191, -21.13091278076172], [0.0, 0.0, 0.0, -43.771305084228516, 11.728486061096191, -21.13091278076172], [0.0, 0.0, 0.0, -43.771305084228516, 11.728486061096191, -21.13091278076172], [0.0, 0.0, 0.0, -43.771305084228516, 11.728486061096191, -21.13091278076172], [0.0, 0.0, 0.0, -43.771305084228516, 11.728486061096191, -21.13091278076172], [0.0, 0.0, 0.0, -43.771305084228516, 11.728486061096191, -21.13091278076172], [0.0, 0.0, 0.0, -43.771305084228516, 11.728486061096191, -21.13091278076172], [0.0, 0.0, 0.0, -43.771305084228516, 11.728486061096191, -21.13091278076172], [0.0, 0.0, 0.0, -43.771305084228516, 11.728486061096191, -21.13091278076172], [0.0, 0.0, 0.0, -43.771305084228516, 11.728486061096191, -21.13091278076172], [0.0, 0.0, 0.0, -43.771305084228516, 11.728486061096191, -21.13091278076172], [0.0, 0.0, 0.0, -43.771305084228516, 11.728486061096191, -21.13091278076172], [0.0, 0.0, 0.0, -43.771305084228516, 11.728486061096191, -21.13091278076172], [0.0, 0.0, 0.0, -43.771305084228516, 11.728486061096191, -21.13091278076172], [0.0, 0.0, 0.0, -43.771305084228516, 11.728486061096191, -21.13091278076172], [0.0, 0.0, 0.0, -43.771305084228516, 11.728486061096191, -21.13091278076172], [0.0, 0.0, 0.0, -43.771305084228516, 11.728486061096191, -21.13091278076172], [0.0, 0.0, 0.0, -43.771305084228516, 11.728486061096191, -21.13091278076172], [0.0, 0.0, 0.0, -43.771305084228516, 11.728486061096191, -21.13091278076172], [0.0, 0.0, 0.0, -43.771305084228516, 11.728486061096191, -21.13091278076172], [0.0, 0.0, 0.0, -43.771305084228516, 11.728486061096191, -21.13091278076172], [0.0, 0.0, 0.0, -43.771305084228516, 11.728486061096191, -21.13091278076172], [0.0, 0.0, 0.0, -43.771305084228516, 11.728486061096191, -21.13091278076172], [0.0, 0.0, 0.0, -43.771305084228516, 11.728486061096191, -21.13091278076172], [0.0, 0.0, 0.0, -43.771305084228516, 11.728486061096191, -21.13091278076172], [0.0, 0.0, 0.0, -43.771305084228516, 11.728486061096191, -21.13091278076172], [0.0, 0.0, 0.0, -43.771305084228516, 11.728486061096191, -21.13091278076172], [0.0, 0.0, 0.0, -43.771305084228516, 11.728486061096191, -21.13091278076172], [0.0, 0.0, 0.0, -43.771305084228516, 11.728486061096191, -21.13091278076172], [0.0, 0.0, 0.0, -43.771305084228516, 11.728486061096191, -21.13091278076172], [0.0, 0.0, 0.0, -43.771305084228516, 11.728486061096191, -21.13091278076172], [0.0, 0.0, 0.0, -43.771305084228516, 11.728486061096191, -21.13091278076172], [0.0, 0.0, 0.0, -43.771305084228516, 11.728486061096191, -21.13091278076172], [0.0, 0.0, 0.0, -43.771305084228516, 11.728486061096191, -21.13091278076172], [0.0, 0.0, 0.0, -43.771305084228516, 11.728486061096191, -21.13091278076172], [0.0, 0.0, 0.0, -43.771305084228516, 11.728486061096191, -21.13091278076172], [0.0, 0.0, 0.0, -43.771305084228516, 11.728486061096191, -21.13091278076172], [0.0, 0.0, 0.0, -43.771305084228516, 11.728486061096191, -21.13091278076172], [0.0, 0.0, 0.0, -43.771305084228516, 11.728486061096191, -21.13091278076172], [0.0, 0.0, 0.0, -43.771305084228516, 11.728486061096191, -21.13091278076172], [0.0, 0.0, 0.0, -43.771305084228516, 11.728486061096191, -21.13091278076172], [0.0, 0.0, 0.0, -43.771305084228516, 11.728486061096191, -21.13091278076172], [0.0, 0.0, 0.0, -43.771305084228516, 11.728486061096191, -21.13091278076172], [0.0, 0.0, 0.0, -43.771305084228516, 11.728486061096191, -21.13091278076172], [0.0, 0.0, 0.0, -43.771305084228516, 11.728486061096191, -21.13091278076172], [0.0, 0.0, 0.0, -43.771305084228516, 11.728486061096191, -21.13091278076172]], "color": "#ffa724", "variables": {}, "domain": {}}, {"type": "rays", "points": [[0.0, 0.0, 0.0, 48.188026428222656, -6.473121166229248, 11.662455558776855]], "color": "#210055", "variables": {}, "domain": {}}, {"type": "rays", "points": [[0.0, 0.0, 0.0, 48.18290328979492, -6.482099533081055, 11.678632736206055]], "color": "#3a008a", "variables": {}, "domain": {}}, {"type": "rays", "points": [[0.0, 0.0, 0.0, 48.177982330322266, -6.490705966949463, 11.694136619567871]], "color": "#5200c2", "variables": {}, "domain": {}}, {"type": "rays", "points": [[0.0, 0.0, 0.0, 48.173255920410156, -6.498956680297852, 11.709003448486328]], "color": "#6500f6", "variables": {}, "domain": {}}, {"type": "rays", "points": [[0.0, 0.0, 0.0, 48.16872024536133, -6.5068745613098145, 11.723268508911133]], "color": "#7200ff", "variables": {}, "domain": {}}, {"type": "rays", "points": [[0.0, 0.0, 0.0, 48.16436004638672, -6.514473915100098, 11.736960411071777]], "color": "#7700ff", "variables": {}, "domain": {}}, {"type": "rays", "points": [[0.0, 0.0, 0.0, 48.16016387939453, -6.521773338317871, 11.750110626220703]], "color": "#7500ff", "variables": {}, "domain": {}}, {"type": "rays", "points": [[0.0, 0.0, 0.0, 48.156131744384766, -6.528787136077881, 11.762747764587402]], "color": "#6a00ff", "variables": {}, "domain": {}}, {"type": "rays", "points": [[0.0, 0.0, 0.0, 48.15224838256836, -6.535530090332031, 11.774895668029785]], "color": "#5000ff", "variables": {}, "domain": {}}, {"type": "rays", "points": [[0.0, 0.0, 0.0, 48.14850616455078, -6.54201602935791, 11.786581993103027]], "color": "#0000ff", "variables": {}, "domain": {}}, {"type": "rays", "points": [[0.0, 0.0, 0.0, 48.14490509033203, -6.5482563972473145, 11.79782485961914]], "color": "#0038ff", "variables": {}, "domain": {}}, {"type": "rays", "points": [[0.0, 0.0, 0.0, 48.14143371582031, -6.55426549911499, 11.808650970458984]], "color": "#006dff", "variables": {}, "domain": {}}, {"type": "rays", "points": [[0.0, 0.0, 0.0, 48.138084411621094, -6.560053825378418, 11.819079399108887]], "color": "#0093ff", "variables": {}, "domain": {}}, {"type": "rays", "points": [[0.0, 0.0, 0.0, 48.13485336303711, -6.5656304359436035, 11.82912826538086]], "color": "#00ade6", "variables": {}, "domain": {}}, {"type": "rays", "points": [[0.0, 0.0, 0.0, 48.13174057006836, -6.57100772857666, 11.838814735412598]], "color": "#00c4be", "variables": {}, "domain": {}}, {"type": "rays", "points": [[0.0, 0.0, 0.0, 48.12873458862305, -6.576192378997803, 11.848155975341797]], "color": "#00d699", "variables": {}, "domain": {}}, {"type": "rays", "points": [[0.0, 0.0, 0.0, 48.125831604003906, -6.581196308135986, 11.857171058654785]], "color": "#00e975", "variables": {}, "domain": {}}, {"type": "rays", "points": [[0.0, 0.0, 0.0, 48.123023986816406, -6.586024761199951, 11.865870475769043]], "color": "#00fb4d", "variables": {}, "domain": {}}, {"type": "rays", "points": [[0.0, 0.0, 0.0, 48.12031555175781, -6.590688228607178, 11.874272346496582]], "color": "#00ff00", "variables": {}, "domain": {}}, {"type": "rays", "points": [[0.0, 0.0, 0.0, 48.11769485473633, -6.595192909240723, 11.8823881149292]], "color": "#00ff00", "variables": {}, "domain": {}}, {"type": "rays", "points": [[0.0, 0.0, 0.0, 48.11515808105469, -6.599544525146484, 11.890230178833008]], "color": "#00ff00", "variables": {}, "domain": {}}, {"type": "rays", "points": [[0.0, 0.0, 0.0, 48.112709045410156, -6.603753566741943, 11.897811889648438]], "color": "#00ff00", "variables": {}, "domain": {}}, {"type": "rays", "points": [[0.0, 0.0, 0.0, 48.11033630371094, -6.607822418212891, 11.905143737792969]], "color": "#00ff00", "variables": {}, "domain": {}}, {"type": "rays", "points": [[0.0, 0.0, 0.0, 48.10803985595703, -6.611758232116699, 11.912235260009766]], "color": "#00ff00", "variables": {}, "domain": {}}, {"type": "rays", "points": [[0.0, 0.0, 0.0, 48.10581588745117, -6.615568161010742, 11.919097900390625]], "color": "#3aff00", "variables": {}, "domain": {}}, {"type": "rays", "points": [[0.0, 0.0, 0.0, 48.103660583496094, -6.619255065917969, 11.925742149353027]], "color": "#a0ff00", "variables": {}, "domain": {}}, {"type": "rays", "points": [[0.0, 0.0, 0.0, 48.1015739440918, -6.622827053070068, 11.93217658996582]], "color": "#daff00", "variables": {}, "domain": {}}, {"type": "rays", "points": [[0.0, 0.0, 0.0, 48.099552154541016, -6.626286506652832, 11.938409805297852]], "color": "#ffff00", "variables": {}, "domain": {}}, {"type": "rays", "points": [[0.0, 0.0, 0.0, 48.097591400146484, -6.629639148712158, 11.944450378417969]], "color": "#fff300", "variables": {}, "domain": {}}, {"type": "rays", "points": [[0.0, 0.0, 0.0, 48.09568786621094, -6.632889270782471, 11.950304985046387]], "color": "#ffdc00", "variables": {}, "domain": {}}, {"type": "rays", "points": [[0.0, 0.0, 0.0, 48.093841552734375, -6.6360392570495605, 11.955981254577637]], "color": "#ffc200", "variables": {}, "domain": {}}, {"type": "rays", "points": [[0.0, 0.0, 0.0, 48.09204864501953, -6.639095306396484, 11.961487770080566]], "color": "#ffa500", "variables": {}, "domain": {}}, {"type": "rays", "points": [[0.0, 0.0, 0.0, 48.09031295776367, -6.642061233520508, 11.96683120727539]], "color": "#ff8400", "variables": {}, "domain": {}}, {"type": "rays", "points": [[0.0, 0.0, 0.0, 48.088623046875, -6.644938945770264, 11.972016334533691]], "color": "#ff5d00", "variables": {}, "domain": {}}, {"type": "rays", "points": [[0.0, 0.0, 0.0, 48.08698272705078, -6.647732734680176, 11.977048873901367]], "color": "#ff2200", "variables": {}, "domain": {}}, {"type": "rays", "points": [[0.0, 0.0, 0.0, 48.085391998291016, -6.650444984436035, 11.981935501098633]], "color": "#ff0000", "variables": {}, "domain": {}}, {"type": "rays", "points": [[0.0, 0.0, 0.0, 48.08384323120117, -6.653080940246582, 11.986682891845703]], "color": "#ff0000", "variables": {}, "domain": {}}, {"type": "rays", "points": [[0.0, 0.0, 0.0, 48.082340240478516, -6.655640125274658, 11.991294860839844]], "color": "#ff0000", "variables": {}, "domain": {}}, {"type": "rays", "points": [[0.0, 0.0, 0.0, 48.080875396728516, -6.658127784729004, 11.995777130126953]], "color": "#ff0000", "variables": {}, "domain": {}}, {"type": "rays", "points": [[0.0, 0.0, 0.0, 48.0794563293457, -6.660547256469727, 12.000136375427246]], "color": "#ff0000", "variables": {}, "domain": {}}, {"type": "rays", "points": [[0.0, 0.0, 0.0, 48.07807159423828, -6.662898540496826, 12.004372596740723]], "color": "#f30000", "variables": {}, "domain": {}}, {"type": "rays", "points": [[0.0, 0.0, 0.0, 48.076725006103516, -6.665185928344727, 12.008493423461914]], "color": "#d50000", "variables": {}, "domain": {}}, {"type": "rays", "points": [[0.0, 0.0, 0.0, 48.075416564941406, -6.6674113273620605, 12.012503623962402]], "color": "#b70000", "variables": {}, "domain": {}}, {"type": "rays", "points": [[0.0, 0.0, 0.0, 48.07414245605469, -6.669576644897461, 12.016403198242188]], "color": "#9c0000", "variables": {}, "domain": {}}, {"type": "rays", "points": [[0.0, 0.0, 0.0, 48.072898864746094, -6.671683311462402, 12.020200729370117]], "color": "#820000", "variables": {}, "domain": {}}, {"type": "rays", "points": [[0.0, 0.0, 0.0, 48.07168960571289, -6.673735618591309, 12.023897171020508]], "color": "#6b0000", "variables": {}, "domain": {}}, {"type": "rays", "points": [[0.0, 0.0, 0.0, 48.07051086425781, -6.675734519958496, 12.027497291564941]], "color": "#570000", "variables": {}, "domain": {}}, {"type": "rays", "points": [[0.0, 0.0, 0.0, 48.06936264038086, -6.677679061889648, 12.031002044677734]], "color": "#460000", "variables": {}, "domain": {}}, {"type": "rays", "points": [[0.0, 0.0, 0.0, 48.06824493408203, -6.679574489593506, 12.034418106079102]], "color": "#380000", "variables": {}, "domain": {}}, {"type": "rays", "points": [[0.0, 0.0, 0.0, 48.06715393066406, -6.681422233581543, 12.037745475769043]], "color": "#2c0000", "variables": {}, "domain": {}}], "show_optical_axis": false, "show_axes": false}';

tlmviewer.embed(document.getElementById("tlmviewer-d387c75d"), data);    
</script>



<div data-jp-suppress-context-menu id='tlmviewer-0fd6ab9d' class='tlmviewer' style='width: 100%; aspect-ratio: 16 / 9;'></div><script type='module'>async function importtlm() {
    try {
        return await import("/tlmviewer.js");
    } catch (error) {
        console.log("error", error);
        return await import("/files/test_notebooks/tlmviewer.js");
    }
}

const module = await importtlm();
const tlmviewer = module.tlmviewer;

const data = '{"mode": "3D", "camera": "orthographic", "data": [{"type": "surfaces", "data": [{"matrix": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], "samples": [[0.0, 0.0], [0.0, 0.0714249238371849], [0.0, 0.1428498476743698], [0.0, 0.21427476406097412], [0.0, 0.2856996953487396], [0.0, 0.3571246266365051], [0.0, 0.42854952812194824], [0.0, 0.49997445940971375], [0.0, 0.5713993906974792], [0.0, 0.6428242921829224], [0.0, 0.7142492532730103], [0.0, 0.7856741547584534], [0.0, 0.8570990562438965], [0.0, 0.9285240173339844], [0.0, 0.9999489188194275], [0.0, 1.0713738203048706], [0.0, 1.1427987813949585], [0.0, 1.2142237424850464], [0.0, 1.2856485843658447], [0.0, 1.3570735454559326], [0.0, 1.4284985065460205], [0.0, 1.4999233484268188], [0.0, 1.5713483095169067], [0.0, 1.6427732706069946], [0.0, 1.714198112487793], [0.0, 1.7856230735778809], [0.0, 1.8570480346679688], [0.0, 1.9284729957580566], [0.0, 1.999897837638855], [0.0, 2.0713226795196533], [0.0, 2.142747640609741], [0.0, 2.214172601699829], [0.0, 2.285597562789917], [0.0, 2.357022523880005], [0.0, 2.4284474849700928], [0.0, 2.4998724460601807], [0.0, 2.5712971687316895], [0.0, 2.6427221298217773], [0.0, 2.7141470909118652], [0.0, 2.785572052001953], [0.0, 2.856997013092041], [0.0, 2.928421974182129], [0.0, 2.9998466968536377], [0.0, 3.0712716579437256], [0.0, 3.1426966190338135], [0.0, 3.2141215801239014], [0.0, 3.2855465412139893], [0.0, 3.356971502304077], [0.0, 3.428396224975586], [0.0, 3.499821186065674], [0.0, 3.57124662399292], [0.0, 3.642671585083008], [0.0, 3.7140963077545166], [0.0, 3.7855212688446045], [0.0, 3.8569462299346924], [0.0, 3.9283711910247803], [0.0, 3.999796152114868], [0.0, 4.071220874786377], [0.0, 4.142645835876465], [0.0, 4.214070796966553], [0.0, 4.285495758056641], [0.0, 4.3569207191467285], [0.0, 4.428345680236816], [0.0, 4.499770641326904], [0.0, 4.571195602416992], [0.0, 4.64262056350708], [0.0, 4.714045524597168], [0.0, 4.785470008850098], [0.0, 4.8568949699401855], [0.0, 4.928319931030273], [0.0, 4.999744892120361], [0.0, 5.071169853210449], [0.0, 5.142594814300537], [0.0, 5.214019775390625], [0.0, 5.285444736480713], [0.0, 5.356869697570801], [0.0, 5.428294658660889], [0.0, 5.499719619750977], [0.0, 5.5711445808410645], [0.0, 5.642569541931152], [0.0, 5.713994026184082], [0.0, 5.78541898727417], [0.0, 5.856843948364258], [0.0, 5.928268909454346], [0.0, 5.999693870544434], [0.0, 6.0711188316345215], [0.0, 6.142543792724609], [0.0, 6.213968753814697], [0.0, 6.285393714904785], [0.0, 6.356818675994873], [0.0, 6.428243637084961], [0.0, 6.499668598175049], [0.0, 6.571093559265137], [0.0, 6.642518043518066], [0.0, 6.713943004608154], [0.0, 6.785367965698242], [0.0, 6.85679292678833], [0.0, 6.928217887878418], [0.0, 6.999642848968506], [0.0, 7.071067810058594]], "clip_planes": [[0.0, -1.0, 0.0, 5.0], [0.0, 1.0, 0.0, 5.0], [0.0, 0.0, -1.0, 5.0], [0.0, 0.0, 1.0, 5.0]]}]}, {"type": "points", "data": [[0.0, 0.0, 0.0]], "color": "#ff0000"}, {"type": "arrows", "data": [[-1, 0, 0, 0.0, 0.0, 0.0, 1.0]]}, {"type": "rays", "points": [[0.0, 0.0, 0.0, -43.771305084228516, 11.728486061096191, -21.13091278076172], [0.0, 0.0, 0.0, -43.771305084228516, 11.728486061096191, -21.13091278076172], [0.0, 0.0, 0.0, -43.771305084228516, 11.728486061096191, -21.13091278076172], [0.0, 0.0, 0.0, -43.771305084228516, 11.728486061096191, -21.13091278076172], [0.0, 0.0, 0.0, -43.771305084228516, 11.728486061096191, -21.13091278076172], [0.0, 0.0, 0.0, -43.771305084228516, 11.728486061096191, -21.13091278076172], [0.0, 0.0, 0.0, -43.771305084228516, 11.728486061096191, -21.13091278076172], [0.0, 0.0, 0.0, -43.771305084228516, 11.728486061096191, -21.13091278076172], [0.0, 0.0, 0.0, -43.771305084228516, 11.728486061096191, -21.13091278076172], [0.0, 0.0, 0.0, -43.771305084228516, 11.728486061096191, -21.13091278076172], [0.0, 0.0, 0.0, -43.771305084228516, 11.728486061096191, -21.13091278076172], [0.0, 0.0, 0.0, -43.771305084228516, 11.728486061096191, -21.13091278076172], [0.0, 0.0, 0.0, -43.771305084228516, 11.728486061096191, -21.13091278076172], [0.0, 0.0, 0.0, -43.771305084228516, 11.728486061096191, -21.13091278076172], [0.0, 0.0, 0.0, -43.771305084228516, 11.728486061096191, -21.13091278076172], [0.0, 0.0, 0.0, -43.771305084228516, 11.728486061096191, -21.13091278076172], [0.0, 0.0, 0.0, -43.771305084228516, 11.728486061096191, -21.13091278076172], [0.0, 0.0, 0.0, -43.771305084228516, 11.728486061096191, -21.13091278076172], [0.0, 0.0, 0.0, -43.771305084228516, 11.728486061096191, -21.13091278076172], [0.0, 0.0, 0.0, -43.771305084228516, 11.728486061096191, -21.13091278076172], [0.0, 0.0, 0.0, -43.771305084228516, 11.728486061096191, -21.13091278076172], [0.0, 0.0, 0.0, -43.771305084228516, 11.728486061096191, -21.13091278076172], [0.0, 0.0, 0.0, -43.771305084228516, 11.728486061096191, -21.13091278076172], [0.0, 0.0, 0.0, -43.771305084228516, 11.728486061096191, -21.13091278076172], [0.0, 0.0, 0.0, -43.771305084228516, 11.728486061096191, -21.13091278076172], [0.0, 0.0, 0.0, -43.771305084228516, 11.728486061096191, -21.13091278076172], [0.0, 0.0, 0.0, -43.771305084228516, 11.728486061096191, -21.13091278076172], [0.0, 0.0, 0.0, -43.771305084228516, 11.728486061096191, -21.13091278076172], [0.0, 0.0, 0.0, -43.771305084228516, 11.728486061096191, -21.13091278076172], [0.0, 0.0, 0.0, -43.771305084228516, 11.728486061096191, -21.13091278076172], [0.0, 0.0, 0.0, -43.771305084228516, 11.728486061096191, -21.13091278076172], [0.0, 0.0, 0.0, -43.771305084228516, 11.728486061096191, -21.13091278076172], [0.0, 0.0, 0.0, -43.771305084228516, 11.728486061096191, -21.13091278076172], [0.0, 0.0, 0.0, -43.771305084228516, 11.728486061096191, -21.13091278076172], [0.0, 0.0, 0.0, -43.771305084228516, 11.728486061096191, -21.13091278076172], [0.0, 0.0, 0.0, -43.771305084228516, 11.728486061096191, -21.13091278076172], [0.0, 0.0, 0.0, -43.771305084228516, 11.728486061096191, -21.13091278076172], [0.0, 0.0, 0.0, -43.771305084228516, 11.728486061096191, -21.13091278076172], [0.0, 0.0, 0.0, -43.771305084228516, 11.728486061096191, -21.13091278076172], [0.0, 0.0, 0.0, -43.771305084228516, 11.728486061096191, -21.13091278076172], [0.0, 0.0, 0.0, -43.771305084228516, 11.728486061096191, -21.13091278076172], [0.0, 0.0, 0.0, -43.771305084228516, 11.728486061096191, -21.13091278076172], [0.0, 0.0, 0.0, -43.771305084228516, 11.728486061096191, -21.13091278076172], [0.0, 0.0, 0.0, -43.771305084228516, 11.728486061096191, -21.13091278076172], [0.0, 0.0, 0.0, -43.771305084228516, 11.728486061096191, -21.13091278076172], [0.0, 0.0, 0.0, -43.771305084228516, 11.728486061096191, -21.13091278076172], [0.0, 0.0, 0.0, -43.771305084228516, 11.728486061096191, -21.13091278076172], [0.0, 0.0, 0.0, -43.771305084228516, 11.728486061096191, -21.13091278076172], [0.0, 0.0, 0.0, -43.771305084228516, 11.728486061096191, -21.13091278076172], [0.0, 0.0, 0.0, -43.771305084228516, 11.728486061096191, -21.13091278076172]], "color": "#ffa724", "variables": {}, "domain": {}}, {"type": "rays", "points": [[0.0, 0.0, 0.0, 24.13612937927246, -21.250551223754883, 38.28657150268555]], "color": "#210055", "variables": {}, "domain": {}}, {"type": "rays", "points": [[0.0, 0.0, 0.0, 24.245840072631836, -21.22111701965332, 38.23353958129883]], "color": "#3a008a", "variables": {}, "domain": {}}, {"type": "rays", "points": [[0.0, 0.0, 0.0, 24.35011100769043, -21.19297981262207, 38.18284606933594]], "color": "#5200c2", "variables": {}, "domain": {}}, {"type": "rays", "points": [[0.0, 0.0, 0.0, 24.449277877807617, -21.166072845458984, 38.134368896484375]], "color": "#6500f6", "variables": {}, "domain": {}}, {"type": "rays", "points": [[0.0, 0.0, 0.0, 24.543710708618164, -21.14031982421875, 38.08796691894531]], "color": "#7200ff", "variables": {}, "domain": {}}, {"type": "rays", "points": [[0.0, 0.0, 0.0, 24.633686065673828, -21.11565589904785, 38.04353332519531]], "color": "#7700ff", "variables": {}, "domain": {}}, {"type": "rays", "points": [[0.0, 0.0, 0.0, 24.719497680664062, -21.092025756835938, 38.00095748901367]], "color": "#7500ff", "variables": {}, "domain": {}}, {"type": "rays", "points": [[0.0, 0.0, 0.0, 24.801414489746094, -21.069366455078125, 37.96013259887695]], "color": "#6a00ff", "variables": {}, "domain": {}}, {"type": "rays", "points": [[0.0, 0.0, 0.0, 24.879663467407227, -21.047630310058594, 37.92097091674805]], "color": "#5000ff", "variables": {}, "domain": {}}, {"type": "rays", "points": [[0.0, 0.0, 0.0, 24.954463958740234, -21.026763916015625, 37.88337707519531]], "color": "#0000ff", "variables": {}, "domain": {}}, {"type": "rays", "points": [[0.0, 0.0, 0.0, 25.026031494140625, -21.0067195892334, 37.847267150878906]], "color": "#0038ff", "variables": {}, "domain": {}}, {"type": "rays", "points": [[0.0, 0.0, 0.0, 25.094545364379883, -20.98746109008789, 37.81256866455078]], "color": "#006dff", "variables": {}, "domain": {}}, {"type": "rays", "points": [[0.0, 0.0, 0.0, 25.160186767578125, -20.968944549560547, 37.779205322265625]], "color": "#0093ff", "variables": {}, "domain": {}}, {"type": "rays", "points": [[0.0, 0.0, 0.0, 25.223112106323242, -20.95113182067871, 37.74711608886719]], "color": "#00ade6", "variables": {}, "domain": {}}, {"type": "rays", "points": [[0.0, 0.0, 0.0, 25.283475875854492, -20.933988571166992, 37.71622848510742]], "color": "#00c4be", "variables": {}, "domain": {}}, {"type": "rays", "points": [[0.0, 0.0, 0.0, 25.341421127319336, -20.917482376098633, 37.68648910522461]], "color": "#00d699", "variables": {}, "domain": {}}, {"type": "rays", "points": [[0.0, 0.0, 0.0, 25.397062301635742, -20.901580810546875, 37.657840728759766]], "color": "#00e975", "variables": {}, "domain": {}}, {"type": "rays", "points": [[0.0, 0.0, 0.0, 25.450542449951172, -20.886255264282227, 37.63022994995117]], "color": "#00fb4d", "variables": {}, "domain": {}}, {"type": "rays", "points": [[0.0, 0.0, 0.0, 25.50196075439453, -20.87148094177246, 37.60361099243164]], "color": "#00ff00", "variables": {}, "domain": {}}, {"type": "rays", "points": [[0.0, 0.0, 0.0, 25.551435470581055, -20.857223510742188, 37.57792282104492]], "color": "#00ff00", "variables": {}, "domain": {}}, {"type": "rays", "points": [[0.0, 0.0, 0.0, 25.599056243896484, -20.843467712402344, 37.553138732910156]], "color": "#00ff00", "variables": {}, "domain": {}}, {"type": "rays", "points": [[0.0, 0.0, 0.0, 25.64491844177246, -20.830184936523438, 37.52920913696289]], "color": "#00ff00", "variables": {}, "domain": {}}, {"type": "rays", "points": [[0.0, 0.0, 0.0, 25.68910789489746, -20.817358016967773, 37.50609588623047]], "color": "#00ff00", "variables": {}, "domain": {}}, {"type": "rays", "points": [[0.0, 0.0, 0.0, 25.731706619262695, -20.804964065551758, 37.483768463134766]], "color": "#00ff00", "variables": {}, "domain": {}}, {"type": "rays", "points": [[0.0, 0.0, 0.0, 25.772785186767578, -20.792984008789062, 37.46218490600586]], "color": "#3aff00", "variables": {}, "domain": {}}, {"type": "rays", "points": [[0.0, 0.0, 0.0, 25.81243133544922, -20.78139877319336, 37.441314697265625]], "color": "#a0ff00", "variables": {}, "domain": {}}, {"type": "rays", "points": [[0.0, 0.0, 0.0, 25.85070037841797, -20.77019500732422, 37.42112350463867]], "color": "#daff00", "variables": {}, "domain": {}}, {"type": "rays", "points": [[0.0, 0.0, 0.0, 25.887662887573242, -20.759347915649414, 37.40158462524414]], "color": "#ffff00", "variables": {}, "domain": {}}, {"type": "rays", "points": [[0.0, 0.0, 0.0, 25.923368453979492, -20.748849868774414, 37.38267135620117]], "color": "#fff300", "variables": {}, "domain": {}}, {"type": "rays", "points": [[0.0, 0.0, 0.0, 25.957881927490234, -20.73868751525879, 37.364356994628906]], "color": "#ffdc00", "variables": {}, "domain": {}}, {"type": "rays", "points": [[0.0, 0.0, 0.0, 25.991260528564453, -20.728837966918945, 37.346614837646484]], "color": "#ffc200", "variables": {}, "domain": {}}, {"type": "rays", "points": [[0.0, 0.0, 0.0, 26.02354621887207, -20.719297409057617, 37.32942199707031]], "color": "#ffa500", "variables": {}, "domain": {}}, {"type": "rays", "points": [[0.0, 0.0, 0.0, 26.054790496826172, -20.710046768188477, 37.31275939941406]], "color": "#ff8400", "variables": {}, "domain": {}}, {"type": "rays", "points": [[0.0, 0.0, 0.0, 26.085037231445312, -20.701078414916992, 37.296600341796875]], "color": "#ff5d00", "variables": {}, "domain": {}}, {"type": "rays", "points": [[0.0, 0.0, 0.0, 26.11432647705078, -20.692378997802734, 37.28092956542969]], "color": "#ff2200", "variables": {}, "domain": {}}, {"type": "rays", "points": [[0.0, 0.0, 0.0, 26.142704010009766, -20.68393898010254, 37.265716552734375]], "color": "#ff0000", "variables": {}, "domain": {}}, {"type": "rays", "points": [[0.0, 0.0, 0.0, 26.170202255249023, -20.675745010375977, 37.25096130371094]], "color": "#ff0000", "variables": {}, "domain": {}}, {"type": "rays", "points": [[0.0, 0.0, 0.0, 26.19686508178711, -20.66779327392578, 37.236629486083984]], "color": "#ff0000", "variables": {}, "domain": {}}, {"type": "rays", "points": [[0.0, 0.0, 0.0, 26.222721099853516, -20.660070419311523, 37.22271728515625]], "color": "#ff0000", "variables": {}, "domain": {}}, {"type": "rays", "points": [[0.0, 0.0, 0.0, 26.247802734375, -20.65256690979004, 37.209197998046875]], "color": "#ff0000", "variables": {}, "domain": {}}, {"type": "rays", "points": [[0.0, 0.0, 0.0, 26.272136688232422, -20.645278930664062, 37.196067810058594]], "color": "#f30000", "variables": {}, "domain": {}}, {"type": "rays", "points": [[0.0, 0.0, 0.0, 26.295761108398438, -20.638195037841797, 37.18330383300781]], "color": "#d50000", "variables": {}, "domain": {}}, {"type": "rays", "points": [[0.0, 0.0, 0.0, 26.31869888305664, -20.631305694580078, 37.170894622802734]], "color": "#b70000", "variables": {}, "domain": {}}, {"type": "rays", "points": [[0.0, 0.0, 0.0, 26.340978622436523, -20.624610900878906, 37.15882873535156]], "color": "#9c0000", "variables": {}, "domain": {}}, {"type": "rays", "points": [[0.0, 0.0, 0.0, 26.362627029418945, -20.618093490600586, 37.14708709716797]], "color": "#820000", "variables": {}, "domain": {}}, {"type": "rays", "points": [[0.0, 0.0, 0.0, 26.383668899536133, -20.611753463745117, 37.13566589355469]], "color": "#6b0000", "variables": {}, "domain": {}}, {"type": "rays", "points": [[0.0, 0.0, 0.0, 26.404117584228516, -20.6055850982666, 37.12455368041992]], "color": "#570000", "variables": {}, "domain": {}}, {"type": "rays", "points": [[0.0, 0.0, 0.0, 26.423995971679688, -20.59958267211914, 37.113739013671875]], "color": "#460000", "variables": {}, "domain": {}}, {"type": "rays", "points": [[0.0, 0.0, 0.0, 26.443344116210938, -20.593734741210938, 37.10320281982422]], "color": "#380000", "variables": {}, "domain": {}}, {"type": "rays", "points": [[0.0, 0.0, 0.0, 26.4621639251709, -20.588041305541992, 37.09294509887695]], "color": "#2c0000", "variables": {}, "domain": {}}], "show_optical_axis": false, "show_axes": false}';

tlmviewer.embed(document.getElementById("tlmviewer-0fd6ab9d"), data);    
</script>

