```python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import math
import torch
import torchlensmaker as tlm

from torch.nn.functional import normalize

from torchlensmaker.core.collision_detection import Newton, GD, LM, CollisionMethod

from torchlensmaker.testing.collision_datasets import CollisionDataset
from torchlensmaker.testing.dataset_view import dataset_view

from torchlensmaker.core.geometry import unit3d_rot

import matplotlib as mpl

from IPython.display import display, HTML

from typing import TypeAlias

Tensor: TypeAlias = torch.Tensor


def analysis_single_ray(surface, P, V):
    dim = P.shape[0]
    P, V = P.unsqueeze(0), V.unsqueeze(0)

    dataset_view(surface, P, V, rays_length=100)

    results = surface.collision_method(surface, P, V, history=True)
    t_solve, t_history = results.t, results.history_fine
    t_min = t_history.min().item()
    t_max = t_history.max().item()

    N = 1000
    H = t_history.size(1)
    tspace = torch.linspace(t_min - (t_max - t_min), t_max + (t_max - t_min), N)

    # t plot
    tpoints = P.expand((N, dim)) + tspace.unsqueeze(1).expand((N, dim)) * V.expand((N, dim))
    Q = surface.Fd(tpoints)
    Qgrad = torch.sum(surface.Fd_grad(tpoints) * V, dim=1)

    assert tpoints.size() == (N, dim)
    assert Q.size() == (N,)
    assert Qgrad.size() == (N,)
    assert t_history.size() == (1, H)
    assert t_solve.size() == (1,)
    
    points_history = P + t_history[0, :].unsqueeze(1).expand((-1, dim)) * V
    assert points_history.size() == (H, dim), points_history.size()
    
    final_point = (P + t_solve.unsqueeze(0).expand(-1, dim) * V).squeeze(0)
    assert final_point.size() == (dim,)
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 5))
    fig.tight_layout(pad=3, w_pad=1.8, h_pad=3)
    ax_t, ax_iter = axes

    # t plot: plot Q and Q grad
    ax_t.plot(tspace.detach().numpy(), Q.detach().numpy(), label="Q(t)=F(P+tV)")
    ax_t.plot(tspace.detach().numpy(), Qgrad.detach().numpy(), label="Q'(t) = F_grad . V")
    ax_t.grid()
    ax_t.set_xlabel("t")
    ax_t.legend()

    F_history = surface.Fd(points_history)
    print(F_history)

    # t plot: plot t history
    ax_t.scatter(t_history[0, :], F_history, c=range(t_history.shape[1]), cmap="viridis", marker="o")

    # History plot: plot F
    ax_iter.plot(range(t_history.shape[1]), torch.abs(F_history), label="|F(P+tV)|")
    ax_iter.legend()
    ax_iter.set_xlabel("iteration")
    ax_iter.set_title(f"final F = {surface.Fd(final_point.unsqueeze(0))[0].item():.6f}")
    ax_iter.set_yscale("log")
    ax_iter.grid()
    ax_iter.set_ylim([1e-8, 100])

    #fig.suptitle(surface.testname() + " " + str(surface.collision_method))
    plt.show(fig)
    display(HTML("<hr/>"))


# 3D baby!
analysis_single_ray(tlm.Sphere(30, R=20),
                    P=torch.tensor([5.0, 1.0, 2.0]),
                    V=unit3d_rot(15.0, 35.0))

```


<div data-jp-suppress-context-menu id='tlmviewer-f64fec40' class='tlmviewer' style='width: 100%; aspect-ratio: 16 / 9;'></div><script type='module'>async function importtlm() {
    try {
        return await import("/tlmviewer.js");
    } catch (error) {
        console.log("error", error);
        return await import("/files/test_notebooks/tlmviewer.js");
    }
}

const module = await importtlm();
const tlmviewer = module.tlmviewer;

const data = '{"mode": "3D", "camera": "orthographic", "data": [{"type": "points", "data": [[5.0, 1.0, 2.0]], "color": "grey"}, {"type": "points", "data": [[0.6717832607957765, -0.15974217993663875, 5.137559743525562]], "color": "#ff0000"}, {"type": "arrows", "data": [[-0.9664108369602115, -0.007987108996831936, 0.2568779871762781, 0.6717832607957765, -0.15974217993663875, 5.137559743525562, 1.0]]}, {"type": "rays", "points": [[-74.12401152362239, -20.201214989665463, 59.35764363510461, 84.12401152362239, 22.201214989665463, -55.35764363510461]], "color": "#ffa724", "variables": {}, "domain": {}, "layers": [0]}, {"type": "surfaces", "data": [{"matrix": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], "samples": [[6.771243444677049, 15.000000000000002], [6.643236116793052, 14.8861297377209], [6.516208918681908, 14.771167119607263], [6.390171171698679, 14.655120581705848], [6.265132124591755, 14.537998639602307], [6.141100952824136, 14.419809887796301], [6.018086757900141, 14.300562999070792], [5.896098566697539, 14.18026672385568], [5.775145330805145, 14.058929889585666], [5.655235925865929, 13.936561400052472], [5.53637915092575, 13.813170234751496], [5.418583727787654, 13.688765448222897], [5.3018583003718405, 13.563356169387127], [5.186211434081404, 13.436951600875084], [5.071651615173792, 13.309561018352811], [4.958187250138032, 13.1811937698408], [4.845826665077917, 13.051859275028079], [4.734578105101001, 12.92156702458097], [4.624449733713554, 12.790326579446623], [4.5154496322215465, 12.658147570151495], [4.407585799137607, 12.525039696094574], [4.300866149594118, 12.391012724835697], [4.195298514762371, 12.256076491378776], [4.090890641277909, 12.120240897450065], [3.9876501906720954, 11.98351591077162], [3.885584738809886, 11.845911564329828], [3.7847017753339003, 11.707437955639161], [3.6850087031148533, 11.56810524600126], [3.5865128377082947, 11.427923659759259], [3.489221406817812, 11.286903483547503], [3.3931415497646356, 11.145055065536747], [3.2982803169637798, 11.002388814674774], [3.204644669406637, 10.858915199922556], [3.11224147815021, 10.714644749486089], [3.021077523812899, 10.569588050043787], [2.9311594960769085, 10.423755745969615], [2.842493993197401, 10.27715853855202], [2.7550875215182735, 10.129807185208675], [2.668946494994728, 9.981712498697028], [2.5840772347226135, 9.832885346320934], [2.5004859684745924, 9.683336649133157], [2.418178830243093, 9.53307738113396], [2.337161859790264, 9.382118568465877], [2.2574410022047076, 9.230471288604571], [2.1790221074652614, 9.078146669545943], [2.1019109300117087, 8.925155888989597], [2.026113128322528, 8.771510173518593], [1.9516342644996314, 8.617220797775595], [1.878479803860273, 8.462299083635601], [1.806655114535932, 8.306756399375057], [1.7361654670784574, 8.15060415883771], [1.6670160340732743, 7.993853820597019], [1.5992118897598075, 7.836516887115298], [1.5327580096591724, 7.678604903899701], [1.467659270209026, 7.520129458654983], [1.4039204484057386, 7.361102180433158], [1.3415462214538785, 7.2015347387801985], [1.280541166422971, 7.041438842879693], [1.220909759911624, 6.88082624069359], [1.1626563777190597, 6.719708718100175], [1.10578529452399, 6.558098098029187], [1.0503006835709456, 6.396006239594217], [0.9962066163640451, 6.233445037222532], [0.9435070623682158, 6.07042641978222], [0.8922058887179212, 5.906962349706822], [0.8423068599333767, 5.743064822117563], [0.7938136376443126, 5.578745863943122], [0.7467297803212851, 5.41401753303706], [0.701058743014535, 5.248891917293051], [0.6568038771004865, 5.0833811337578405], [0.6139684300357757, 4.9174973277420655], [0.5725555451189912, 4.751252671929068], [0.5325682612599962, 4.58465936548163], [0.4940095127569357, 4.417729633146763], [0.45688212908090264, 4.250475724358701], [0.42118883466833523, 4.082909912339996], [0.3869322487210738, 3.915044493200882], [0.35411488501416244, 3.74689178503702], [0.32273915171140644, 3.5784641270255637], [0.29280735118863177, 3.4097738785196774], [0.26432167986475363, 3.240833418141647], [0.237284228040604, 3.0716551428745005], [0.21169697974552548, 2.9022514671522814], [0.18756181259178817, 2.732634821949123], [0.16488049763683676, 2.562817653867024], [0.14365469925327545, 2.3928124242224826], [0.123885975006786, 2.2226316081321373], [0.10557577554180142, 2.0522876935972607], [0.0887254444750667, 1.8817931805874453], [0.07333621829705095, 1.7111605801233145], [0.059409226281196936, 1.540402413358433], [0.046945490401057555, 1.3695312106605444], [0.035945925255319366, 1.1985595106920572], [0.026411338000670526, 1.0274998594899218], [0.018342428292569934, 0.8563648095450456], [0.011739788233921189, 0.6851669188811543], [0.0066039023316122325, 0.5139187501332485], [0.0029351474609669026, 0.34263286962580014], [0.0007337928380799497, 0.17132184645059978], [7.460698725481052e-14, -1.748455600074495e-06]]}]}]}';

setTimeout(() => {
    tlmviewer.embed(document.getElementById("tlmviewer-f64fec40"), data);    
}, 0);
</script>


    tensor([-7.8810e-08, -1.5762e-08, -3.1524e-09, -6.3048e-10, -1.2610e-10,
            -2.5219e-11, -5.0436e-12, -1.0095e-12, -2.0095e-13, -4.0634e-14],
           dtype=torch.float64)



    
![png](collision_detection_analysis_single_ray_files/collision_detection_analysis_single_ray_0_2.png)
    



<hr/>

