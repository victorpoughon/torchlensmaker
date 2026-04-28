```python
# Example from Modern Optical Engineering figure 14.4

import torchlensmaker as tlm
import indicio

D = 25
BK7 = tlm.material_from_indicio(indicio.get_material(shelf="specs", book="SCHOTT-optical", page="N-BK7"))
SF1 = tlm.material_from_indicio(indicio.get_material(shelf="specs", book="SCHOTT-optical", page="SF1"))

print(BK7)
print(SF1)

L1 = tlm.lenses.singlet(
    tlm.SphereByCurvature(D, 1/50.098),
    tlm.InnerGap(4.5),
    tlm.SphereByCurvature(D, 1/-983.420),
    material=BK7,
)

L2 = tlm.lenses.singlet(
    tlm.SphereByCurvature(D, 1/56.671),
    tlm.InnerGap(4.5),
    tlm.SphereByCurvature(D, 1/-171.150),
    material=BK7,
)

L3 = tlm.lenses.singlet(
    tlm.SphereByCurvature(0.85*D, 1/-97.339),
    tlm.InnerGap(3.5),
    tlm.SphereByCurvature(0.85*D, 1/81.454),
    material=SF1,
)

lens = tlm.Sequential(
    L1,
    tlm.Gap(0.1),
    L2,
    tlm.Gap(5.571),
    L3,
)

optics = tlm.Sequential(
    tlm.ObjectAtInfinity(0.9*D, 2.0),
    lens,
    #tlm.Gap(75.132),
    tlm.AbsolutePosition2D(),
    tlm.Gap(93.2791),
    tlm.ImagePlane(20),
)

tlm.set_sampling2d(optics, pupil=10, wavel=5, field=5)
tlm.show2d(optics, end=10)

rfp = tlm.paraxial.rear_focal_point(lens, 500, 0.01)
print(rfp)
```

    SellmeierMaterial(B1=1.0396121740341187, B2=0.23179234564304352, B3=1.0104694366455078, C1=0.006000698544085026, C2=0.020017914474010468, C3=103.56065368652344)
    SellmeierMaterial(B1=1.559129238128662, B2=0.2842462956905365, B3=0.9688429236412048, C1=0.012148099951446056, C2=0.05345490574836731, C3=112.17481231689453)



<TLMViewer src="./refractive_index_database_files/refractive_index_database_0.json?url" />


    tensor([93.4247])

