import tlmviewer as tlmv

ID_MATRIX: list[list[float]] = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]

ELEMENT_TYPE_STRINGS: dict[type, str] = {
    tlmv.AmbientLight: "ambient-light",
    tlmv.DirectionalLight: "directional-light",
    tlmv.SceneAxis: "scene-axis",
    tlmv.SceneTitle: "scene-title",
    tlmv.Arrows: "arrows",
    tlmv.Points: "points",
    tlmv.Rays: "rays",
    tlmv.Box3D: "box3D",
    tlmv.Cylinder: "cylinder",
    tlmv.SurfaceDisk: "surface-disk",
    tlmv.SurfaceLathe: "surface-lathe",
    tlmv.SurfaceSphereR: "surface-sphere-r",
    tlmv.SurfaceSag: "surface-sag",
    tlmv.SurfaceBSpline: "surface-bspline",
}

ELEMENTS = [
    tlmv.AmbientLight(color="#fff", intensity=1.0),
    tlmv.DirectionalLight(color="#fff", intensity=1.0, position=(0, 1, 0)),
    tlmv.SceneAxis(axis="x", length=10.0, color="#f00"),
    tlmv.SceneTitle(title="test"),
    tlmv.Arrows(arrows=[[0, 0, 0, 1, 0, 0, 1]]),
    tlmv.Points(data=[[0, 0, 0]], color="#f00", radius=0.1, category="kinematic-joint"),
    tlmv.Rays(points=[[0, 0], [1, 0]], color="#f00", category="rays-valid"),
    tlmv.Box3D(size=(1, 1, 1), matrix=ID_MATRIX),
    tlmv.Cylinder(xmin=-1.0, xmax=1.0, radius=2.0, matrix=ID_MATRIX),
    tlmv.SurfaceDisk(radius=5.0, matrix=ID_MATRIX),
    tlmv.SurfaceLathe(samples=[[0, 0], [1, 1]], matrix=ID_MATRIX),
    tlmv.SurfaceSphereR(R=10.0, diameter=5.0, matrix=ID_MATRIX),
    tlmv.SurfaceSag(
        diameter=5.0, sag_function={"sag-type": "spherical", "C": 0.1}, matrix=ID_MATRIX
    ),
    tlmv.SurfaceBSpline(
        points=[[[0, 0, 0]]],
        weights=[[1.0]],
        degree=(2, 2),
        knot_type="clamped",
        samples=(10, 10),
        matrix=ID_MATRIX,
    ),
]
