# Export with build123


```python
import torchlensmaker as tlm
from torchlensmaker.export_build123d import surface_to_sketch

from IPython.display import display

test_shapes = [
    tlm.Sphere(15.0, 20.0),
    tlm.Sphere(15.0, -20.0),
    tlm.Parabola(15.0, 0.02),
    tlm.Parabola(15.0, -0.02),
    tlm.CircularPlane(15.0),
]

for shape in test_shapes:
    sk = surface_to_sketch(shape)
    display(sk)
```


<div id=shape-b92a7a97></div><script>function render(data, div_id, ratio){

    // Initial setup
    const renderWindow = vtk.Rendering.Core.vtkRenderWindow.newInstance();
    const renderer = vtk.Rendering.Core.vtkRenderer.newInstance({ background: [1, 1, 1 ] });
    renderWindow.addRenderer(renderer);

    // iterate over all children children
    for (var el of data){
        var trans = el.position;
        var rot = el.orientation;
        var rgba = el.color;
        var shape = el.shape;

        // load the inline data
        var reader = vtk.IO.XML.vtkXMLPolyDataReader.newInstance();
        const textEncoder = new TextEncoder();
        reader.parseAsArrayBuffer(textEncoder.encode(shape));

        // setup actor,mapper and add
        const mapper = vtk.Rendering.Core.vtkMapper.newInstance();
        mapper.setInputConnection(reader.getOutputPort());
        mapper.setResolveCoincidentTopologyToPolygonOffset();
        mapper.setResolveCoincidentTopologyPolygonOffsetParameters(0.5,100);

        const actor = vtk.Rendering.Core.vtkActor.newInstance();
        actor.setMapper(mapper);

        // set color and position
        actor.getProperty().setColor(rgba.slice(0,3));
        actor.getProperty().setOpacity(rgba[3]);

        actor.rotateZ(rot[2]*180/Math.PI);
        actor.rotateY(rot[1]*180/Math.PI);
        actor.rotateX(rot[0]*180/Math.PI);

        actor.setPosition(trans);

        renderer.addActor(actor);

    };

    renderer.resetCamera();

    const openglRenderWindow = vtk.Rendering.OpenGL.vtkRenderWindow.newInstance();
    renderWindow.addView(openglRenderWindow);

    // Get the div container    
    const container = document.getElementById(div_id);
    const dims = container.parentElement.getBoundingClientRect();

    openglRenderWindow.setContainer(container);

    // handle size
    if (ratio){
        openglRenderWindow.setSize(dims.width, dims.width*ratio);
    }else{
        openglRenderWindow.setSize(dims.width, dims.height);
    };

    // Interaction setup
    const interact_style = vtk.Interaction.Style.vtkInteractorStyleManipulator.newInstance();

    const manips = {
        rot: vtk.Interaction.Manipulators.vtkMouseCameraTrackballRotateManipulator.newInstance(),
        pan: vtk.Interaction.Manipulators.vtkMouseCameraTrackballPanManipulator.newInstance(),
        zoom1: vtk.Interaction.Manipulators.vtkMouseCameraTrackballZoomManipulator.newInstance(),
        zoom2: vtk.Interaction.Manipulators.vtkMouseCameraTrackballZoomManipulator.newInstance(),
        roll: vtk.Interaction.Manipulators.vtkMouseCameraTrackballRollManipulator.newInstance(),
    };

    manips.zoom1.setControl(true);
    manips.zoom2.setScrollEnabled(true);
    manips.roll.setShift(true);
    manips.pan.setButton(2);

    for (var k in manips){
        interact_style.addMouseManipulator(manips[k]);
    };

    const interactor = vtk.Rendering.Core.vtkRenderWindowInteractor.newInstance();
    interactor.setView(openglRenderWindow);
    interactor.initialize();
    interactor.bindEvents(container);
    interactor.setInteractorStyle(interact_style);

    // Orientation marker

    const axes = vtk.Rendering.Core.vtkAnnotatedCubeActor.newInstance();
    axes.setXPlusFaceProperty({text: '+X'});
    axes.setXMinusFaceProperty({text: '-X'});
    axes.setYPlusFaceProperty({text: '+Y'});
    axes.setYMinusFaceProperty({text: '-Y'});
    axes.setZPlusFaceProperty({text: '+Z'});
    axes.setZMinusFaceProperty({text: '-Z'});

    const orientationWidget = vtk.Interaction.Widgets.vtkOrientationMarkerWidget.newInstance({
        actor: axes,
        interactor: interactor });
    orientationWidget.setEnabled(true);
    orientationWidget.setViewportCorner(vtk.Interaction.Widgets.vtkOrientationMarkerWidget.Corners.BOTTOM_LEFT);
    orientationWidget.setViewportSize(0.2);

};


new Promise(
  function(resolve, reject)
  {
    if (typeof(require) !== "undefined" ){
        require.config({
         "paths": {"vtk": "https://unpkg.com/vtk"},
        });
        require(["vtk"], resolve, reject);
    } else if ( typeof(vtk) === "undefined" ){
        var script = document.createElement("script");
    	script.onload = resolve;
    	script.onerror = reject;
    	script.src = "https://unpkg.com/vtk.js";
    	document.head.appendChild(script);
    } else { resolve() };
 }
).then(() => {
    // data, div_id and ratio are templated by python
    const div_id = "shape-b92a7a97";
    const data = [{"shape": "<?xml version=\"1.0\"?>\n<VTKFile type=\"PolyData\" version=\"0.1\" byte_order=\"LittleEndian\" header_type=\"UInt32\" compressor=\"vtkZLibDataCompressor\">\n  <PolyData>\n    <Piece NumberOfPoints=\"19\"                   NumberOfVerts=\"2\"                    NumberOfLines=\"16\"                   NumberOfStrips=\"0\"                    NumberOfPolys=\"0\"                   >\n      <PointData Normals=\"Normals\">\n        <DataArray type=\"Float32\" Name=\"Normals\" NumberOfComponents=\"3\" format=\"appended\" RangeMin=\"1\"                    RangeMax=\"1\"                    offset=\"0\"                   />\n      </PointData>\n      <CellData>\n        <DataArray type=\"Int64\" IdType=\"1\" Name=\"SUBSHAPE_IDS\" format=\"appended\" RangeMin=\"-1\"                   RangeMax=\"3\"                    offset=\"48\"                  />\n        <DataArray type=\"Int64\" IdType=\"1\" Name=\"MESH_TYPES\" format=\"appended\" RangeMin=\"2\"                    RangeMax=\"3\"                    offset=\"96\"                  />\n      </CellData>\n      <Points>\n        <DataArray type=\"Float32\" Name=\"Points\" NumberOfComponents=\"3\" format=\"appended\" RangeMin=\"0.05978449434\"        RangeMax=\"7.6294963101\"         offset=\"144\"                 />\n      </Points>\n      <Verts>\n        <DataArray type=\"Int64\" Name=\"connectivity\" format=\"appended\" RangeMin=\"\"                     RangeMax=\"\"                     offset=\"336\"                 />\n        <DataArray type=\"Int64\" Name=\"offsets\" format=\"appended\" RangeMin=\"\"                     RangeMax=\"\"                     offset=\"380\"                 />\n      </Verts>\n      <Lines>\n        <DataArray type=\"Int64\" Name=\"connectivity\" format=\"appended\" RangeMin=\"\"                     RangeMax=\"\"                     offset=\"424\"                 />\n        <DataArray type=\"Int64\" Name=\"offsets\" format=\"appended\" RangeMin=\"\"                     RangeMax=\"\"                     offset=\"516\"                 />\n      </Lines>\n      <Strips>\n        <DataArray type=\"Int64\" Name=\"connectivity\" format=\"appended\" RangeMin=\"\"                     RangeMax=\"\"                     offset=\"604\"                 />\n        <DataArray type=\"Int64\" Name=\"offsets\" format=\"appended\" RangeMin=\"\"                     RangeMax=\"\"                     offset=\"620\"                 />\n      </Strips>\n      <Polys>\n        <DataArray type=\"Int64\" Name=\"connectivity\" format=\"appended\" RangeMin=\"\"                     RangeMax=\"\"                     offset=\"636\"                 />\n        <DataArray type=\"Int64\" Name=\"offsets\" format=\"appended\" RangeMin=\"\"                     RangeMax=\"\"                     offset=\"652\"                 />\n      </Polys>\n    </Piece>\n  </PolyData>\n  <AppendedData encoding=\"base64\">\n   _AQAAAACAAADkAAAAEQAAAA==eJxjYICBBnuGYcoGABPnDi4=AQAAAACAAACQAAAAEQAAAA==eJxjYoAAZij9f4ABACTof4Y=AQAAAACAAACQAAAAEQAAAA==eJxjYoAAJijNPMA0AA+AADU=AQAAAACAAADkAAAAfAAAAA==eJxj0Npsz8Dw4QADCEDYDkhssLjQ6Xb7T5aXwWxfBhd7we9bwWyN/cz2qzxmgNmRH+fadTaUgNnaBuJ2LvfNwexqSVbbtYu+7gexpULE9oa9LgWzpz0o2cvA4KCOJG6PpN4eyRwHJPMdkOx1QHKPA5I7kd0PZgMAFrY9dw==AQAAAACAAAAQAAAADgAAAA==eJxjYIAARigNAAAYAAI=AQAAAACAAAAQAAAADgAAAA==eJxjZIAAJigNAAAwAAQ=AQAAAACAAAAAAQAAMQAAAA==eJxdxbkRgCAAADAPRZFHwP2HtcAqaRK2ZeeDI598ceKbMxeu3PjhzoMnv/8fe0ABQQ==AQAAAACAAACAAAAAMAAAAA==eJwtxdEGwCAAAMBMMkmSzEyy///KHrp7uSsc0cm3s4urm7uHH7/+PL38ewMzgAERAAAAAACAAAAAAAAAAAAAAACAAAAAAAAAAAAAAACAAAAAAAAAAAAAAACAAAAAAAAA\n  </AppendedData>\n</VTKFile>\n", "color": [1, 0.8, 0, 1], "position": [0, 0, 0], "orientation": [0, 0, 0]}];
    const ratio = 0.5;

    render(data, div_id, ratio);
});
</script>



<div id=shape-d8a4e5c4></div><script>function render(data, div_id, ratio){

    // Initial setup
    const renderWindow = vtk.Rendering.Core.vtkRenderWindow.newInstance();
    const renderer = vtk.Rendering.Core.vtkRenderer.newInstance({ background: [1, 1, 1 ] });
    renderWindow.addRenderer(renderer);

    // iterate over all children children
    for (var el of data){
        var trans = el.position;
        var rot = el.orientation;
        var rgba = el.color;
        var shape = el.shape;

        // load the inline data
        var reader = vtk.IO.XML.vtkXMLPolyDataReader.newInstance();
        const textEncoder = new TextEncoder();
        reader.parseAsArrayBuffer(textEncoder.encode(shape));

        // setup actor,mapper and add
        const mapper = vtk.Rendering.Core.vtkMapper.newInstance();
        mapper.setInputConnection(reader.getOutputPort());
        mapper.setResolveCoincidentTopologyToPolygonOffset();
        mapper.setResolveCoincidentTopologyPolygonOffsetParameters(0.5,100);

        const actor = vtk.Rendering.Core.vtkActor.newInstance();
        actor.setMapper(mapper);

        // set color and position
        actor.getProperty().setColor(rgba.slice(0,3));
        actor.getProperty().setOpacity(rgba[3]);

        actor.rotateZ(rot[2]*180/Math.PI);
        actor.rotateY(rot[1]*180/Math.PI);
        actor.rotateX(rot[0]*180/Math.PI);

        actor.setPosition(trans);

        renderer.addActor(actor);

    };

    renderer.resetCamera();

    const openglRenderWindow = vtk.Rendering.OpenGL.vtkRenderWindow.newInstance();
    renderWindow.addView(openglRenderWindow);

    // Get the div container    
    const container = document.getElementById(div_id);
    const dims = container.parentElement.getBoundingClientRect();

    openglRenderWindow.setContainer(container);

    // handle size
    if (ratio){
        openglRenderWindow.setSize(dims.width, dims.width*ratio);
    }else{
        openglRenderWindow.setSize(dims.width, dims.height);
    };

    // Interaction setup
    const interact_style = vtk.Interaction.Style.vtkInteractorStyleManipulator.newInstance();

    const manips = {
        rot: vtk.Interaction.Manipulators.vtkMouseCameraTrackballRotateManipulator.newInstance(),
        pan: vtk.Interaction.Manipulators.vtkMouseCameraTrackballPanManipulator.newInstance(),
        zoom1: vtk.Interaction.Manipulators.vtkMouseCameraTrackballZoomManipulator.newInstance(),
        zoom2: vtk.Interaction.Manipulators.vtkMouseCameraTrackballZoomManipulator.newInstance(),
        roll: vtk.Interaction.Manipulators.vtkMouseCameraTrackballRollManipulator.newInstance(),
    };

    manips.zoom1.setControl(true);
    manips.zoom2.setScrollEnabled(true);
    manips.roll.setShift(true);
    manips.pan.setButton(2);

    for (var k in manips){
        interact_style.addMouseManipulator(manips[k]);
    };

    const interactor = vtk.Rendering.Core.vtkRenderWindowInteractor.newInstance();
    interactor.setView(openglRenderWindow);
    interactor.initialize();
    interactor.bindEvents(container);
    interactor.setInteractorStyle(interact_style);

    // Orientation marker

    const axes = vtk.Rendering.Core.vtkAnnotatedCubeActor.newInstance();
    axes.setXPlusFaceProperty({text: '+X'});
    axes.setXMinusFaceProperty({text: '-X'});
    axes.setYPlusFaceProperty({text: '+Y'});
    axes.setYMinusFaceProperty({text: '-Y'});
    axes.setZPlusFaceProperty({text: '+Z'});
    axes.setZMinusFaceProperty({text: '-Z'});

    const orientationWidget = vtk.Interaction.Widgets.vtkOrientationMarkerWidget.newInstance({
        actor: axes,
        interactor: interactor });
    orientationWidget.setEnabled(true);
    orientationWidget.setViewportCorner(vtk.Interaction.Widgets.vtkOrientationMarkerWidget.Corners.BOTTOM_LEFT);
    orientationWidget.setViewportSize(0.2);

};


new Promise(
  function(resolve, reject)
  {
    if (typeof(require) !== "undefined" ){
        require.config({
         "paths": {"vtk": "https://unpkg.com/vtk"},
        });
        require(["vtk"], resolve, reject);
    } else if ( typeof(vtk) === "undefined" ){
        var script = document.createElement("script");
    	script.onload = resolve;
    	script.onerror = reject;
    	script.src = "https://unpkg.com/vtk.js";
    	document.head.appendChild(script);
    } else { resolve() };
 }
).then(() => {
    // data, div_id and ratio are templated by python
    const div_id = "shape-d8a4e5c4";
    const data = [{"shape": "<?xml version=\"1.0\"?>\n<VTKFile type=\"PolyData\" version=\"0.1\" byte_order=\"LittleEndian\" header_type=\"UInt32\" compressor=\"vtkZLibDataCompressor\">\n  <PolyData>\n    <Piece NumberOfPoints=\"19\"                   NumberOfVerts=\"2\"                    NumberOfLines=\"16\"                   NumberOfStrips=\"0\"                    NumberOfPolys=\"0\"                   >\n      <PointData Normals=\"Normals\">\n        <DataArray type=\"Float32\" Name=\"Normals\" NumberOfComponents=\"3\" format=\"appended\" RangeMin=\"1\"                    RangeMax=\"1\"                    offset=\"0\"                   />\n      </PointData>\n      <CellData>\n        <DataArray type=\"Int64\" IdType=\"1\" Name=\"SUBSHAPE_IDS\" format=\"appended\" RangeMin=\"-1\"                   RangeMax=\"3\"                    offset=\"48\"                  />\n        <DataArray type=\"Int64\" IdType=\"1\" Name=\"MESH_TYPES\" format=\"appended\" RangeMin=\"2\"                    RangeMax=\"3\"                    offset=\"96\"                  />\n      </CellData>\n      <Points>\n        <DataArray type=\"Float32\" Name=\"Points\" NumberOfComponents=\"3\" format=\"appended\" RangeMin=\"0.05978449434\"        RangeMax=\"7.6294963101\"         offset=\"144\"                 />\n      </Points>\n      <Verts>\n        <DataArray type=\"Int64\" Name=\"connectivity\" format=\"appended\" RangeMin=\"\"                     RangeMax=\"\"                     offset=\"336\"                 />\n        <DataArray type=\"Int64\" Name=\"offsets\" format=\"appended\" RangeMin=\"\"                     RangeMax=\"\"                     offset=\"380\"                 />\n      </Verts>\n      <Lines>\n        <DataArray type=\"Int64\" Name=\"connectivity\" format=\"appended\" RangeMin=\"\"                     RangeMax=\"\"                     offset=\"424\"                 />\n        <DataArray type=\"Int64\" Name=\"offsets\" format=\"appended\" RangeMin=\"\"                     RangeMax=\"\"                     offset=\"516\"                 />\n      </Lines>\n      <Strips>\n        <DataArray type=\"Int64\" Name=\"connectivity\" format=\"appended\" RangeMin=\"\"                     RangeMax=\"\"                     offset=\"604\"                 />\n        <DataArray type=\"Int64\" Name=\"offsets\" format=\"appended\" RangeMin=\"\"                     RangeMax=\"\"                     offset=\"620\"                 />\n      </Strips>\n      <Polys>\n        <DataArray type=\"Int64\" Name=\"connectivity\" format=\"appended\" RangeMin=\"\"                     RangeMax=\"\"                     offset=\"636\"                 />\n        <DataArray type=\"Int64\" Name=\"offsets\" format=\"appended\" RangeMin=\"\"                     RangeMax=\"\"                     offset=\"652\"                 />\n      </Polys>\n    </Piece>\n  </PolyData>\n  <AppendedData encoding=\"base64\">\n   _AQAAAACAAADkAAAAEQAAAA==eJxjYICBBnuGYcoGABPnDi4=AQAAAACAAACQAAAAEQAAAA==eJxjYoAAZij9f4ABACTof4Y=AQAAAACAAACQAAAAEQAAAA==eJxjYoAAJijNPMA0AA+AADU=AQAAAACAAADkAAAAfQAAAA==eJxj0Nq8n4HhgwMDCEDYB5DYYHGh0+37P1leBrN9GVz2C37fCmZr7Gfev8pjBpgd+XHuvs6GEjBb20B8n8t9czC7WpJ179pFX+1BbKkQMduw16Vg9rQHJbYMDA7LkcT3I6nfj2TOASTzDyDZewDJPQeQ3InsfjAbAK/FRHc=AQAAAACAAAAQAAAADgAAAA==eJxjYIAARigNAAAYAAI=AQAAAACAAAAQAAAADgAAAA==eJxjZIAAJigNAAAwAAQ=AQAAAACAAAAAAQAAMQAAAA==eJxdxbkRgCAAADAPRZFHwP2HtcAqaRK2ZeeDI598ceKbMxeu3PjhzoMnv/8fe0ABQQ==AQAAAACAAACAAAAAMAAAAA==eJwtxdEGwCAAAMBMMkmSzEyy///KHrp7uSsc0cm3s4urm7uHH7/+PL38ewMzgAERAAAAAACAAAAAAAAAAAAAAACAAAAAAAAAAAAAAACAAAAAAAAAAAAAAACAAAAAAAAA\n  </AppendedData>\n</VTKFile>\n", "color": [1, 0.8, 0, 1], "position": [0, 0, 0], "orientation": [0, 0, 0]}];
    const ratio = 0.5;

    render(data, div_id, ratio);
});
</script>



<div id=shape-6c1f24dc></div><script>function render(data, div_id, ratio){

    // Initial setup
    const renderWindow = vtk.Rendering.Core.vtkRenderWindow.newInstance();
    const renderer = vtk.Rendering.Core.vtkRenderer.newInstance({ background: [1, 1, 1 ] });
    renderWindow.addRenderer(renderer);

    // iterate over all children children
    for (var el of data){
        var trans = el.position;
        var rot = el.orientation;
        var rgba = el.color;
        var shape = el.shape;

        // load the inline data
        var reader = vtk.IO.XML.vtkXMLPolyDataReader.newInstance();
        const textEncoder = new TextEncoder();
        reader.parseAsArrayBuffer(textEncoder.encode(shape));

        // setup actor,mapper and add
        const mapper = vtk.Rendering.Core.vtkMapper.newInstance();
        mapper.setInputConnection(reader.getOutputPort());
        mapper.setResolveCoincidentTopologyToPolygonOffset();
        mapper.setResolveCoincidentTopologyPolygonOffsetParameters(0.5,100);

        const actor = vtk.Rendering.Core.vtkActor.newInstance();
        actor.setMapper(mapper);

        // set color and position
        actor.getProperty().setColor(rgba.slice(0,3));
        actor.getProperty().setOpacity(rgba[3]);

        actor.rotateZ(rot[2]*180/Math.PI);
        actor.rotateY(rot[1]*180/Math.PI);
        actor.rotateX(rot[0]*180/Math.PI);

        actor.setPosition(trans);

        renderer.addActor(actor);

    };

    renderer.resetCamera();

    const openglRenderWindow = vtk.Rendering.OpenGL.vtkRenderWindow.newInstance();
    renderWindow.addView(openglRenderWindow);

    // Get the div container    
    const container = document.getElementById(div_id);
    const dims = container.parentElement.getBoundingClientRect();

    openglRenderWindow.setContainer(container);

    // handle size
    if (ratio){
        openglRenderWindow.setSize(dims.width, dims.width*ratio);
    }else{
        openglRenderWindow.setSize(dims.width, dims.height);
    };

    // Interaction setup
    const interact_style = vtk.Interaction.Style.vtkInteractorStyleManipulator.newInstance();

    const manips = {
        rot: vtk.Interaction.Manipulators.vtkMouseCameraTrackballRotateManipulator.newInstance(),
        pan: vtk.Interaction.Manipulators.vtkMouseCameraTrackballPanManipulator.newInstance(),
        zoom1: vtk.Interaction.Manipulators.vtkMouseCameraTrackballZoomManipulator.newInstance(),
        zoom2: vtk.Interaction.Manipulators.vtkMouseCameraTrackballZoomManipulator.newInstance(),
        roll: vtk.Interaction.Manipulators.vtkMouseCameraTrackballRollManipulator.newInstance(),
    };

    manips.zoom1.setControl(true);
    manips.zoom2.setScrollEnabled(true);
    manips.roll.setShift(true);
    manips.pan.setButton(2);

    for (var k in manips){
        interact_style.addMouseManipulator(manips[k]);
    };

    const interactor = vtk.Rendering.Core.vtkRenderWindowInteractor.newInstance();
    interactor.setView(openglRenderWindow);
    interactor.initialize();
    interactor.bindEvents(container);
    interactor.setInteractorStyle(interact_style);

    // Orientation marker

    const axes = vtk.Rendering.Core.vtkAnnotatedCubeActor.newInstance();
    axes.setXPlusFaceProperty({text: '+X'});
    axes.setXMinusFaceProperty({text: '-X'});
    axes.setYPlusFaceProperty({text: '+Y'});
    axes.setYMinusFaceProperty({text: '-Y'});
    axes.setZPlusFaceProperty({text: '+Z'});
    axes.setZMinusFaceProperty({text: '-Z'});

    const orientationWidget = vtk.Interaction.Widgets.vtkOrientationMarkerWidget.newInstance({
        actor: axes,
        interactor: interactor });
    orientationWidget.setEnabled(true);
    orientationWidget.setViewportCorner(vtk.Interaction.Widgets.vtkOrientationMarkerWidget.Corners.BOTTOM_LEFT);
    orientationWidget.setViewportSize(0.2);

};


new Promise(
  function(resolve, reject)
  {
    if (typeof(require) !== "undefined" ){
        require.config({
         "paths": {"vtk": "https://unpkg.com/vtk"},
        });
        require(["vtk"], resolve, reject);
    } else if ( typeof(vtk) === "undefined" ){
        var script = document.createElement("script");
    	script.onload = resolve;
    	script.onerror = reject;
    	script.src = "https://unpkg.com/vtk.js";
    	document.head.appendChild(script);
    } else { resolve() };
 }
).then(() => {
    // data, div_id and ratio are templated by python
    const div_id = "shape-6c1f24dc";
    const data = [{"shape": "<?xml version=\"1.0\"?>\n<VTKFile type=\"PolyData\" version=\"0.1\" byte_order=\"LittleEndian\" header_type=\"UInt32\" compressor=\"vtkZLibDataCompressor\">\n  <PolyData>\n    <Piece NumberOfPoints=\"17\"                   NumberOfVerts=\"2\"                    NumberOfLines=\"14\"                   NumberOfStrips=\"0\"                    NumberOfPolys=\"0\"                   >\n      <PointData Normals=\"Normals\">\n        <DataArray type=\"Float32\" Name=\"Normals\" NumberOfComponents=\"3\" format=\"appended\" RangeMin=\"1\"                    RangeMax=\"1\"                    offset=\"0\"                   />\n      </PointData>\n      <CellData>\n        <DataArray type=\"Int64\" IdType=\"1\" Name=\"SUBSHAPE_IDS\" format=\"appended\" RangeMin=\"-1\"                   RangeMax=\"3\"                    offset=\"48\"                  />\n        <DataArray type=\"Int64\" IdType=\"1\" Name=\"MESH_TYPES\" format=\"appended\" RangeMin=\"2\"                    RangeMax=\"3\"                    offset=\"96\"                  />\n      </CellData>\n      <Points>\n        <DataArray type=\"Float32\" Name=\"Points\" NumberOfComponents=\"3\" format=\"appended\" RangeMin=\"0.11663650418\"        RangeMax=\"7.5839056561\"         offset=\"144\"                 />\n      </Points>\n      <Verts>\n        <DataArray type=\"Int64\" Name=\"connectivity\" format=\"appended\" RangeMin=\"\"                     RangeMax=\"\"                     offset=\"368\"                 />\n        <DataArray type=\"Int64\" Name=\"offsets\" format=\"appended\" RangeMin=\"\"                     RangeMax=\"\"                     offset=\"412\"                 />\n      </Verts>\n      <Lines>\n        <DataArray type=\"Int64\" Name=\"connectivity\" format=\"appended\" RangeMin=\"\"                     RangeMax=\"\"                     offset=\"456\"                 />\n        <DataArray type=\"Int64\" Name=\"offsets\" format=\"appended\" RangeMin=\"\"                     RangeMax=\"\"                     offset=\"540\"                 />\n      </Lines>\n      <Strips>\n        <DataArray type=\"Int64\" Name=\"connectivity\" format=\"appended\" RangeMin=\"\"                     RangeMax=\"\"                     offset=\"624\"                 />\n        <DataArray type=\"Int64\" Name=\"offsets\" format=\"appended\" RangeMin=\"\"                     RangeMax=\"\"                     offset=\"640\"                 />\n      </Strips>\n      <Polys>\n        <DataArray type=\"Int64\" Name=\"connectivity\" format=\"appended\" RangeMin=\"\"                     RangeMax=\"\"                     offset=\"656\"                 />\n        <DataArray type=\"Int64\" Name=\"offsets\" format=\"appended\" RangeMin=\"\"                     RangeMax=\"\"                     offset=\"672\"                 />\n      </Polys>\n    </Piece>\n  </PolyData>\n  <AppendedData encoding=\"base64\">\n   _AQAAAACAAADMAAAAEQAAAA==eJxjYICBBnuGIc4GANfXDLA=AQAAAACAAACAAAAAEQAAAA==eJxjYoAAZij9n84AAKOYb5Y=AQAAAACAAACAAAAAEQAAAA==eJxjYoAAJijNTGcaAAxIAC8=AQAAAACAAADMAAAAlgAAAA==eJxjYJhgz8Dw4QADGIDZDkhssPjxF+72yV+Og9mbt/63O/ZiPpidv6bfLnhXPpjNc/2freRieTD7odRfm33i8/eD2N+W9ll+ufduL4i94kqzzdRFqfYgdl/7VNuWCV/A7JdK+nb31a3B9v47Pc+uhr0GzK4X/G3XUzwPzP5jYGYf/XgfmL1SJMN+beF1ZHeC2QA+vUAiAQAAAACAAAAQAAAADgAAAA==eJxjYIAARigNAAAYAAI=AQAAAACAAAAQAAAADgAAAA==eJxjZIAAJigNAAAwAAQ=AQAAAACAAADgAAAALQAAAA==eJxdxbkRgCAAADBOkFeF/ae1gCppcoUtcuKbMxeu3Ljz4Idf/njyOv9WaAD9AQAAAACAAABwAAAALAAAAA==eJwtxUEKABEAAEBJkiRJkvb/79yDmcvE8CRnF1c3dw9PL28fX3/+ASNwANM=AAAAAACAAAAAAAAAAAAAAACAAAAAAAAAAAAAAACAAAAAAAAAAAAAAACAAAAAAAAA\n  </AppendedData>\n</VTKFile>\n", "color": [1, 0.8, 0, 1], "position": [0, 0, 0], "orientation": [0, 0, 0]}];
    const ratio = 0.5;

    render(data, div_id, ratio);
});
</script>



<div id=shape-a8b05cd4></div><script>function render(data, div_id, ratio){

    // Initial setup
    const renderWindow = vtk.Rendering.Core.vtkRenderWindow.newInstance();
    const renderer = vtk.Rendering.Core.vtkRenderer.newInstance({ background: [1, 1, 1 ] });
    renderWindow.addRenderer(renderer);

    // iterate over all children children
    for (var el of data){
        var trans = el.position;
        var rot = el.orientation;
        var rgba = el.color;
        var shape = el.shape;

        // load the inline data
        var reader = vtk.IO.XML.vtkXMLPolyDataReader.newInstance();
        const textEncoder = new TextEncoder();
        reader.parseAsArrayBuffer(textEncoder.encode(shape));

        // setup actor,mapper and add
        const mapper = vtk.Rendering.Core.vtkMapper.newInstance();
        mapper.setInputConnection(reader.getOutputPort());
        mapper.setResolveCoincidentTopologyToPolygonOffset();
        mapper.setResolveCoincidentTopologyPolygonOffsetParameters(0.5,100);

        const actor = vtk.Rendering.Core.vtkActor.newInstance();
        actor.setMapper(mapper);

        // set color and position
        actor.getProperty().setColor(rgba.slice(0,3));
        actor.getProperty().setOpacity(rgba[3]);

        actor.rotateZ(rot[2]*180/Math.PI);
        actor.rotateY(rot[1]*180/Math.PI);
        actor.rotateX(rot[0]*180/Math.PI);

        actor.setPosition(trans);

        renderer.addActor(actor);

    };

    renderer.resetCamera();

    const openglRenderWindow = vtk.Rendering.OpenGL.vtkRenderWindow.newInstance();
    renderWindow.addView(openglRenderWindow);

    // Get the div container    
    const container = document.getElementById(div_id);
    const dims = container.parentElement.getBoundingClientRect();

    openglRenderWindow.setContainer(container);

    // handle size
    if (ratio){
        openglRenderWindow.setSize(dims.width, dims.width*ratio);
    }else{
        openglRenderWindow.setSize(dims.width, dims.height);
    };

    // Interaction setup
    const interact_style = vtk.Interaction.Style.vtkInteractorStyleManipulator.newInstance();

    const manips = {
        rot: vtk.Interaction.Manipulators.vtkMouseCameraTrackballRotateManipulator.newInstance(),
        pan: vtk.Interaction.Manipulators.vtkMouseCameraTrackballPanManipulator.newInstance(),
        zoom1: vtk.Interaction.Manipulators.vtkMouseCameraTrackballZoomManipulator.newInstance(),
        zoom2: vtk.Interaction.Manipulators.vtkMouseCameraTrackballZoomManipulator.newInstance(),
        roll: vtk.Interaction.Manipulators.vtkMouseCameraTrackballRollManipulator.newInstance(),
    };

    manips.zoom1.setControl(true);
    manips.zoom2.setScrollEnabled(true);
    manips.roll.setShift(true);
    manips.pan.setButton(2);

    for (var k in manips){
        interact_style.addMouseManipulator(manips[k]);
    };

    const interactor = vtk.Rendering.Core.vtkRenderWindowInteractor.newInstance();
    interactor.setView(openglRenderWindow);
    interactor.initialize();
    interactor.bindEvents(container);
    interactor.setInteractorStyle(interact_style);

    // Orientation marker

    const axes = vtk.Rendering.Core.vtkAnnotatedCubeActor.newInstance();
    axes.setXPlusFaceProperty({text: '+X'});
    axes.setXMinusFaceProperty({text: '-X'});
    axes.setYPlusFaceProperty({text: '+Y'});
    axes.setYMinusFaceProperty({text: '-Y'});
    axes.setZPlusFaceProperty({text: '+Z'});
    axes.setZMinusFaceProperty({text: '-Z'});

    const orientationWidget = vtk.Interaction.Widgets.vtkOrientationMarkerWidget.newInstance({
        actor: axes,
        interactor: interactor });
    orientationWidget.setEnabled(true);
    orientationWidget.setViewportCorner(vtk.Interaction.Widgets.vtkOrientationMarkerWidget.Corners.BOTTOM_LEFT);
    orientationWidget.setViewportSize(0.2);

};


new Promise(
  function(resolve, reject)
  {
    if (typeof(require) !== "undefined" ){
        require.config({
         "paths": {"vtk": "https://unpkg.com/vtk"},
        });
        require(["vtk"], resolve, reject);
    } else if ( typeof(vtk) === "undefined" ){
        var script = document.createElement("script");
    	script.onload = resolve;
    	script.onerror = reject;
    	script.src = "https://unpkg.com/vtk.js";
    	document.head.appendChild(script);
    } else { resolve() };
 }
).then(() => {
    // data, div_id and ratio are templated by python
    const div_id = "shape-a8b05cd4";
    const data = [{"shape": "<?xml version=\"1.0\"?>\n<VTKFile type=\"PolyData\" version=\"0.1\" byte_order=\"LittleEndian\" header_type=\"UInt32\" compressor=\"vtkZLibDataCompressor\">\n  <PolyData>\n    <Piece NumberOfPoints=\"17\"                   NumberOfVerts=\"2\"                    NumberOfLines=\"14\"                   NumberOfStrips=\"0\"                    NumberOfPolys=\"0\"                   >\n      <PointData Normals=\"Normals\">\n        <DataArray type=\"Float32\" Name=\"Normals\" NumberOfComponents=\"3\" format=\"appended\" RangeMin=\"1\"                    RangeMax=\"1\"                    offset=\"0\"                   />\n      </PointData>\n      <CellData>\n        <DataArray type=\"Int64\" IdType=\"1\" Name=\"SUBSHAPE_IDS\" format=\"appended\" RangeMin=\"-1\"                   RangeMax=\"3\"                    offset=\"48\"                  />\n        <DataArray type=\"Int64\" IdType=\"1\" Name=\"MESH_TYPES\" format=\"appended\" RangeMin=\"2\"                    RangeMax=\"3\"                    offset=\"96\"                  />\n      </CellData>\n      <Points>\n        <DataArray type=\"Float32\" Name=\"Points\" NumberOfComponents=\"3\" format=\"appended\" RangeMin=\"0.11663650418\"        RangeMax=\"7.5839056561\"         offset=\"144\"                 />\n      </Points>\n      <Verts>\n        <DataArray type=\"Int64\" Name=\"connectivity\" format=\"appended\" RangeMin=\"\"                     RangeMax=\"\"                     offset=\"372\"                 />\n        <DataArray type=\"Int64\" Name=\"offsets\" format=\"appended\" RangeMin=\"\"                     RangeMax=\"\"                     offset=\"416\"                 />\n      </Verts>\n      <Lines>\n        <DataArray type=\"Int64\" Name=\"connectivity\" format=\"appended\" RangeMin=\"\"                     RangeMax=\"\"                     offset=\"460\"                 />\n        <DataArray type=\"Int64\" Name=\"offsets\" format=\"appended\" RangeMin=\"\"                     RangeMax=\"\"                     offset=\"544\"                 />\n      </Lines>\n      <Strips>\n        <DataArray type=\"Int64\" Name=\"connectivity\" format=\"appended\" RangeMin=\"\"                     RangeMax=\"\"                     offset=\"628\"                 />\n        <DataArray type=\"Int64\" Name=\"offsets\" format=\"appended\" RangeMin=\"\"                     RangeMax=\"\"                     offset=\"644\"                 />\n      </Strips>\n      <Polys>\n        <DataArray type=\"Int64\" Name=\"connectivity\" format=\"appended\" RangeMin=\"\"                     RangeMax=\"\"                     offset=\"660\"                 />\n        <DataArray type=\"Int64\" Name=\"offsets\" format=\"appended\" RangeMin=\"\"                     RangeMax=\"\"                     offset=\"676\"                 />\n      </Polys>\n    </Piece>\n  </PolyData>\n  <AppendedData encoding=\"base64\">\n   _AQAAAACAAADMAAAAEQAAAA==eJxjYICBBnuGIc4GANfXDLA=AQAAAACAAACAAAAAEQAAAA==eJxjYoAAZij9n84AAKOYb5Y=AQAAAACAAACAAAAAEQAAAA==eJxjYoAAJijNTGcaAAxIAC8=AQAAAACAAADMAAAAmAAAAA==eJxjYJiwn4HhwwEGMACzHZDYYPHjL9z3J385DmZv3vp/37EX88Hs/DX9+4J35YPZPNf/7ZVcLA9mP5T6u2ef+Pz9IPa3pX07v9x7txfEXnGlec/URan2IHZf+9S9LRO+gNkvlfT33Ve3Btv77/S8fTXsNWB2veDvfT3F88DsPwZm+6Mf7wOzV4pk7F9beB3ZnWA2ALtqSKI=AQAAAACAAAAQAAAADgAAAA==eJxjYIAARigNAAAYAAI=AQAAAACAAAAQAAAADgAAAA==eJxjZIAAJigNAAAwAAQ=AQAAAACAAADgAAAALQAAAA==eJxdxbkRgCAAADBOkFeF/ae1gCppcoUtcuKbMxeu3Ljz4Idf/njyOv9WaAD9AQAAAACAAABwAAAALAAAAA==eJwtxUEKABEAAEBJkiRJkvb/79yDmcvE8CRnF1c3dw9PL28fX3/+ASNwANM=AAAAAACAAAAAAAAAAAAAAACAAAAAAAAAAAAAAACAAAAAAAAAAAAAAACAAAAAAAAA\n  </AppendedData>\n</VTKFile>\n", "color": [1, 0.8, 0, 1], "position": [0, 0, 0], "orientation": [0, 0, 0]}];
    const ratio = 0.5;

    render(data, div_id, ratio);
});
</script>



<div id=shape-5a23a924></div><script>function render(data, div_id, ratio){

    // Initial setup
    const renderWindow = vtk.Rendering.Core.vtkRenderWindow.newInstance();
    const renderer = vtk.Rendering.Core.vtkRenderer.newInstance({ background: [1, 1, 1 ] });
    renderWindow.addRenderer(renderer);

    // iterate over all children children
    for (var el of data){
        var trans = el.position;
        var rot = el.orientation;
        var rgba = el.color;
        var shape = el.shape;

        // load the inline data
        var reader = vtk.IO.XML.vtkXMLPolyDataReader.newInstance();
        const textEncoder = new TextEncoder();
        reader.parseAsArrayBuffer(textEncoder.encode(shape));

        // setup actor,mapper and add
        const mapper = vtk.Rendering.Core.vtkMapper.newInstance();
        mapper.setInputConnection(reader.getOutputPort());
        mapper.setResolveCoincidentTopologyToPolygonOffset();
        mapper.setResolveCoincidentTopologyPolygonOffsetParameters(0.5,100);

        const actor = vtk.Rendering.Core.vtkActor.newInstance();
        actor.setMapper(mapper);

        // set color and position
        actor.getProperty().setColor(rgba.slice(0,3));
        actor.getProperty().setOpacity(rgba[3]);

        actor.rotateZ(rot[2]*180/Math.PI);
        actor.rotateY(rot[1]*180/Math.PI);
        actor.rotateX(rot[0]*180/Math.PI);

        actor.setPosition(trans);

        renderer.addActor(actor);

    };

    renderer.resetCamera();

    const openglRenderWindow = vtk.Rendering.OpenGL.vtkRenderWindow.newInstance();
    renderWindow.addView(openglRenderWindow);

    // Get the div container    
    const container = document.getElementById(div_id);
    const dims = container.parentElement.getBoundingClientRect();

    openglRenderWindow.setContainer(container);

    // handle size
    if (ratio){
        openglRenderWindow.setSize(dims.width, dims.width*ratio);
    }else{
        openglRenderWindow.setSize(dims.width, dims.height);
    };

    // Interaction setup
    const interact_style = vtk.Interaction.Style.vtkInteractorStyleManipulator.newInstance();

    const manips = {
        rot: vtk.Interaction.Manipulators.vtkMouseCameraTrackballRotateManipulator.newInstance(),
        pan: vtk.Interaction.Manipulators.vtkMouseCameraTrackballPanManipulator.newInstance(),
        zoom1: vtk.Interaction.Manipulators.vtkMouseCameraTrackballZoomManipulator.newInstance(),
        zoom2: vtk.Interaction.Manipulators.vtkMouseCameraTrackballZoomManipulator.newInstance(),
        roll: vtk.Interaction.Manipulators.vtkMouseCameraTrackballRollManipulator.newInstance(),
    };

    manips.zoom1.setControl(true);
    manips.zoom2.setScrollEnabled(true);
    manips.roll.setShift(true);
    manips.pan.setButton(2);

    for (var k in manips){
        interact_style.addMouseManipulator(manips[k]);
    };

    const interactor = vtk.Rendering.Core.vtkRenderWindowInteractor.newInstance();
    interactor.setView(openglRenderWindow);
    interactor.initialize();
    interactor.bindEvents(container);
    interactor.setInteractorStyle(interact_style);

    // Orientation marker

    const axes = vtk.Rendering.Core.vtkAnnotatedCubeActor.newInstance();
    axes.setXPlusFaceProperty({text: '+X'});
    axes.setXMinusFaceProperty({text: '-X'});
    axes.setYPlusFaceProperty({text: '+Y'});
    axes.setYMinusFaceProperty({text: '-Y'});
    axes.setZPlusFaceProperty({text: '+Z'});
    axes.setZMinusFaceProperty({text: '-Z'});

    const orientationWidget = vtk.Interaction.Widgets.vtkOrientationMarkerWidget.newInstance({
        actor: axes,
        interactor: interactor });
    orientationWidget.setEnabled(true);
    orientationWidget.setViewportCorner(vtk.Interaction.Widgets.vtkOrientationMarkerWidget.Corners.BOTTOM_LEFT);
    orientationWidget.setViewportSize(0.2);

};


new Promise(
  function(resolve, reject)
  {
    if (typeof(require) !== "undefined" ){
        require.config({
         "paths": {"vtk": "https://unpkg.com/vtk"},
        });
        require(["vtk"], resolve, reject);
    } else if ( typeof(vtk) === "undefined" ){
        var script = document.createElement("script");
    	script.onload = resolve;
    	script.onerror = reject;
    	script.src = "https://unpkg.com/vtk.js";
    	document.head.appendChild(script);
    } else { resolve() };
 }
).then(() => {
    // data, div_id and ratio are templated by python
    const div_id = "shape-5a23a924";
    const data = [{"shape": "<?xml version=\"1.0\"?>\n<VTKFile type=\"PolyData\" version=\"0.1\" byte_order=\"LittleEndian\" header_type=\"UInt32\" compressor=\"vtkZLibDataCompressor\">\n  <PolyData>\n    <Piece NumberOfPoints=\"4\"                    NumberOfVerts=\"2\"                    NumberOfLines=\"1\"                    NumberOfStrips=\"0\"                    NumberOfPolys=\"0\"                   >\n      <PointData Normals=\"Normals\">\n        <DataArray type=\"Float32\" Name=\"Normals\" NumberOfComponents=\"3\" format=\"appended\" RangeMin=\"1\"                    RangeMax=\"1\"                    offset=\"0\"                   />\n      </PointData>\n      <CellData>\n        <DataArray type=\"Int64\" IdType=\"1\" Name=\"SUBSHAPE_IDS\" format=\"appended\" RangeMin=\"-1\"                   RangeMax=\"3\"                    offset=\"48\"                  />\n        <DataArray type=\"Int64\" IdType=\"1\" Name=\"MESH_TYPES\" format=\"appended\" RangeMin=\"2\"                    RangeMax=\"3\"                    offset=\"96\"                  />\n      </CellData>\n      <Points>\n        <DataArray type=\"Float32\" Name=\"Points\" NumberOfComponents=\"3\" format=\"appended\" RangeMin=\"7.5\"                  RangeMax=\"7.5\"                  offset=\"144\"                 />\n      </Points>\n      <Verts>\n        <DataArray type=\"Int64\" Name=\"connectivity\" format=\"appended\" RangeMin=\"\"                     RangeMax=\"\"                     offset=\"200\"                 />\n        <DataArray type=\"Int64\" Name=\"offsets\" format=\"appended\" RangeMin=\"\"                     RangeMax=\"\"                     offset=\"244\"                 />\n      </Verts>\n      <Lines>\n        <DataArray type=\"Int64\" Name=\"connectivity\" format=\"appended\" RangeMin=\"\"                     RangeMax=\"\"                     offset=\"288\"                 />\n        <DataArray type=\"Int64\" Name=\"offsets\" format=\"appended\" RangeMin=\"\"                     RangeMax=\"\"                     offset=\"332\"                 />\n      </Lines>\n      <Strips>\n        <DataArray type=\"Int64\" Name=\"connectivity\" format=\"appended\" RangeMin=\"\"                     RangeMax=\"\"                     offset=\"372\"                 />\n        <DataArray type=\"Int64\" Name=\"offsets\" format=\"appended\" RangeMin=\"\"                     RangeMax=\"\"                     offset=\"388\"                 />\n      </Strips>\n      <Polys>\n        <DataArray type=\"Int64\" Name=\"connectivity\" format=\"appended\" RangeMin=\"\"                     RangeMax=\"\"                     offset=\"404\"                 />\n        <DataArray type=\"Int64\" Name=\"offsets\" format=\"appended\" RangeMin=\"\"                     RangeMax=\"\"                     offset=\"420\"                 />\n      </Polys>\n    </Piece>\n  </PolyData>\n  <AppendedData encoding=\"base64\">\n   _AQAAAACAAAAwAAAAEAAAAA==eJxjYICBBnsGItgAOuQC/Q==AQAAAACAAAAYAAAAEAAAAA==eJxjYoAAZij9HwoAJFQH/g==AQAAAACAAAAYAAAAEAAAAA==eJxjYoAAJijNDKUBAIAACA==AQAAAACAAAAwAAAAFgAAAA==eJxjYACBDwcY4OCDAxIbQxwAjjAFwQ==AQAAAACAAAAQAAAADgAAAA==eJxjYIAARigNAAAYAAI=AQAAAACAAAAQAAAADgAAAA==eJxjZIAAJigNAAAwAAQ=AQAAAACAAAAQAAAADgAAAA==eJxjYoAAZigNAABIAAY=AQAAAACAAAAIAAAACwAAAA==eJxjYoAAAAAYAAM=AAAAAACAAAAAAAAAAAAAAACAAAAAAAAAAAAAAACAAAAAAAAAAAAAAACAAAAAAAAA\n  </AppendedData>\n</VTKFile>\n", "color": [1, 0.8, 0, 1], "position": [0, 0, 0], "orientation": [0, 0, 0]}];
    const ratio = 0.5;

    render(data, div_id, ratio);
});
</script>

