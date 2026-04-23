import tlmviewer from "tlmviewer";

const { parseSagFunction, glslRender } = tlmviewer.testing;

// const str = `{
//                 "sag-type": "sum",
//                 "terms": [
//                     {
//                         "sag-type": "spherical",
//                         "C": 1.0
//                     },
//                     {
//                         "sag-type": "aspheric",
//                         "coefficients": [
//                             0.5,
//                             1.0
//                         ]
//                     }
//                 ]
//             }`;

// const str = `{"sag-type": "spherical", "C": 1.0}`;

// const str = `{"sag-type": "sum", "terms": [{"sag-type": "conical", "K": -1.600000023841858, "C": -0.06666667014360428}, {"sag-type": "aspheric", "coefficients": [0.00011999999696854502]}]}`;

const str = `{
                "sag-type": "xypolynomial",
                "coefficients": [
                    [
                        0,
                        0,
                        0,
                        1
                    ],
                    [
                        1,
                        1,
                        0,
                        0
                    ],
                    [
                        0,
                        0,
                        1,
                        0
                    ]
                ]
            }`;

const tau = 1.0;
const sag = parseSagFunction(JSON.parse(str), tau);

const vertexShader = glslRender(
    sag.shaderG(tau),
    sag.shaderGgrad(tau),
    sag.name,
);

console.log(vertexShader);

// shader3D returns a list of functions to declare
// and the current sag function shader function name (entry point)

// top level concatenates all function bodies in the right order
// calls top level entry point
