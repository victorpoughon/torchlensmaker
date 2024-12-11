import torch


def scale_lines(lines, scale):
    a, b, c = lines[:, 0], lines[:, 1], lines[:, 2]
    return torch.column_stack((
        a * scale[1],
        b * scale[0],
        c * scale[0] * scale[1],
    ))


class Surface:
    """
    A surface places a shape in absolute 2D space.

    Shapes are defined in a local 2D coordinate system where the origin (0,0) is
    the center point of the shape. The travel direction of light rays
    is generally assumed to be along the positive X axis.

    A surface (this class), takes a shape and places it in absolute 2D space by applying:
    * A scale factor along the X axis. Typically -1 is used to make symmetric lenses
    * A translation to a point in absolute space
    
    The surface class wraps a shape with:
        * 'pos': an absolute position in 2D space
        * 'anchor': relative position on the shape that attaches to the absolute position
        * 'scale': an optional scale parameter

    Valid anchors are:
        * 'origin' (default): origin (0,0) of the shape
        * 'extent': point on the X axis that aligns with the shape's greatest extent
    """

    valid_anchors = ["origin", "extent"]

    def __init__(self, shape, pos, scale=1., anchor="origin"):
        self.shape = shape
        
        self.pos = torch.as_tensor(pos)
        self.scale = torch.stack((torch.as_tensor(scale, dtype=torch.float32), torch.tensor(1.)))
        self.anchor = anchor

        if not anchor in self.valid_anchors:
            raise ValueError(f"Invalid anchor value '{self.anchor}', must be one of {self.valid_anchors}")

    def coefficients(self):
        return self.shape.coefficients()

    def parameters(self):
        return self.shape.parameters()

    def domain(self):
        return self.shape.domain()

    def anchor_offset(self, anchor):
        "Relative position of the given anchor"

        if anchor == "origin":
            return torch.tensor([0., 0.])

        elif anchor == "extent":
            # Assuming the shape is symmetric, get the extent along the X axis
            off = self.shape.evaluate(self.shape.domain()[1:])[0] * self.scale
            return torch.stack((off[0], torch.tensor(0.)))

        else:
            raise ValueError(f"Invalid anchor value '{anchor}'")
    
    def at(self, anchor):
        "Absolute position of the given shape anchor"

        # get absolute pos of origin point, then add relative position of given anchor
        return self.to_abs() + self.anchor_offset(anchor)

    def to_abs(self):
        "The offset that needs to be added to a relative point to convert it to absolute space"

        return self.pos - self.anchor_offset(self.anchor)

    def evaluate(self, ts):
        "Convert the inner shape evaluate() to absolute space"
        relative_points = self.shape.evaluate(ts) * self.scale
        return relative_points + self.to_abs()

    def normal(self, ts):
        return self.shape.normal(ts) * self.scale

    def collide(self, absolute_lines):
        "Collide with lines expressed in absolute space"

        # Convert lines to relative space
        relative_lines = []
        a, b, c = absolute_lines[:, 0], absolute_lines[:, 1], absolute_lines[:, 2]
        P = self.to_abs()
        new_c = c + a*P[0] + b*P[1]
        relative_lines = torch.column_stack((a, b, new_c))

        # Apply inverse scale
        relative_lines = scale_lines(relative_lines, 1. / self.scale)

        # Collide
        return self.shape.collide(relative_lines)
