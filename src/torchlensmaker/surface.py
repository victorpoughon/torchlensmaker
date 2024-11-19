import torch


class Surface:
    """
    A surface places a shape in absolute 2D space
    
    The surface class wraps a shape with:
        * 'pos': an absolute position in 2D space
        * 'anchor': relative position on the shape that attaches to the absolute position

    Valid anchors are:
        * 'origin' (default): origin (0,0) of the shape
        * 'extent': point on the Y axis that aligns with the shape's greatest extent
    """

    valid_anchors = ["origin", "extent"]

    def __init__(self, shape, pos, anchor="origin"):
        self.shape = shape
        
        self.pos = torch.as_tensor(pos)
        self.anchor = anchor

        if not anchor in self.valid_anchors:
            raise ValueError(f"Invalid anchor value '{self.anchor}', must be one of {self.valid_anchors}")

    def parameters(self):
        return self.shape.parameters()

    def domain(self):
        return self.shape.domain()

    def anchor_offset(self, anchor):
        "Relative position of the given anchor"

        if anchor == "origin":
            return torch.tensor([0., 0.])

        elif anchor == "extent":
            # Assuming the shape is symmetric, get the extend along the Y axis
            off = self.shape.evaluate(self.shape.domain()[1:])[0][1]
            return torch.tensor([0., off])

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
        relative_points = self.shape.evaluate(ts)
        return relative_points + self.to_abs()

    def normal(self, ts):
        return self.shape.normal(ts)

    def collide(self, absolute_lines):
        "Collide with lines expressed in absolute space"

        # Convert lines to relative space
        relative_lines = []
        a, b, c = absolute_lines[:, 0], absolute_lines[:, 1], absolute_lines[:, 2]
        P = self.to_abs()
        new_c = c + a*P[0] + b*P[1]
        relative_lines = torch.column_stack((a, b, new_c))

        # Collide
        relative_points, normals = self.shape.collide(relative_lines)

        # Convert relative points back to absolute space
        return relative_points + P, normals
