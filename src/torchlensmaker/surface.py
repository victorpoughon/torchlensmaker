import torch


class Surface:
    """
    A surface places a shape in absolute 2D space
    
    The surface class wraps a shape with:
        - 'pos' position, the position of the shape in 2D space
        - 'anchor' that tells which point of the shape is attached to its anchor in 2D space

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

    def domain(self):
        return self.shape.domain()

    def relative_anchor(self):
        if self.anchor == "origin":
            return torch.tensor([0., 0.])

        elif self.anchor == "extent":
            # Assuming the shape is symmetric, get the extend along the Y axis
            off = self.shape.evaluate(self.shape.domain()[1:])[0][1]
            return torch.tensor([0., off])

        else:
            raise ValueError(f"Invalid anchor value '{self.anchor}'")

    def to_abs(self):
        "The offset that needs to be added to a relative point to convert it to absolute space"

        return self.pos - self.relative_anchor()

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
