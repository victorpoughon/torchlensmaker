


class BaseShape:
    """
    Base class for parametric 2D shapes,
    used to represent the surface profile of a lens.
    """

    def __init__(self):
        pass

    def domain(self):
        raise NotImplementedError
    
    def evaluate(self, ts):
        raise NotImplementedError
    
    # derivative
    
    def normal(self, ts):
        raise NotImplementedError

    def collide(self, lines):
        raise NotImplementedError
