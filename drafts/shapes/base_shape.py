class BaseShape:
    """
    Base class for parametric 2D shapes,
    used to represent the surface profile of a lens.
    """

    def __init__(self):
        pass
    
    def coefficients(self):
        raise NotImplementedError

    def parameters(self):
        "Dictionary of name -> nn.Parameter"
        raise NotImplementedError

    def domain(self):
        raise NotImplementedError
    
    def evaluate(self, ts):
        raise NotImplementedError
    
    def derivative(self, ts):
        raise NotImplementedError
    
    def normal(self, ts):
        "Normal vectors at the given parametric locations"
        raise NotImplementedError

    def collide(self, lines):
        raise NotImplementedError
