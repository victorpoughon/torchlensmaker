


class BaseShape:
    """
    Base class for lens surface profile shapes
    """

    def __init__(self):
        pass

    def domain(self):
        raise NotImplementedError
    
    def evaluate(self, ts):
        raise NotImplementedError
    
    def normal(self, ts):
        raise NotImplementedError

    def collide(self, ts):
        raise NotImplementedError
