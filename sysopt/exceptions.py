class DanglingInputError(ValueError):
    """Raised when an external input is not forwarded"""
    pass


class UnconnectedInputError(ValueError):
    """Raised when a subcomponent doesn't have data feeding it."""    
    pass


class InvalidWire(ValueError):
    """Raised when a wire is not well defined."""
    pass


