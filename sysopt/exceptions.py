class DanglingInputError(ValueError):
    """Raised when an external input is not forwarded"""
    pass


class UnconnectedInputError(ValueError):
    """Raised when a subcomponent doesn't have data feeding it."""    
    def __init__(self, block, child_ports):
        message = f"Composite block {block} has components with unconnected " \
                  f"input channels: {child_ports}"
        super().__init__(message)


class InvalidWire(ValueError):
    """Raised when a wire is not defined."""
    def __init__(self, src, dest, reason):
        message = f'Could not create a wire from {src} to {dest}: {reason}'
        super().__init__(message)


class InvalidComponentError(ValueError):
    pass


class UnconnectedOutputError(ValueError):
    pass
