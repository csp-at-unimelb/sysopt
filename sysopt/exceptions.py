class DanglingInputError(ValueError):
    """Raised when an external input is not forwarded"""
    def __init__(self, block, channels):
        message = f'{block} has external inputs that are not connected to' \
                  f'internal components: {channels}'

        super().__init__(message)


class UnconnectedInputError(ValueError):
    """Raised when a subcomponent doesn't have data feeding it."""    
    def __init__(self, block, child_ports):
        message = f"Composite block {block} has child components with " \
                  f"unconnected input channels: {child_ports}"
        super().__init__(message)


class InvalidWire(ValueError):
    """Raised when a wire is not defined."""
    def __init__(self, src, dest, reason):
        message = f'Could not create a wire from {src} to {dest}: {reason}'
        super().__init__(message)


class InvalidComponentError(ValueError):
    pass


class UnconnectedOutputError(ValueError):
    def __init__(self, component, channel):
        message = f'{component} has output ports without no sources:'
        message += ','.join(list(str(c) for c in channel))
        super().__init__(message)



class InvalidPort(ValueError):
    pass
