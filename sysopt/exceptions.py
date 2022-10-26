"""Exceptions and Error reporting objects."""
from typing import Tuple


class DanglingInputError(ValueError):
    """Raised when an external input is not forwarded"""
    def __init__(self, block, channels):
        message = f'{block} has external inputs that are not connected to' \
                  f'internal components: {channels}'

        super().__init__(message)


class UnconnectedInputError(ValueError):
    """Raised when a subcomponent doesn't have data feeding it."""
    def __init__(self, block, child_ports):
        message = f'Composite block {block} has child components with ' \
                  f'unconnected input channels: {child_ports}'
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
        message = f'Composite {component} has output ports without no sources:'
        message += ','.join(list(str(c) for c in channel))
        super().__init__(message)


class InvalidPort(ValueError):
    pass


class FunctionError(ValueError):
    def __init__(self, block, function, error):
        message = f'Failed to evaluate {function} on block {block}: {error}'
        super().__init__(message)


class InvalidShape(ValueError):
    pass


class EvaluationError(ValueError):
    def __init__(self, graph, context, path, exception):
        message = f'''{exception} was raised during the evaluation of an
        expression graph.\n Evaluation order: {path} with context {context}'''
        super().__init__(message)


class InternalWireNotFound(Exception):
    def __init__(self, wire: Tuple, *args):
        message = f'Failed to connect {str(wire[0])} -> {str(wire[1])}.'
        super().__init__(message, *args)


class NoTopLevelOutputs(Exception):
    def __init__(self, root):
        message = f'Top level model {root} has no defined outputs. You need ' \
                  f'to wire some from internal components to have any ' \
                  f'observables.'
        super().__init__(message)


class InvalidParameterException(Exception):
    pass
