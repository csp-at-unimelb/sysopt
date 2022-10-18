from typing import List, Dict, Any

from sysopt.symbolic.core import ExpressionGraph, GraphWrapper, ConstantFunction, recursively_apply, Variable


def function_from_graph(graph: ExpressionGraph, arguments: List[Variable]):

    if graph in arguments or isinstance(graph, ExpressionGraph):
        return GraphWrapper(graph, arguments)

    if graph is None:
        return None

    return ConstantFunction(graph, arguments)


def substitute(graph: ExpressionGraph, symbols: Dict[Variable, Any]):

    def on_leaf_node(node):
        try:
            return symbols[node]
        except KeyError:
            return node

    def on_trunk_node(op, *args):
        return ExpressionGraph(op, *args)

    return recursively_apply(graph,
                             trunk_function=on_trunk_node,
                             leaf_function=on_leaf_node)

