"""Implements some utility functions for XADDs."""

from typing import Callable, Dict, List, Set, Union

import symengine.lib.symengine_wrapper as core
from xaddpy.utils.util import get_bound
from xaddpy.xadd.xadd import XADD, XADDLeafOperation, VAR_TYPE


def get_bernoulli_node_id(
    expr_or_var: Union[core.Symbol, core.BooleanAtom]
) -> int:
    """Returns the node ID associated with the Bernoulli param."""
    # Note: the pattern is f'{dist}_params_{len(args)}_{proba}'.
    if isinstance(expr_or_var, core.BooleanAtom):
        return 1 if expr_or_var else 0
    name = str(expr_or_var)
    dist, _, len_args, param = name.split('_')
    assert dist == 'Bernoulli', (
        "The leaf node must be a Bernoulli node."
    )
    assert len_args == '1', (
        "The length of the arguments must be 1."
    )
    assert param.isdigit(), (
        "The parameter must be a digit corresponding to"
        "the node ID of the Bernoulli parameter."
    )
    return int(param)


class BoundAnalysis(XADDLeafOperation):
    """Leaf operation that configures bounds over set variables.
    
    Args:
        context: The XADD context.
        var_set: The set of variables to configure bounds for.
    """

    def __init__(
            self,
            context: XADD,
            var_set: Set[VAR_TYPE],
    ):
        super().__init__(context)
        self.var_set = var_set
        self.lb_dict: Dict[VAR_TYPE, Union[core.Basic, int, float]] = {}
        self.ub_dict: Dict[VAR_TYPE, Union[core.Basic, int, float]] = {}

    def process_xadd_leaf(
            self,
            decisions: List[core.Basic],
            decision_values: List[bool],
            leaf_val: core.Basic,
    ) -> int:
        """Processes an XADD partition to configure bounds.

        Args:
            decisions: The list of decisions.
            decision_values: The list of decision values.
            leaf_val: The leaf value.
        Returns:
            The ID of the leaf node passed.
        """
        assert isinstance(leaf_val, core.BooleanAtom) or isinstance(leaf_val, bool)
        # False leaf node represents an invalid partition.
        if not leaf_val:
            return self._context.get_leaf_node(leaf_val)

        # Iterate over the decisions and decision values.
        for dec_expr, is_true in zip(decisions, decision_values):
            # Iterate over the variables.
            for v in self.var_set:
                if v not in dec_expr.atoms():
                    continue
                assert dec_expr not in self._context._bool_var_set
                lhs, rhs = dec_expr.args
                lt = dec_expr.is_Relational and isinstance(dec_expr, core.LessThan)
                lt = (lt and is_true) or (not lt and not is_true)
                expr = lhs <= rhs if lt else lhs >= rhs

                # Get bounds over `v`.
                bound_expr, is_ub = get_bound(v, expr)
                if is_ub:
                    ub = self.ub_dict.setdefault(v, bound_expr)
                    # Get the tightest upper bound.
                    self.ub_dict[v] = min(ub, bound_expr)
                else:
                    lb = self.lb_dict.setdefault(v, bound_expr)
                    # Get the tighest lower bound.
                    self.lb_dict[v] = max(lb, bound_expr)

        return self._context.get_leaf_node(leaf_val)


class ValueAssertion(XADDLeafOperation):
    """Leaf operation that applies an assertion function.
    
    Args:
        context: The XADD context.
        fn: The function to apply.
        msg: The message to display if the assertion fails.
    """

    def __init__(
            self,
            context: XADD,
            fn: Callable[[core.Basic], bool],
            msg: str = None,
    ):
        super().__init__(context)
        self.fn = fn
        self.msg = 'Assertion failed on {leaf_val}' if msg is None else msg

    def process_xadd_leaf(
            self,
            decisions: List[core.Basic],
            decision_values: List[bool],
            leaf_val: core.Basic,
    ) -> int:
        """Processes an XADD partition to assert the type.

        Args:
            *args: Unused arguments.
            leaf_val: The leaf value.

        Returns:
            The ID of the leaf node passed.
        """
        assert self.fn(leaf_val), self.msg.format(leaf_val=str(leaf_val))
        return self._context.get_leaf_node(leaf_val)
