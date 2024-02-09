from typing import Dict, Any, Tuple, Union
import numpy as np

from xaddpy.xadd.xadd import VAR_TYPE

from pyRDDLGym.core.compiler.model import RDDLPlanningModel
from pyRDDLGym.core.debug.logger import Logger
from pyRDDLGym.core.grounder import RDDLGrounder
from pyRDDLGym.core.parser.expr import Expression, Value
from pyRDDLGym.core.simulator import RDDLSimulator

from pyRDDLGym_symbolic.core.model import RDDLModelXADD


class RDDLSimulatorXADD(RDDLSimulator):

    def __init__(
            self, 
            rddl: RDDLPlanningModel,
            allow_synchronous_state: bool = True,
            logger: Logger = None,
            rng: np.random.Generator = np.random.default_rng(),
            **_,
    ) -> None:
        super().__init__(
            rddl, allow_synchronous_state, rng, logger, keep_tensors=False
        )
        self._rng = rng

    def _compile(self):
        """Compile the simulator. """
        super()._compile()

        # Ground the model
        grounder = RDDLGrounder(self.rddl.ast)
        g_model = grounder.ground()

        # Perform XADD compilation
        xadd_model = RDDLModelXADD(g_model, reparam=True)
        xadd_model.compile()
        self.xadd_model = xadd_model
        self.context = self.xadd_model.context

        # Update the CPFs
        cpfs = []
        for name, expr, dtype in self.cpfs:
            _val = False if dtype == 'bool' else 0
            node_ids = np.zeros_like(self.subs[name], dtype=int)
            shape = node_ids.shape
            node_ids = node_ids.flatten()
            for i, (gvar, _) in enumerate(self.rddl.ground_var_with_value(name, _val)):
                node_id = xadd_model.cpfs[gvar]
                node_ids[i] = node_id
            node_ids = node_ids.reshape(shape)
            cpfs.append((name, (expr, node_ids), dtype))
        self.cpfs = cpfs

    def _sample_grounded_variable(
            self, node_id: int, subs: Dict[str, Any]
    ) -> Union[bool, int, float]:
        """Sample the grounded variable.

        Args:
            node_id (int): The node id of the grounded variable
            subs (Dict[str, Any]): The dictionary containing 'fluent_name -> array of values'

        Returns:
            Any: The sampled value
        """
        var_set = self.context.collect_vars(node_id)
        cont_assign = {}
        bool_assign = {}

        # If there exists a random variable, sample it first
        # Note that due to `free_symbols` of a SymEngine expression returning a Symbol
        # object rather than the custom RandomVar object, we need to use `v.var`
        rv_set = {v.var for v in self.context._random_var_set if v.var in var_set}
        if len(rv_set) > 0:
            for rv in rv_set:
                if str(rv).startswith('#_UNIFORM'):
                    cont_assign[rv] = self._sample_uniform(node_id, subs)
                elif str(rv).startswith('#_GAUSSIAN'):
                    cont_assign[rv] = self._sample_normal(node_id, subs)
                else:
                    raise ValueError
        cont_assign.update({
            var: val for var, val in subs.items()
            if var in var_set and var in self.context._cont_var_set
        })
        
        bool_assign.update({
            var: val for var, val in subs.items()
            if var in var_set and var in self.context._bool_var_set
        })
        res = self.context.evaluate(
            node_id=node_id,
            bool_assign=bool_assign,
            cont_assign=cont_assign,
            primitive_type=True,
        )
        return res

    def _sample(self, expr: Tuple[Expression, np.ndarray], subs: Dict[str, Any]):
        """Evaluates the XADD nodes by substituting the values stored in 'subs'.

        Args:
            expr (Tuple[Expression, np.ndarray]): The expression and the node ids
            subs (Dict[str, Any]): The dictionary containing 'fluent_name -> array of values'
        """
        expr, node_ids = expr

        # Get the grounded substitutions
        g_subs = self.get_grounded_subs_from_subs(subs)

        # Define a vectorized function to apply the sample
        _apply_sample = np.vectorize(lambda x: self._sample_grounded_variable(x, g_subs))

        # Apply the sample function
        res = _apply_sample(node_ids)
        return res

    def get_grounded_subs_from_subs(self, subs: Dict[str, Any]) -> Dict[VAR_TYPE, Any]:
        """Get the grounded substitutions from the substitutions."""
        # Remove the non-fluents
        non_fluents = self.rddl.ground_vars_with_values(self.rddl.non_fluents)
        g_subs = self.rddl.ground_vars_with_values(subs)
        g_subs = {k: v for k, v in g_subs.items() if k not in non_fluents}

        # Prepare a symbol to value mapping
        # Note: only the non-prime variables are used
        g_subs = {
            self.var_name_to_sym_var[self.var_name_to_sym_var_name[v]]: val
            for v, val in g_subs.items()
            if not str(v).endswith("'")
        }
        return g_subs

    def sample_reward(self) -> float:
        """Samples the current reward given the current state and action."""
        g_subs = self.get_grounded_subs_from_subs(self.subs)
        reward = self._sample_grounded_variable(self.xadd_model.reward, g_subs)
        return reward

    def _sample_uniform(self, expr, subs):
        return self._rng.uniform(0, 1)

    def _sample_normal(self, expr, subs):
        return self._rng.standard_normal()

    def _sample_exponential(self, expr, subs):
        raise NotImplementedError

    def _sample_bernoulli(self, expr, subs):
        raise NotImplementedError

    def _sample_binomial(self, expr, subs):
        raise NotImplementedError

    def _sample_gamma(self, expr, subs):
        raise NotImplementedError

    def _sample_beta(self, expr, subs):
        raise NotImplementedError

    def _sample_weibull(self, expr, subs):
        raise NotImplementedError

    def _sample_dirac_delta(self, expr, subs):
        raise NotImplementedError

    def _sample_geometric(self, expr, subs):
        raise NotImplementedError

    def _sample_gumbel(self, expr, subs):
        raise NotImplementedError

    def _sample_kron_delta(self, expr, subs):
        raise NotImplementedError

    def _sample_negative_binomial(self, expr, subs):
        raise NotImplementedError

    def _sample_poisson(self, expr, subs):
        raise NotImplementedError

    def _sample_student(self, expr, subs):
        raise NotImplementedError

    @property
    def sym_var_name_to_var_name(self):
        return self.xadd_model._sym_var_name_to_var_name

    @property
    def var_name_to_sym_var_name(self):
        return self.xadd_model._var_name_to_sym_var_name

    @property
    def var_name_to_sym_var(self):
        return self.context._str_var_to_var
