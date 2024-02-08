"""An example VI run."""

import argparse
import os

from pyRDDLGym.core.grounder import RDDLGrounder
from pyRDDLGym.core.parser.reader import RDDLReader
from pyRDDLGym.core.parser.parser import RDDLParser

from pyRDDLGym_symbolic.core.model import RDDLModelXADD
from pyRDDLGym_symbolic.mdp.mdp_parser import MDPParser
from pyRDDLGym_symbolic.solver.vi import ValueIteration


_DIR = 'pyRDDLGym_symbolic/examples/files/{domain}/'
_DOMAIN_PATH = _DIR + 'domain.rddl'
_INSTANCE_PATH = _DIR + 'instance{instance}.rddl'


def run_vi(args: argparse.Namespace):
    """Runs VI."""
    # Read and parse domain and instance
    domain = args.domain
    instance = args.instance
    domain_file = _DOMAIN_PATH.format(domain=domain)
    instance_file = _INSTANCE_PATH.format(domain=domain, instance=instance)
    reader = RDDLReader(
        domain_file,
        instance_file,
    )
    rddl_txt = reader.rddltxt
    parser = RDDLParser(None, False)
    parser.build()

    # Parse RDDL file
    rddl_ast = parser.parse(rddl_txt)

    # Ground domain
    grounder = RDDLGrounder(rddl_ast)
    model = grounder.ground()

    # XADD compilation
    xadd_model = RDDLModelXADD(model, reparam=False)
    xadd_model.compile()

    mdp_parser = MDPParser()
    mdp = mdp_parser.parse(
        xadd_model,
        xadd_model.discount,
        concurrency=rddl_ast.instance.max_nondef_actions,
        is_linear=args.is_linear,
        include_noop=not args.skip_noop,
        is_vi=True,
    )

    vi_solver = ValueIteration(
        mdp=mdp,
        max_iter=args.max_iter,
        enable_early_convergence=args.enable_early_convergence,
        perform_reduce_lp=args.reduce_lp,
    )
    res = vi_solver.solve()

    # Export the solution to a file.
    env_path = os.path.dirname(domain_file)
    sol_dir = os.path.join(env_path, 'sdp', 'vi')
    os.makedirs(sol_dir, exist_ok=True)
    for i in range(args.max_iter):
        sol_fpath = os.path.join(sol_dir, f'value_dd_iter_{i+1}.xadd')
        value_dd = res['value_dd'][i]
        mdp.context.export_xadd(value_dd, fname=sol_fpath)

        # Visualize the solution XADD.
        if args.save_graph:
            # XADD.save_graph by default adds files to ./tmp...
            # Below is a hack to just override the file path to the sol_dir
            graph_fpath = os.path.join(os.path.abspath(sol_dir), f'value_dd_iter_{i+1}.pdf')
            mdp.context.save_graph(value_dd, file_name=graph_fpath)
    print(f'Times per iterations: {res["time"]}')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--domain', type=str, default='RobotLinear_1D',
                        help='The name of the RDDL environment')
    parser.add_argument('--instance', type=str, default='0',
                        help='The instance number of the RDDL environment')
    parser.add_argument('--max_iter', type=int, default=10,
                        help='The maximum number of iterations')
    parser.add_argument('--enable_early_convergence', action='store_true',
                        help='Whether to enable early convergence')
    parser.add_argument('--is_linear', action='store_true',
                        help='Whether the MDP is linear or not')
    parser.add_argument('--reduce_lp', action='store_true',
                        help='Whether to perform the reduce LP function.')
    parser.add_argument('--skip_noop', action='store_true',
                        help='Whether to skip the noop action')
    parser.add_argument('--save_graph', action='store_true',
                        help='Whether to save the XADD graph to a file')
    args = parser.parse_args()

    run_vi(args)
