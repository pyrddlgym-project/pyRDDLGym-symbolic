from pathlib import Path
from typing import cast, Optional

from pyRDDLGym.core.grounder import RDDLGrounder
from pyRDDLGym.core.parser.reader import RDDLReader
from pyRDDLGym.core.parser.parser import RDDLParser
from pyRDDLGym_symbolic.core.model import RDDLModelXADD


_DIR = 'pyRDDLGym_symbolic/examples/files/{domain}/'
_DOMAIN_PATH = _DIR + 'domain.rddl'
_INSTANCE_PATH = _DIR + 'instance{instance}.rddl'


def main(
        domain: str = 'Wildfire',
        instance: str = '0',
        cpf: Optional[str] = None,
        save_graph: bool = False,
        reparam: bool = False,
):
    # Read and parse domain and instance
    reader = RDDLReader(
        _DOMAIN_PATH.format(domain=domain),
        _INSTANCE_PATH.format(domain=domain, instance=instance)
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
    xadd_model = RDDLModelXADD(model, reparam=reparam)
    xadd_model.compile()
    context = xadd_model.context

    if save_graph:
        Path(f'tmp/{domain}').mkdir(parents=True, exist_ok=True)
        
    if cpf is not None or cpf == 'reward':
        if cpf == 'reward':
            expr = xadd_model.reward
        else:
            expr = xadd_model.cpfs.get(f"{cpf}'")
        if expr is None:
            raise AttributeError(f"Cannot retrieve {cpf}' from 'model.cpfs'")
        print(f"cpf {cpf}':", end='\n')
        xadd_model.print(expr)
        if save_graph:
            f_path = f"{domain}/{domain}_inst0_{cpf}"
            context.save_graph(expr, f_path)
    else:
        for cpf_, expr in xadd_model.cpfs.items():
            print(f"cpf {cpf_}:", end='\n')
            xadd_model.print(expr)
            if save_graph:
                cpf = cpf_.strip("'")
                f_path = f"{domain}/{domain}_inst0_{cpf}"
                context.save_graph(expr, f_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--domain', type=str, default='Wildfire',
                        help='The name of the RDDL environment')
    parser.add_argument('--instance', type=str, default='0',
                        help='The instance number of the RDDL environment')
    parser.add_argument('--cpf', type=str, default=None,
                        help='If specified, only print out this CPF')
    parser.add_argument('--save_graph', action='store_true',
                        help='Save the graph as pdf file')
    parser.add_argument('--reparam', action='store_true',
                        help='Reparameterize the random variables in the model.')
    args = parser.parse_args()

    # Run XADD model compilation.
    main(
        domain=args.domain,
        instance=args.instance,
        cpf=args.cpf,
        save_graph=args.save_graph,
        reparam=args.reparam,
    )
