import argparse

from pyRDDLGym_symbolic.core.visualizer import RDDL2Graph


_DIR = 'pyRDDLGym_symbolic/examples/files/{domain}/'
_DOMAIN_PATH = _DIR + 'domain.rddl'
_INSTANCE_PATH = _DIR + 'instance{instance}.rddl'


def main(args: argparse.Namespace):
    domain = args.domain
    print(f"Creating DBN graph for {domain}...")
    domain_file = _DOMAIN_PATH.format(domain=domain)
    instance_file = _INSTANCE_PATH.format(domain=domain, instance=args.instance)

    r2g = RDDL2Graph(
        domain=domain,
        instance=args.instance,
        domain_file=domain_file,
        instance_file=instance_file,
        directed=not args.undirected,
        strict_grouping=args.strict_grouping,
        reparam=args.reparam,
    )
    if args.fluent and args.gfluent:
        r2g.save_dbn(file_name=args.domain, fluent=args.fluent, gfluent=args.gfluent)
    elif args.fluent:
        r2g.save_dbn(file_name=args.domain, fluent=args.fluent)
    else:
        r2g.save_dbn(file_name=args.domain)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--domain', type=str, default='Wildfire',
                        help='The name of the RDDL environment')
    parser.add_argument('--instance', type=str, default='0',
                        help='The instance number of the RDDL environment')
    parser.add_argument("--undirected", action='store_true')
    parser.add_argument("--strict_grouping", action='store_true')
    parser.add_argument("--fluent", type=str, default=None)
    parser.add_argument("--gfluent", type=str, default=None)
    parser.add_argument('--reparam', action='store_true')
    args = parser.parse_args()

    # Run DBN visualization.
    main(args)
