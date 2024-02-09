"""An example RDDL simulation run."""

import argparse
import pprint

import pyRDDLGym
from pyRDDLGym.core.policy import RandomAgent

from pyRDDLGym_symbolic.core.simulator import RDDLSimulatorXADD


def main(args: argparse.Namespace):
    # Make the environment
    env = pyRDDLGym.make(
        args.domain,
        args.instance,
        backend=RDDLSimulatorXADD,
    )
    env.seed(args.seed)

    # Set up an example agent
    agent = RandomAgent(
        action_space=env.action_space,
        num_actions=env.max_allowed_actions,
        seed=args.seed,
    )

    # Main evaluation loop
    for episode in range(args.num_episodes):
        total_reward = 0
        state, _ = env.reset()

        for step in range(env.horizon):
            env.render()
            action = agent.sample_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            print(f'step       = {step}\n'
                  f'state      =\n{pprint.pformat(state, indent=4)}\n'
                  f'action     =\n{pprint.pformat(action, indent=4)}\n'
                  f'next state =\n{pprint.pformat(next_state, indent=4)}\n'
                  f'reward     = {reward}\n')
            total_reward += reward
            state = next_state
            if done:
                break
        print(f'episode {episode} ended with return {total_reward}')

    # Important when logging to save all traces
    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run RDDL simulation.')

    parser.add_argument('--domain', type=str, default='Wildfire',
                        help='RDDL domain name.')
    parser.add_argument('--instance', type=str, default='0',
                        help='RDDL instance number.')
    parser.add_argument('--num_episodes', type=int, default=1,
                        help='Number of episodes to run.')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed for environment.')
    args = parser.parse_args()

    main(args)
