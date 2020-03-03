from finitemdp import FiniteMDP
from gridworld import GridWorld
import numpy as np


class ToyEnv1(FiniteMDP):
    """
    Simple environment that gives a reward of 1 when going to the
    last state and 0 otherwise.

    Args:
        gamma : discount factor
        seed_val    (int): Random number generator seed
    """

    def __init__(self, gamma, seed_val=42):
        # Transition probabilities
        # shape (Ns, Na, Ns)
        # P[s, a, s'] = Prob(S_{t+1}=s'| S_t = s, A_t = a)

        Ns = 3
        Na = 2
        P = np.zeros((Ns, Na, Ns))

        P[:, 0, :] = np.array([[0.25, 0.5, 0.25], [0.1, 0.7, 0.2], [0.1, 0.8, 0.1]])
        P[:, 1, :] = np.array([[0.3, 0.3, 0.4], [0.7, 0.2, 0.1], [0.25, 0.25, 0.5]])

        # Initialize base class
        states = np.arange(Ns)
        action_sets = [np.arange(Na).tolist()]*Ns
        super().__init__(gamma, states, action_sets, P, seed_val)

    def reward_fn(self, state, action, next_state):
        return 1.0 * (next_state == self.Ns - 1)

    def render(self):
        self.print()


class SimpleGridWorld(GridWorld):
    def __init__(self,
                 gamma,
                 success_probability=1.0,
                 seed_val=42):
        nrows = 3
        ncols = 6

        # No walls
        walls = ()

        self.goal_coord = (0, ncols - 1)
        traps = [(0, 1), (0, 2), (0, 3), (0, 4)]
        #traps = [(0, 1), (1, 1)]
        start_coord = (0, 0)
        reward_at = {self.goal_coord: 1}
        for trap in traps:
            reward_at[trap] = -1
        terminal_states = (self.goal_coord, )
        default_reward = 0.0

        super().__init__(gamma,
                        seed_val,
                         nrows,
                         ncols,
                         start_coord,
                         terminal_states,
                         success_probability,
                         reward_at,
                         walls)


if __name__ == '__main__':
    env = SimpleGridWorld(gamma=0.95)

