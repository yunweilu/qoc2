import jax.numpy as jnp

class Infidelity():
    """
    This cost penalizes the infidelity of an evolved state
    and a target state.

    Fields:
    cost_multiplier
    name
    requires_step_evaluation
    state_count
    target_states_dagger
    neglect_relative_phase
    target_states
    grads_factor
    inner_products_sum
    type
    """
    name = "Infidelity"


    def __init__(self, target_states, cost_multiplier=1., ):
        """
        See class fields for arguments not listed here.

        Arguments:
        target_states
        """
        self.cost_multiplier = cost_multiplier
        self.state_count = target_states.shape[0]
        self.target_states = target_states
        self.target_states_dagger = jnp.conjugate(target_states)


    def _cost(self):
        return lambda controls,states: cost(controls, states, self.state_count , self.cost_multiplier, self.target_states_dagger)


def cost(controls, states, state_count, cost_multiplier, target_states_dagger):
    """
            Compute the penalty.

            Arguments:
            controls
            states
            gradients_method

            Returns:
            cost
            """
    # The cost is the infidelity of each evolved state and its target state.
    state = states[-1]
    inner_products = jnp.matmul(target_states_dagger, state)
    inner_products_sum = jnp.trace(inner_products)
    fidelity = jnp.real(
        inner_products_sum * jnp.conjugate(inner_products_sum)) / state_count ** 2
    infidelity = 1 - fidelity
    cost_value = infidelity * cost_multiplier

    return cost_value
