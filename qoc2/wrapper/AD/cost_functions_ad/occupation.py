import jax.numpy as jnp
import jax
class Occupation():

    name = "Occupation"

    def __init__(self, forbidden_states,
                 cost_multiplier=1.,):
        """
        See class fields for arguments not listed here.

        Arguments:
        cost_eval_step
        forbidden_states
        system_eval_count
        """
        self.cost_multiplier = cost_multiplier
        self.forbidden_states_dagger = jnp.conjugate(forbidden_states.transpose())
        self.forbidden_states = forbidden_states
        self.state_count = jnp.count_nonzero(forbidden_states)

    def _cost(self):
        return lambda controls,states: cost(controls, states, self.cost_multiplier, self.forbidden_states_dagger)



def cost(controls, states,cost_multiplier,forbidden_states_dagger):
    """
    Compute the penalty.

    Arguments:
    controls
    states
    gradients_method

    Returns:
    cost
    """
    # The cost is the overlap (fidelity) of the evolved state and each
    # forbidden state.
    def scan_function(carry, state):
        inner_products, forbidden_states_dagger = carry
        inner_products += jnp.sum(jnp.abs(jnp.multiply(forbidden_states_dagger, state)) ** 2)
        return (inner_products, forbidden_states_dagger), None  # None is a dummy output

    # Assuming `states` and `forbidden_states_dagger` are defined
    control_eval_count = len(controls[0])
    state_count = states.shape[2]
    grads_factor = cost_multiplier / (state_count * control_eval_count)

    # Initial value of inner_products
    initial_inner_products = 0.0

    # Use jax.lax.scan
    final_carry, _ = jax.lax.scan(scan_function, (initial_inner_products, forbidden_states_dagger), states)

    # Extract the final value of inner_products
    inner_products = final_carry[0]
    fidelity = inner_products

    return grads_factor * fidelity