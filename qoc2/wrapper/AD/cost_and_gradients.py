from .state_evolution_ad import evolution_ad
from jax import value_and_grad
import jax.numpy as jnp
from jax import jit

def _cost_AD(controls,grape,reporter):
    controls = jnp.reshape(controls, grape['control_shape'])
    cost = (evolution_ad)(controls,grape)
    return cost

def _gradients_AD(controls, grape_h, grape_nh):
    # Initialize a new array with the desired shape
    reshaped_controls = jnp.zeros(grape_h.control_shape)

    # Element-wise copy from 'controls' to 'reshaped_controls'
    # Note: Assuming 'controls' can be reshaped to 'grape['control_shape']' without loss of data
    reshaped_controls = reshaped_controls.at[:].set(controls.reshape(grape_h.control_shape))
    (cost,cost_set), grads = value_and_grad(evolution_ad,has_aux=True)(reshaped_controls,grape_h,grape_nh)
    grads = jnp.ravel(grads)
    return cost, grads, cost_set

gradients_AD = jit(_gradients_AD, static_argnums=1)
cost_AD = jit(_cost_AD, static_argnums=1)
