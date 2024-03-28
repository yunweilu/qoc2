from .grape_infoclass import grape_info
from qoc2.wrapper.AD.cost_and_gradients import cost_AD, gradients_AD
from qoc2.optimizers.adam import Adam
from jax import config
import jax.numpy as jnp

config.update("jax_enable_x64", True)


# hard code type annotation / result = optimization.run
def GRAPE(H_s,
          H_controls,
          control_eval_count,
          initial_controls,
          costs,evolution_time,
          initial_states,
          max_iteration,  optimizer = Adam(), gradient_method = 0):

    grape_h, grape_nh = grape_info(H_s,
          H_controls,
          control_eval_count,
          costs,evolution_time,
          initial_states,
          max_iteration,
          gradient_method).to_namedtuple()
    # Define a dictionary mapping gradient methods to their corresponding functions
    method_functions = {
        0: (cost_AD, gradients_AD),
        # 1: (cost_HG, gradients_HG),
        # 2: (cost_SAD, gradients_SAD)
    }
    # Assign the cost_function and gradients_function based on the gradient_method
    cost_function, gradients_function = method_functions[gradient_method]
    print("compilation is done")
    pulse,result = optimizer.run(cost_function,
                        max_iteration,
                        initial_controls,
                        gradients_function,
                        args=(grape_h,grape_nh))
    return pulse,result


