import jax.numpy as jnp
import jax
from qoc2.matrix_exponential.pade import expm_pade

def get_H_total(controls, H_controls, H_s, time_step):
    # Hypothetical function structure
    # Use JAX operations to compute the total Hamiltonian
    # For example, if H_total is a linear combination of control Hamiltonians:
    H_total = H_s.copy()  # Assuming H_s is a static part of the Hamiltonian

    for i, H_control in enumerate(H_controls):
        # Update H_total based on controls and H_controls
        H_total += controls[i,time_step] * H_control  # Element-wise multiplication

    return H_total
def print_callback(arg, transform):
    print("Value during JAX transform:", arg, "Transform:", transform)
    return arg
def evolution_ad(controls, grape_h, grape_nh):
    control_eval_count = grape_h.control_eval_count
    control_shape = grape_h.control_shape
    states = jnp.transpose(grape_nh.initial_states)
    dt = grape_h.evolution_time / control_eval_count
    H_s = grape_nh.H_s
    costs = grape_h.costs
    cost_num = grape_h.cost_num
    H_controls = grape_nh.H_controls
    pade_order = grape_h.pade_order
    scale = grape_h.scale
    time_steps = jnp.arange(control_eval_count, dtype=jnp.int64)
    times = jnp.linspace(0, grape_h.evolution_time-dt, control_eval_count )

    def scan_inner_loop(carry, x):
        control_k = carry
        time_i, i = x
        # Direct use of i as an index for JAX arrays is fine.
        updated_control = control_func_spec(control_k, time_i, i.astype(int))
        return control_k, updated_control

    updated_controls = jnp.zeros_like(controls)
    if grape_h.control_func is not None:
        control_func = grape_h.control_func
        for k in range(controls.shape[0]):
            global control_func_spec
            control_func_spec = control_func[k]
            control_k = controls[k]
            xs = jnp.stack((times, jnp.arange(control_eval_count)), axis=1)
            _, updated_controls_k = jax.lax.scan(scan_inner_loop, control_k, xs)
            updated_controls = updated_controls.at[k].set(updated_controls_k)
    all_states = jnp.zeros((states.shape[0], states.shape[1]), dtype=jnp.complex128)

    def scan_body(carry, time_step):
        nonlocal all_states
        states = carry

        H_total = get_H_total(updated_controls, H_controls, H_s, time_step)
        propagator = expm_pade(-1j * dt * H_total, scale=scale, pade_order=pade_order)
        new_states = jnp.matmul(propagator, states)
        all_states = new_states

        return new_states, all_states

    states, all_states = jax.lax.scan(scan_body, states, time_steps)
    intermediate_results = jnp.zeros(cost_num)
    cost_value = 0
    for i, field_name in enumerate(costs._fields):
        # Get the function variable dynamically using getattr
        function_variable = getattr(costs, field_name)
        _cost_value = function_variable(updated_controls, all_states)
        cost_value += _cost_value
        intermediate_results = intermediate_results.at[i].set(_cost_value)  # Store intermediate result

    # for time_step in range(control_eval_count):
    #     H_total = get_H_total(controls, H_controls, H_s, time_step)
    #     propagator = expm_pade(-1j *dt *H_total, pade_order, scale)
    #     states = jnp.matmul(propagator, states)
    #
    # target_states = jnp.array([jnp.zeros(len(H_s))])
    # target_states_dagger = jnp.conjugate(target_states)
    # inner_products = jnp.matmul(target_states_dagger, states)
    # inner_products_sum = jnp.trace(inner_products)
    # fidelity = jnp.real(
    #     inner_products_sum * jnp.conjugate(inner_products_sum))
    # infidelity = 1 - fidelity
    # cost_value = infidelity
    return cost_value, intermediate_results