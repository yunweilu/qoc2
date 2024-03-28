import jax.numpy as jnp

def tlist(tg, control_eval_count) :
    return jnp.linspace(0,tg , control_eval_count)

def generate_random_values(N):
    # Generate random values for a1 to aN
    a_values = [jnp.random.uniform(-1, 1) for _ in range(1, N + 1)]

    # Calculate a0 as the negative sum of a1 to aN
    a0 = -sum(a_values)
    a_values.insert(0, a0)

    # Generate random values for b1 to bN
    b_values = [jnp.random.uniform(-1, 1) for _ in range(1, N + 1)]

    return a_values, b_values

def random_pulse(control_eval_count, tg,max_amplitude,max_bandwidth  ):
    N = int(max_bandwidth / (1 / tg))
    time = tlist(tg, control_eval_count)
    print(N)
    a_s,b_s = generate_random_values(N)
    control = a_s[0]*jnp.ones(control_eval_count)
    for j in range(N):
        omega = 2*jnp.pi*j/tg
        control += a_s[j+1]*jnp.cos(omega*time)+b_s[j]*jnp.sin(omega*time)
    scale = max_amplitude/(max(abs(control))/2/jnp.pi)
    return control*scale/2/jnp.pi

def random_initial_cont(control_eval_count, tg , nn,max_bandwidth= 0.5,max_amplitude = 0.01*2*jnp.pi ):
    return [[random_pulse(control_eval_count, tg,max_amplitude,max_bandwidth )] for _ in range(nn)]

def flat_initial_cont(control_eval_count, tg, nn,max_bandwidth= 0.5,max_amplitude = 0.01*2*jnp.pi ):
    return [[max_amplitude*jnp.ones(control_eval_count) for _ in range(nn)]]

