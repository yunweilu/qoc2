import jax.numpy as jnp
def cos_func(freq):
    def _cos_func(control,times,i):
        return jnp.cos(times * freq) * control[i]
    return _cos_func

def pwc():
    def _cos_func(control,times,i):
        return control[i]
    return _cos_func