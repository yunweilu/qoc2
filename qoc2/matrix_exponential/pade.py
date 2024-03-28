from jax import config
import jax.numpy as jnp
config.update("jax_enable_x64", True)
import jax
# Pade approximants from algorithm 2.3.
B = (
    64764752532480000,
    32382376266240000,
    7771770303897600,
    1187353796428800,
    129060195264000,
    10559470521600,
    670442572800,
    33522128640,
    1323241920,
    40840800,
    960960,
    16380,
    182,
    1,
)


def one_norm(a):
    """
    Return the one-norm of the matrix.

    References:
    [0] https://www.mathworks.com/help/dsp/ref/matrix1norm.html

    Arguments:
    a :: ndarray(N x N) - The matrix to compute the one norm of.

    Returns:
    one_norm_a :: float - The one norm of a.
    """
    return jnp.max(jnp.sum(jnp.abs(a), axis=0))


def pade3(a, i):
    a2 = jnp.matmul(a, a)
    u = jnp.matmul(a, B[2] * a2) + B[1] * a
    v = B[2] * a2 + B[0] * i
    return u, v


def pade5(a, i):
    a2 = jnp.matmul(a, a)
    a4 = jnp.matmul(a2, a2)
    u = jnp.matmul(a, B[5] * a4 + B[3] * a2) + B[1] * a
    v = B[4] * a4 + B[2] * a2 + B[0] * i
    return u, v


def pade7(a, i):
    a2 = jnp.matmul(a, a)
    a4 = jnp.matmul(a2, a2)
    a6 = jnp.matmul(a2, a4)
    u = jnp.matmul(a, B[7] * a6 + B[5] * a4 + B[3] * a2) + B[1] * a
    v = B[6] * a6 + B[4] * a4 + B[2] * a2 + B[0] * i
    return u, v


def pade9(a, i):
    a2 = jnp.matmul(a, a)
    a4 = jnp.matmul(a2, a2)
    a6 = jnp.matmul(a2, a4)
    a8 = jnp.matmul(a2, a6)
    u = jnp.matmul(a, B[9] * a8 + B[7] * a6 + B[5] * a4 + B[3] * a2) + B[1] * a
    v = B[8] * a8 + B[6] * a6 + B[4] * a4 + B[2] * a2 + B[0] * i
    return u, v


def pade13(a, i):
    a2 = jnp.matmul(a, a)
    a4 = jnp.matmul(a2, a2)
    a6 = jnp.matmul(a2, a4)
    u = jnp.matmul(a, jnp.matmul(a6, B[13] * a6 + B[11] * a4 + B[9] * a2) + B[7] * a6 + B[5] * a4 + B[3] * a2) + B[1] * a
    v = jnp.matmul(a6, B[12] * a6 + B[10] * a4 + B[8] * a2) + B[6] * a6 + B[4] * a4 + B[2] * a2 + B[0] * i
    return u, v
# Valid pade orders for algorithm 2.3.
PADE_ORDERS = (
    3,
    5,
    7,
    9,
    13,
)

# Pade approximation functions.
PADE = [
    None,
    None,
    None,
    pade3,
    None,
    pade5,
    None,
    pade7,
    None,
    pade9,
    None,
    None,
    None,
    pade13,
]

# Constants taken from table 2.3.
THETA = (
    0,
    0,
    0,
    1.495585217958292e-2,
    0,
    2.539398330063230e-1,
    0,
    9.504178996162932e-1,
    0,
    2.097847961257068,
    0,
    0,
    0,
    5.371920351148152,
)
# Convert PADE_ORDERS and THETA to JAX arrays
PADE_ORDERS_JAX = jnp.array(PADE_ORDERS)
THETA_JAX = jnp.array(THETA)


# JIT-friendly function to determine Pade order
def determine_pade_order(one_norm_, pade_orders_jax, thetas_jax):
    max_order = pade_orders_jax.shape[0]

    def body_fun(i, val):
        pade_order, found = val
        condition = (one_norm_ < thetas_jax[pade_orders_jax[i]]) & (~found)
        new_pade_order = jax.lax.cond(condition, lambda _: pade_orders_jax[i], lambda _: pade_order, operand=None)
        return new_pade_order, found | condition

    pade_order, _ = jax.lax.fori_loop(0, max_order, body_fun, (13, False))  # Default to 13
    return pade_order

def expm_pade(a, pade_order, scale):

    size = a.shape[0]
    a_scaled = a * (2 ** -(scale))
    i = jnp.eye(size)
    # Execute Pade approximant
    # Define branches for jax.lax.switch
    branches = [lambda _: pade3(a_scaled, i),
                lambda _: pade5(a_scaled, i),
                lambda _: pade7(a_scaled, i),
                lambda _: pade9(a_scaled, i),
                lambda _: pade13(a_scaled, i)]

    # Convert pade_order to index for jax.lax.switch
    # Execute Pade approximant
    u, v = jax.lax.switch(pade_order, branches, None)
    r = jnp.linalg.solve(-u + v, u + v)
    # Squaring
    def squaring_step(r, _):
        return jnp.matmul(r, r), None

    r, _ = jax.lax.scan(squaring_step, r, xs=None, length=scale)
    return r
