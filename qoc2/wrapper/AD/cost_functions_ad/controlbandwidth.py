# import jax.numpy as jnp
#
#
#
# class ControlBandwidth():
#     """
#     This cost penalizes control frequencies above a set maximum.
#
#     Fields:
#     max_bandwidths :: ndarray (CONTROL_COUNT, 2) - This array contains the minimum and maximum allowed bandwidth of each control.
#     control_count
#     dt
#     name
#     requires_step_evaluation
#     type
#
#     Example Usage:
#     dt = 1 # zns
#     MAX_BANDWIDTH_0 = [0.01,0.4] # GHz
#     MAX_BANDWIDTHS = anp.array[MAX_BANDWIDTH_0,] for single control field
#     COSTS = [ControlBandwidthMax(dt, MAX_BANDWIDTHS)]
#     """
#     name = "control_bandwidth"
#
#
#     def __init__(self,  max_bandwidths,
#                  cost_multiplier=1., ):
#         """
#         See class fields for arguments not listed here.
#
#         Arguments:
#         dt
#         max_bandwidths
#         cost_multiplier
#         """
#         self.cost_multiplier = cost_multiplier
#         self.bandwidths = max_bandwidths
#
#     def _dt(self, dt, control_eval_count):
#         self.dt = dt
#         self.control_eval_count = control_eval_count
#         self.freqs = jnp.fft.fftfreq(jnp.int64(control_eval_count), d=dt)
#
#     def _cost(self):
#
#         return lambda controls,states: cost(controls, states, self.dt, self.bandwidths, self.cost_multiplier, self.control_eval_count,self.freqs)
#
# def cost(controls, states, dt, bandwidths,cost_multiplier,control_eval_count,freqs ):
#     """
#     Compute the penalty.
#
#     Arguments:
#     controls
#     states
#     gradients_method
#
#     Returns:
#     cost
#     """
#
#     cost = 0
#     # Iterate over the controls, penalize each control that has
#     # frequencies greater than its maximum frequency or smaller than its minimum frequency.
#     for i, bandwidth in enumerate(bandwidths):
#         min_bandwidth, max_bandwidth = bandwidth
#         control_fft = jnp.fft.fft(controls[i])
#         control_fft_sq = jnp.abs(control_fft)
#
#         # Create masks for frequencies outside the allowed bandwidth
#         mask_high = jnp.round(jnp.abs(freqs), 10) > max_bandwidth
#         mask_low = jnp.round(jnp.abs(freqs), 10) < min_bandwidth
#         non_zero_count_high = jnp.sum(mask_high)
#         non_zero_count_low = jnp.sum(mask_low)
#         # Apply masks directly to calculate penalties
#         penalty_high = jnp.sum(control_fft_sq * mask_high)
#         penalty_low = jnp.sum(control_fft_sq * mask_low)
#         print(jnp.sum(control_fft_sq * mask_high).primal)
#         # Sum the penalties and add to the total cost
#         cost += penalty_high + penalty_low
#     cost_value = cost * cost_multiplier/(non_zero_count_high+non_zero_count_low)
#     print(cost_value.primal)
#     return cost_value

import jax.numpy as jnp
import jax


class ControlBandwidth():
    """
    This cost penalizes control frequencies above a set maximum.

    Fields:
    max_bandwidths :: ndarray (CONTROL_COUNT, 2) - This array contains the minimum and maximum allowed bandwidth of each control.
    control_count
    dt
    name
    requires_step_evaluation
    type

    Example Usage:
    dt = 1 # zns
    MAX_BANDWIDTH_0 = [0.01,0.4] # GHz
    MAX_BANDWIDTHS = anp.array[MAX_BANDWIDTH_0,] for single control field
    COSTS = [ControlBandwidthMax(dt, MAX_BANDWIDTHS)]
    """
    name = "control_bandwidth"


    def __init__(self,  max_bandwidths,
                 cost_multiplier=1., ):
        """
        See class fields for arguments not listed here.

        Arguments:
        dt
        max_bandwidths
        cost_multiplier
        """
        self.cost_multiplier = cost_multiplier
        self.bandwidths = max_bandwidths

    def _dt(self, dt, control_eval_count):
        self.dt = dt
        self.control_eval_count = control_eval_count
        self.freqs = jnp.fft.fftfreq(jnp.int64(control_eval_count), d=dt)
        self.bandwidths = jnp.pad(self.bandwidths, ((0, 0), (0, control_eval_count - 2)))

    def _cost(self):

        return lambda controls,states: cost(controls, states, self.dt, self.bandwidths, self.cost_multiplier, self.control_eval_count,self.freqs)

def cost(controls, states, dt, bandwidths,cost_multiplier,control_eval_count,freqs ):
    """
    Compute the penalty.

    Arguments:
    controls
    states
    gradients_method

    Returns:
    cost
    """

    # Iterate over the controls, penalize each control that has
    # frequencies greater than its maximum frequency or smaller than its minimum frequency.
    def calculate_penalty(carry, control_bandwidth):
        cost, freqs, cost_multiplier = carry
        control = control_bandwidth[0]
        bandwidths = control_bandwidth[1]
        max_bandwidth = bandwidths[1]
        min_bandwidth = bandwidths[0]
        control_fft = jnp.fft.fft(control)
        control_fft_sq = jnp.abs(control_fft)

        # Create masks for frequencies outside the allowed bandwidth
        mask_high = jnp.round(jnp.abs(freqs), 10) > max_bandwidth
        mask_low = jnp.round(jnp.abs(freqs), 10) < min_bandwidth
        non_zero_count_high = jnp.sum(mask_high)
        non_zero_count_low = jnp.sum(mask_low)
        # Apply masks directly to calculate penalties
        penalty_high = jnp.sum(control_fft_sq * mask_high)
        penalty_low = jnp.sum(control_fft_sq * mask_low)

        # Update the total cost
        new_cost = cost + (penalty_high + penalty_low)/(non_zero_count_high+non_zero_count_low)
        return (new_cost, freqs, cost_multiplier), None

    # Prepare the control-bandwidth pairs
    control_bandwidth_pairs = jnp.stack([controls,bandwidths],axis=1)
    # Initial value for the cost accumulator
    initial_cost = 0.0

    # Use jax.lax.scan
    final_carry, _ = jax.lax.scan(calculate_penalty, (initial_cost, freqs, cost_multiplier), control_bandwidth_pairs)
    # Extract the final total cost and apply cost_multiplier
    total_cost = final_carry[0] * final_carry[2]
    return total_cost