from collections import namedtuple
import jax.numpy as jnp
import jax
from qoc2.matrix_exponential.pade import determine_pade_order, one_norm, THETA_JAX, PADE_ORDERS_JAX


class grape_info():
    def __init__(self, H_s, H_controls, control_eval_count, costs, evolution_time,
                 initial_states, control_func, impose_control_conditions, max_iteration, gradient_method):
        self.H_s = H_s
        self.H_controls = H_controls
        self.control_eval_count = int(control_eval_count)
        self.evolution_time = evolution_time
        self.initial_states = initial_states
        self.max_iteration = max_iteration
        self.gradient_method = gradient_method
        self.control_count = len(self.H_controls)
        self.control_shape = (self.control_count, self.control_eval_count)
        self.dt = self.evolution_time / self.control_eval_count
        self.scale, self.pade_order = self.pade_args()
        self.cost_num = len(costs)
        self.costs = self.costs_format_converter(costs)
        self.control_func = self.control_func_converter(control_func )
        self.impose_control_conditions = impose_control_conditions
    def pade_args(self):
        one_norm_ = one_norm(-1j * self.H_s * self.dt)

        pade_order = determine_pade_order(one_norm_, PADE_ORDERS_JAX, THETA_JAX)

        # Scaling and squaring if necessary
        scale =  jax.lax.cond(one_norm_ >= THETA_JAX[pade_order],
                                  lambda _: jnp.ceil(jnp.log2(one_norm_ / THETA_JAX[13])).astype(int),
                                  lambda _: 0,
                                  operand=None)
        return float(scale),int(pade_order)


    # Named tuple for H-related attributes
    GrapeInfoTupleNH = namedtuple('GrapeInfoTupleH', [
        'H_s', 'H_controls', 'initial_states'
    ])

    # Named tuple for non-H attributes
    GrapeInfoTupleH = namedtuple('GrapeInfoTupleNH', [
        'control_eval_count', 'evolution_time', 'max_iteration',
        'gradient_method', 'control_count', 'control_shape','pade_order','scale','costs'
        ,'cost_num', 'impose_control_conditions', 'control_func'
    ])

    def costs_format_converter(self, costs):
        cost_named_tuple_fields = []
        cost_named_tuple_values = []

        for i, cost in enumerate(costs):
            if cost.name == "control_bandwidth":
                cost._dt(self.dt,self.control_eval_count)
            cost_named_tuple_fields.append(cost.name)
            cost_named_tuple_values.append(cost._cost())  # Call the format_covert method and append the result

        # Create the named tuple dynamically
        cost_named_tuple = namedtuple('Costs', cost_named_tuple_fields)

        return cost_named_tuple(*cost_named_tuple_values)

    def control_func_converter(self, control_funcs):
        control_named_tuple_fields = []
        control_named_tuple_values = []

        for i, control_func in enumerate(control_funcs):
            control_named_tuple_fields.append("control"+str(i))
            control_named_tuple_values.append(control_func)  # Call the format_covert method and append the result

        # Create the named tuple dynamically
        control_named_tuple = namedtuple('control_funcs', control_named_tuple_fields)

        return control_named_tuple(*control_named_tuple_values)

    def to_namedtuple(self):
        grape_nh = self.GrapeInfoTupleNH(
            H_s=self.H_s,
            H_controls=self.H_controls,
            initial_states=self.initial_states,

        )

        grape_h = self.GrapeInfoTupleH(
            control_eval_count=self.control_eval_count,
            evolution_time=self.evolution_time,
            max_iteration=self.max_iteration,
            gradient_method=self.gradient_method,
            control_count=self.control_count,
            control_shape=self.control_shape,
            pade_order = self.pade_order,
            scale =  self.scale,
            costs=self.costs,
            cost_num= self.cost_num,
            impose_control_conditions = self.impose_control_conditions, control_func = self.control_func 
        )

        return grape_h, grape_nh

