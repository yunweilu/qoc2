import numpy as np

class OptimizationResult:
    def __init__(self,):
        # Initialize a 2D array to store cost values
        self.cost_values = []
        self.grads = []
        self.iteration_num = 0

    def result_update(self, total_cost, cost_values, grads):
        # Update the cost values for all cost functions at a given iteration
        self.cost_num = len(cost_values)
        if self.iteration_num ==0:
            self.print_title()
        self.cost_values.append(cost_values)
        grads_norm = np.linalg.norm(grads)
        self.grads.append(grads_norm)
        self.iteration_num += 1
        self.print_result(total_cost, cost_values, grads_norm)

    def print_title(self, ):
        output = "iter   |   total error  |"
        dash = "========================="
        for i in range(self.cost_num):
            output += ("       cost" + str(i) + "      |")
            dash += "================="
        output += ("   grads_norm  ")
        dash += "======================="
        print(output)
        print(dash)

    def print_result(self, total_cost, cost_values, grads):
        grads_norm = np.linalg.norm(grads)
        output = "{:^6d} | {:^1.8e} |".format(self.iteration_num, total_cost, )
        cost_len = len(cost_values)
        for i in range(cost_len):
            output += "  {:^1.8e}  |".format(cost_values[i])
        output += "  {:^1.8e}  ".format(grads_norm)
        print(output)

    def reformat(self):
        self.cost_values = jnp.array(self.cost_values)
        self.cost_values = self.cost_values.reshape((-1, self.optimization_result.shape[0]))
