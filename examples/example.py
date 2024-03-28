import jax.numpy as np
from qoc2.core.optimization_trunk import GRAPE
from qoc2.wrapper.AD.cost_functions_ad.infidelity import Infidelity


# Function to generate H_s and H_controls for a given dimension d
def generate_matrices(d):
    H_s = np.diag(np.linspace(-0.5, 0.5, d))  # Diagonal matrix of size d
    H_controls = [np.eye(d)]  # Example control Hamiltonian, adjust as needed
    return H_s, H_controls

def bc(control):
    control[0][0]=0
    control[0][1] = 0
    return control
# Initialize other parameters
evolution_time = 10
max_iteration = 10000
control_eval_count = 1000
# Record runtimes
runtimes = {}

d=2
# Generate matrices for current dimension d
H_s = np.array([[-0.5,0],[0,0.5]])*2*np.pi
H_controls = [np.array([[0,1.],[1.,0]])]
initial_controls = np.array([0.01*2*np.pi*np.ones(control_eval_count)])
# Initialize states for dimension d
initial_states = np.array([[1.,0]],dtype=np.complex128)
target_states = np.array([[0,1.]],dtype=np.complex128)

# Update costs with the new target states
costs = [Infidelity(target_states=target_states)]

# Measure runtime
print(GRAPE(H_s, H_controls, control_eval_count, initial_controls, costs, evolution_time,
      initial_states, max_iteration,impose_control_conditions=bc))

