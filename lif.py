import ray
import numpy as np
import time
import random
# -------------------------------
# Lava-based LIF Simulation Functions (Spiking Network)
# -------------------------------
# Note: This section uses Lava library functions.
from lava.magma.core.process.process import AbstractProcess as Process
from lava.magma.core.process.variable import Var
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.decorator import implements
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi1SimCfg
def lif_simulation(spike_train, num_steps=None, v_th=1.0, du=0.95):
    """
    Run a pure NumPy-based LIF simulation.
    
    Parameters:
        spike_train: numpy array of shape (n_neurons, num_steps)
        num_steps:   int, number of simulation time steps (default = spike_train.shape[1])
        v_th:        float, voltage threshold for spiking.
        du:          float, decay factor.
    
    Returns:
        out:         numpy array of shape (n_neurons, num_steps)
                     The output spike train for each neuron.
    """
    if num_steps is None:
        num_steps = spike_train.shape[1]
        
    n_neurons = spike_train.shape[0]
    v = np.zeros(n_neurons, dtype=np.float32)
    out = np.zeros((n_neurons, num_steps), dtype=np.float32)
    
    for t in range(num_steps):
        inp = spike_train[:, t]
        v = du * v + inp
        spk = (v >= v_th).astype(np.float32)
        v = np.where(spk == 1, 0.0, v)
        out[:, t] = spk
        
    return out

@ray.remote
def run_lava_lif_simulation_on_node_remote(spike_train_node, num_steps=20):
    # Import inside the function to avoid serialization issues.
    return lif_simulation(spike_train_node, num_steps=num_steps, v_th=1.0, du=0.95)

def convert_to_spike_train_for_node(x, num_steps=20, noise_std=0.05):
    """
    Convert node features to a spiking pattern with noise.
    The input is perturbed with Gaussian noise, thresholded by the median,
    and then tiled across num_steps.
    """
    x_noisy = x + np.random.normal(0, noise_std, size=x.shape)
    threshold = np.median(x_noisy)
    spike_pattern = (x_noisy > threshold).astype(np.float32)
    return np.tile(spike_pattern, (num_steps, 1)).T

def run_spiking_network(node_features, num_steps=20, batch_size=50, use_lava=True):
    """
    Process nodes in batches to compute spike-based embeddings.
    Uses Lava for simulation if use_lava=True.
    """
    num_nodes = node_features.shape[0]
    all_embeddings = []
    
    print(f"[INFO] Starting spiking network processing with {num_nodes} nodes (Batch Size: {batch_size})...")
    start_time = time.time()
    
    simulation_func = run_lava_lif_simulation_on_node_remote  # Only Lava simulation in this example.
    
    for i in range(0, num_nodes, batch_size):
        batch_start_time = time.time()
        futures = [
            simulation_func.remote(
                convert_to_spike_train_for_node(node_features[j], num_steps=num_steps),
                num_steps
            )
            for j in range(i, min(i + batch_size, num_nodes))
        ]
        try:
            batch_results = ray.get(futures)
        except Exception as e:
            print(f"[ERROR] Exception during batch {i} processing: {e}")
            raise
        all_embeddings.extend(batch_results)
        
        if i % (batch_size * 10) == 0:
            elapsed = time.time() - batch_start_time
            print(f"[INFO] Processed {i}/{num_nodes} nodes - Batch time: {elapsed:.2f}s")
    
    spike_embeddings = np.array(all_embeddings, dtype=np.float32)
    total_time = time.time() - start_time
    print(f"[INFO] Finished spiking network processing in {total_time:.2f}s")
    print(f"[INFO] Extracted spike-based embeddings of shape: {spike_embeddings.shape}")
    
    return spike_embeddings
