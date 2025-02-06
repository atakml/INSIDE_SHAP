import os
import time
import subprocess
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

def is_gpu_available(required_memory=1024):
    """
    Checks if any GPU has at least `required_memory` MB of free memory.
    Adjust `required_memory` to the minimum memory needed for your process.
    """
    try:
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            encoding="utf-8"
        )
        # Get free memory for all GPUs
        free_memory = [int(x) for x in result.strip().split("\n")]
        return any(mem >= required_memory for mem in free_memory)
    except FileNotFoundError:
        print("nvidia-smi command not found. Ensure NVIDIA drivers are installed.")
        return True  # Assume availability if nvidia-smi isn't available
    except Exception as e:
        print(f"Error checking GPU availability: {e}")
        return False

def wait_for_gpu(required_memory=1024, check_interval=30):
    """
    Waits until a GPU has enough memory available.
    """
    while not is_gpu_available(required_memory):
        print(f"No GPU available with {required_memory}MB free memory. Retrying in {check_interval} seconds...")
        time.sleep(check_interval)

def run_sequentially_for_dataset(dataset, required_memory=1024):
    """
    Run the command sequentially 5 times for a specific dataset, ensuring GPU availability.
    """
    for _ in range(5):
        wait_for_gpu(required_memory)  # Ensure sufficient GPU memory before execution
        command = f"python surrogatemod/traingin.py {dataset} --method inside --random"
        os.system(command)

datasets = ["ba2", "AlkaneCarbonyl", "aids", "BBBP", "mutag", "Benzen"]

# Use ProcessPoolExecutor to run each dataset in parallel
with ProcessPoolExecutor() as executor:
    # Submit each dataset to the pool
    futures = [
        executor.submit(run_sequentially_for_dataset, dataset, required_memory=1024)
        for dataset in datasets
    ]

    # Optionally track progress
    for future in tqdm(futures, desc="Processing datasets"):
        future.result()  # Wait for each process to complete
