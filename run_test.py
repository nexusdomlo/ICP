import open3d as o3d
import open3d.data
import numpy as np
import subprocess
import os

def run_command(command):
    """Executes a command in the shell and prints its output."""
    print(f"\n--- Running command: {' '.join(command)} ---\n")
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8')
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
        rc = process.poll()
        if rc != 0:
            print(f"\n--- Command failed with exit code {rc} ---")
        else:
            print(f"\n--- Command finished successfully ---")
        return rc
    except FileNotFoundError:
        print(f"Error: The command '{command[0]}' was not found.")
        return -1
    except Exception as e:
        print(f"An error occurred: {e}")
        return -1

def main():
    """
    Downloads test data and runs the registration script with different settings.
    """
    # Download the Redwood dataset for registration
    # This dataset contains two fragments of a living room scene
    print("--- Downloading test data... ---")
    dataset = open3d.data.RedwoodLivingRoomFragments()
    
    # Paths to the source and target point clouds
    src_path = dataset.paths[0]
    tgt_path = dataset.paths[1]

    print(f"Source point cloud: {src_path}")
    print(f"Target point cloud: {tgt_path}")

    # --- Test Case 1: Standard CPU Registration ---
    # This uses the default settings: FPFH global registration + CPU ICP refinement.
    cpu_command = [
        "python", "ICP/demo.py",
        src_path,
        tgt_path,
        "--voxel", "0.05",  # Use a larger voxel size for faster testing
    ]
    run_command(cpu_command)

    # --- Test Case 2: GPU-accelerated Registration ---
    # This adds the --use-gpu flag to perform ICP refinement on the GPU.
    # Note: This requires a CUDA-enabled build of Open3D.
    # The script will fall back to CPU if CUDA is not available.
    gpu_command = [
        "python", "ICP/demo.py",
        src_path,
        tgt_path,
        "--voxel", "0.05",
        "--use-gpu"
    ]
    run_command(gpu_command)
    
    # --- Test Case 3: CPU Registration without Global Alignment ---
    # This uses the --no-global flag, relying on simple centroid alignment
    # as the initial guess. This is faster but may fail if the clouds are far apart.
    no_global_command = [
        "python", "ICP/demo.py",
        src_path,
        tgt_path,
        "--voxel", "0.05",
        "--no-global"
    ]
    run_command(no_global_command)

    print("\n--- All tests completed. ---")
    print("You can now run the commands manually in your terminal to experiment with different parameters.")
    print(f"Example: python ICP/demo.py {src_path} {tgt_path} --voxel 0.02 --use-gpu")


if __name__ == "__main__":
    # Ensure the script is run from the root directory of the project
    if not os.path.exists("ICP/demo.py"):
        print("Error: This script must be run from the parent directory of 'ICP'.")
        print("Please change your directory and run as 'python ICP/run_test.py'")
    else:
        main()
