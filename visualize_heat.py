import numpy as np
import matplotlib.pyplot as plt

def load_heatmap(filename):
    """Loads the heatmap data from a CSV file into a NumPy array."""
    try:
        data = np.loadtxt(filename, delimiter=",")
        return data
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

def plot_heatmap(data, output_image="heatmap.png"):
    """Plots the heatmap using Matplotlib."""
    plt.figure(figsize=(8, 6))
    plt.imshow(data, cmap="hot", origin="upper", interpolation="nearest")
    plt.colorbar(label="Temperature (°C)")
    plt.title("Heat Diffusion Simulation")
    plt.xlabel("X-axis (Columns)")
    plt.ylabel("Y-axis (Rows)")
    
    # Save and show the plot
    plt.savefig(output_image, dpi=300)
    plt.show()

if __name__ == "__main__":
    filename = "heat_output_cuda.csv"  # Adjust if needed
    heatmap_data = load_heatmap(filename)
    
    if heatmap_data is not None:
        plot_heatmap(heatmap_data)
