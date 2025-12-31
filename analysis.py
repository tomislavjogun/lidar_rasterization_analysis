import numpy as np
import pandas as pd
import laspy
from scipy.stats import binned_statistic_2d
import os

def generate_sample_building(type='gable', width=10, length=15, eave_h=3, ridge_h=6, 
                             density=8, buffer=5, sigma=0.03, outlier_prob=0.001):
    """
    Generates a synthetic building with Gaussian noise and extreme outliers.
    outlier_prob: 0.001 represents 0.1% frequency of 'high outliers' (e.g., birds/noise).
    """
    total_w, total_l = width + (2 * buffer), length + (2 * buffer)
    num_points = int(total_w * total_l * density)
    
    x = np.random.uniform(0, total_w, num_points)
    y = np.random.uniform(0, total_l, num_points)
    z = np.zeros(num_points)
    
    # Mask for building footprint
    mask = (x >= buffer) & (x <= buffer + width) & (y >= buffer) & (y <= buffer + length)
    xi, yi = x[mask] - buffer, y[mask] - buffer
    
    # Define geometries and analytical volumes
    if type == 'flat':
        z[mask] = ridge_h
        v_true = width * length * ridge_h
    elif type == 'gable':
        v_true = (width * length * eave_h) + (0.5 * width * (ridge_h - eave_h) * length)
        dist_from_center = np.abs(xi - width/2)
        z[mask] = eave_h + (ridge_h - eave_h) * (1 - (dist_from_center / (width/2)))
    elif type == 'hip':
        # Hip roof: pyramid-like slope from all sides
        v_true = (width * length * eave_h) + (1/3 * width * length * (ridge_h - eave_h))
        dist_x, dist_y = np.abs(xi - width/2) / (width/2), np.abs(yi - length/2) / (length/2)
        z[mask] = eave_h + (ridge_h - eave_h) * (1 - np.maximum(dist_x, dist_y))

    # Add standard sensor noise (Gaussian)
    z += np.random.normal(0, sigma, num_points)

    # Add Gross Outliers (High noise)
    num_outliers = int(num_points * outlier_prob)
    if num_outliers > 0:
        outlier_indices = np.random.choice(np.arange(num_points), num_outliers, replace=False)
        # Random heights 1 to 10 meters above the actual roof/ground
        z[outlier_indices] += np.random.uniform(1, 10, num_outliers)

    return x, y, z, v_true

def export_to_las(x, y, z, filename="output.las"):
    """Exports the generated point cloud to a LAS file for visual inspection."""
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.offsets = [np.min(x), np.min(y), np.min(z)]
    header.scales = [0.001, 0.001, 0.001]
    
    las = laspy.LasData(header)
    las.x, las.y, las.z = x, y, z
    las.write(filename)
    print(f"--- Exported: {filename} ---")

def run_comprehensive_benchmark(iterations=100):
    """Runs a Monte Carlo simulation across different LiDAR densities and raster resolutions."""
    # Reduced default iterations to 50 for speed, as we regenerate points every time now
    resolutions = [0.25, 0.5, 1.0, 2.0]
    densities = [2, 4, 8, 16]
    building_types = ['flat', 'gable', 'hip']
    
    # Statistics to test
    stats_to_test = {
        'Max': 'max', 
        'Mean': 'mean',
        'Median': 'median', 
        'P90': lambda x: np.percentile(x, 90) if len(x) > 0 else np.nan,
        'P95': lambda x: np.percentile(x, 95) if len(x) > 0 else np.nan
    }
    
    results_master = []

    for b_type in building_types:
        print(f"Processing Building Type: {b_type}...")
        for dens in densities:
            for res in resolutions:
                for i in range(iterations):
                    # This ensures that each iteration has a different random set of outliers.
                    x, y, z, v_true = generate_sample_building(type=b_type, density=dens, outlier_prob=0.001)
                    
                    # Introduce random grid offset to simulate real-world capture variation
                    off_x, off_y = np.random.uniform(0, res), np.random.uniform(0, res)
                    xs, ys = x + off_x, y + off_y
                    
                    x_bins = np.arange(np.min(xs), np.max(xs) + res, res)
                    y_bins = np.arange(np.min(ys), np.max(ys) + res, res)
                    
                    for name, func in stats_to_test.items():
                        # Perform 2D Binning
                        bin_stat, _, _, _ = binned_statistic_2d(xs, ys, z, statistic=func, bins=[x_bins, y_bins])
                        
                        # Calculate volume (thresholded at 0.5m to isolate building from ground)
                        v_est = np.nansum(bin_stat[bin_stat > 0.5]) * (res**2)
                        rve = ((v_est - v_true) / v_true) * 100
                        
                        results_master.append({
                            'Type': b_type,
                            'Density': dens,
                            'Res': res,
                            'Stat': name,
                            'RVE': rve
                        })
    
    return pd.DataFrame(results_master)

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # 1. Generate visual samples (LAS files)
    print("Generating sample LAS files for visual validation...")
    for b_type in ['flat', 'gable', 'hip']:
        # Generate high-density sample to see outliers clearly
        xv, yv, zv, vt = generate_sample_building(type=b_type, density=16, outlier_prob=0.001)
        export_to_las(xv, yv, zv, f"synthetic_{b_type}_with_outliers.las")

    # 2. Run statistical simulation
    print("\nStarting Monte Carlo Simulation...")
    # Using 30 iterations to save time (since we regenerate points now). 
    # Increase to 100 if you have time.
    df_results = run_comprehensive_benchmark(iterations=100) 

    # 3. Create Summary Tables
    # We group by Density as well to see its impact
    summary_median = df_results.groupby(['Type', 'Res', 'Density', 'Stat'])['RVE'].median().unstack().round(3)
    summary_std = df_results.groupby(['Type', 'Res', 'Density', 'Stat'])['RVE'].std().unstack().round(3)

    # Save results
    summary_median.to_csv("lidar_accuracy_median.csv")
    summary_std.to_csv("lidar_precision_std.csv")
    
    print("\n--- MEDIAN ERROR (ACCURACY %) ---")
    print("Shows systematic bias (e.g., Max always overestimates)")
    print(summary_median)
    
    print("\n--- STANDARD DEVIATION (PRECISION %) ---")
    print("Shows reliability. Higher number = Unsafe method (highly sensitive to outliers)")
    print(summary_std)
    
    print("\nSimulation results saved to CSV files.")
