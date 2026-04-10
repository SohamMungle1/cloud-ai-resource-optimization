"""
Data Simulation Module
======================
Generates synthetic time-series data for cloud resource usage simulation.
Simulates CPU usage, RAM usage, and incoming requests with realistic patterns.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os


def generate_synthetic_cloud_data(
    num_days=30,
    samples_per_hour=4,
    output_path="data/simulated_cloud_data.csv"
):
    """
    Generate synthetic cloud resource usage data.
    
    Parameters:
    -----------
    num_days : int
        Number of days of data to generate
    samples_per_hour : int
        Number of data points per hour (e.g., 4 = every 15 minutes)
    output_path : str
        Path to save the CSV file
    
    Returns:
    --------
    pd.DataFrame : Generated dataset with columns:
        - timestamp: datetime
        - cpu_usage: CPU usage percentage (0-100)
        - ram_usage: RAM usage percentage (0-100)
        - requests: Number of incoming requests
    """
    
    # Calculate total number of samples
    total_samples = num_days * 24 * samples_per_hour
    
    # Generate timestamps (every 15 minutes if samples_per_hour=4)
    start_time = datetime(2024, 1, 1, 0, 0, 0)
    timestamps = [start_time + timedelta(minutes=60/samples_per_hour * i) 
                  for i in range(total_samples)]
    
    # Initialize arrays for resource usage
    cpu_usage = []
    ram_usage = []
    requests = []
    
    # Base patterns for realistic simulation
    np.random.seed(42)  # For reproducibility
    
    for i in range(total_samples):
        # Time-based patterns (diurnal cycle)
        hour = timestamps[i].hour
        day_of_week = timestamps[i].weekday()
        
        # Base CPU usage with daily pattern (higher during business hours)
        base_cpu = 30 + 20 * np.sin(2 * np.pi * hour / 24 - np.pi/2)
        
        # Weekend effect (lower usage)
        if day_of_week >= 5:  # Saturday or Sunday
            base_cpu *= 0.6
        
        # Add random spikes and drops
        spike_probability = 0.05  # 5% chance of spike
        if np.random.random() < spike_probability:
            spike = np.random.uniform(30, 50)
            base_cpu += spike
        
        # Add noise and ensure bounds
        cpu = base_cpu + np.random.normal(0, 5)
        cpu = np.clip(cpu, 5, 95)  # Keep between 5% and 95%
        
        # RAM usage (correlated with CPU but with some lag)
        base_ram = cpu * 0.8 + np.random.normal(0, 3)
        ram = np.clip(base_ram, 10, 90)
        
        # Requests (correlated with CPU usage)
        base_requests = (cpu / 10) * np.random.uniform(50, 150)
        # Add occasional traffic spikes
        if np.random.random() < 0.03:  # 3% chance of traffic spike
            base_requests *= np.random.uniform(2, 5)
        requests_count = int(np.clip(base_requests, 10, 1000))
        
        cpu_usage.append(round(cpu, 2))
        ram_usage.append(round(ram, 2))
        requests.append(requests_count)
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'cpu_usage': cpu_usage,
        'ram_usage': ram_usage,
        'requests': requests
    })
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"✅ Generated {total_samples} samples ({num_days} days)")
    print(f"✅ Saved to: {output_path}")
    
    return df


if __name__ == "__main__":
    # Generate sample data when run directly
    print("🔄 Generating synthetic cloud resource data...")
    df = generate_synthetic_cloud_data(
        num_days=30,
        samples_per_hour=4,
        output_path="data/simulated_cloud_data.csv"
    )
    print(f"\n📊 Dataset Summary:")
    print(df.describe())
    print(f"\n📅 Date Range: {df['timestamp'].min()} to {df['timestamp'].max()}")

