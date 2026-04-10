"""
Scenario Generation Module
==========================
Defines various workload scenarios to test the robustness of the AI optimizer.
"""

import pandas as pd
import numpy as np

class WorkloadScenario:
    """
    Manages different workload scenarios for experimental evaluation.
    """
    
    @staticmethod
    def apply_scenario(data, scenario_name):
        """
        Apply a specific workload scenario to the dataset.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Base dataset
        scenario_name : str
            Name of the scenario to apply
            
        Returns:
        --------
        pd.DataFrame : Modified dataset
        """
        df = data.copy()
        
        if scenario_name == "Sudden Traffic Spike":
            return WorkloadScenario._sudden_traffic_spike(df)
        elif scenario_name == "Night-Time Low Load":
            return WorkloadScenario._night_time_low_load(df)
        elif scenario_name == "Flash Sale Scenario":
            return WorkloadScenario._flash_sale(df)
        elif scenario_name == "Gradual Growth":
            return WorkloadScenario._gradual_growth(df)
        else:
            return df

    @staticmethod
    def _sudden_traffic_spike(df):
        """
        Simulate a sudden 3x spike in traffic in the middle of the timeline.
        """
        print("   >>> Applying 'Sudden Traffic Spike' pattern...")
        mid_point = len(df) // 2
        # Spike lasts for 10% of the duration
        duration = len(df) // 10
        
        # Apply spike (x3 multiplier)
        df.iloc[mid_point:mid_point+duration, df.columns.get_loc('cpu_usage')] *= 3.0
        df.iloc[mid_point:mid_point+duration, df.columns.get_loc('ram_usage')] *= 2.5
        df.iloc[mid_point:mid_point+duration, df.columns.get_loc('requests')] *= 3.0
        
        # Clip to max values
        df['cpu_usage'] = df['cpu_usage'].clip(upper=100)
        df['ram_usage'] = df['ram_usage'].clip(upper=100)
        
        return df

    @staticmethod
    def _night_time_low_load(df):
        """
        Simulate a period of very low activity (e.g., system maintenance or holiday).
        """
        print("   >>> Applying 'Night-Time Low Load' pattern...")
        # Scale down entire dataset to 20% of original load
        df['cpu_usage'] *= 0.2
        df['ram_usage'] *= 0.25
        df['requests'] = (df['requests'] * 0.2).astype(int)
        
        return df

    @staticmethod
    def _flash_sale(df):
        """
        Simulate a Flash Sale: Short, extreme burst of traffic followed by return to normal.
        """
        print("   >>> Applying 'Flash Sale' pattern...")
        # Create 3 short bursts
        burst_points = [len(df)//4, len(df)//2, 3*len(df)//4]
        duration = len(df) // 20  # Short duration
        
        for point in burst_points:
            # Extreme spike (x4)
            df.iloc[point:point+duration, df.columns.get_loc('cpu_usage')] = 95.0 + np.random.normal(0, 2, duration)
            df.iloc[point:point+duration, df.columns.get_loc('ram_usage')] = 90.0 + np.random.normal(0, 2, duration)
            df.iloc[point:point+duration, df.columns.get_loc('requests')] *= 5.0
            
        # Clip
        df['cpu_usage'] = df['cpu_usage'].clip(upper=100)
        df['ram_usage'] = df['ram_usage'].clip(upper=100)
        
        return df

    @staticmethod
    def _gradual_growth(df):
        """
        Simulate Gradual Growth: Linear increase in base load over time.
        """
        print("   >>> Applying 'Gradual Growth' pattern...")
        n = len(df)
        # Linear multiplier from 1.0 to 2.5 over the timeline
        growth_factor = np.linspace(1.0, 2.5, n)
        
        df['cpu_usage'] *= growth_factor
        df['ram_usage'] *= growth_factor
        df['requests'] = (df['requests'] * growth_factor).astype(int)
        
        # Clip
        df['cpu_usage'] = df['cpu_usage'].clip(upper=100)
        df['ram_usage'] = df['ram_usage'].clip(upper=100)
        
        return df
