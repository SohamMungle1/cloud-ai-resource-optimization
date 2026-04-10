"""
Auto-Scaling Decision Module
=============================
Makes scaling decisions (SCALE UP / SCALE DOWN / NO ACTION) based on predictions.
"""

import numpy as np


class AutoScaler:
    """
    Auto-scaling decision engine based on predicted resource demand.
    """
    
    def __init__(self,
                 scale_up_threshold=0.75,    # Scale up if utilization > 75%
                 scale_down_threshold=0.40, # Scale down if utilization < 40%
                 min_instances=1,            # Minimum number of instances
                 max_instances=10,           # Maximum number of instances
                 scale_factor=1.5):         # Scaling factor (multiply by this)
        """
        Initialize auto-scaler.
        
        Parameters:
        -----------
        scale_up_threshold : float
            Utilization threshold to trigger scale up (0.0 to 1.0)
        scale_down_threshold : float
            Utilization threshold to trigger scale down (0.0 to 1.0)
        min_instances : int
            Minimum number of instances
        max_instances : int
            Maximum number of instances
        scale_factor : float
            Factor to scale by (e.g., 1.5 = 50% increase)
        """
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.min_instances = min_instances
        self.max_instances = max_instances
        self.scale_factor = scale_factor
    
    def decide_scaling_action(self, predicted_cpu, predicted_ram, 
                            current_cpu_allocation, current_ram_allocation,
                            current_instances=1):
        """
        Decide scaling action based on predictions.
        
        Parameters:
        -----------
        predicted_cpu : float
            Predicted CPU usage percentage
        predicted_ram : float
            Predicted RAM usage percentage
        current_cpu_allocation : float
            Current CPU allocation per instance
        current_ram_allocation : float
            Current RAM allocation per instance
        current_instances : int
            Current number of instances
        
        Returns:
        --------
        dict : Scaling decision with keys:
            - action: 'SCALE_UP', 'SCALE_DOWN', or 'NO_ACTION'
            - new_instances: New number of instances
            - reason: Explanation of decision
            - cpu_utilization: Current CPU utilization
            - ram_utilization: Current RAM utilization
        """
        # Calculate current utilization
        total_cpu_capacity = current_cpu_allocation * current_instances
        total_ram_capacity = current_ram_allocation * current_instances
        
        cpu_utilization = predicted_cpu / total_cpu_capacity if total_cpu_capacity > 0 else 0
        ram_utilization = predicted_ram / total_ram_capacity if total_ram_capacity > 0 else 0
        
        # Use maximum of CPU and RAM utilization for decision
        max_utilization = max(cpu_utilization, ram_utilization)
        
        # Clamp utilization to 0-1
        cpu_utilization = min(1.0, max(0.0, cpu_utilization))
        ram_utilization = min(1.0, max(0.0, ram_utilization))
        max_utilization = min(1.0, max(0.0, max_utilization))
        
        # Decision logic
        action = 'NO_ACTION'
        new_instances = current_instances
        reason = ""
        
        if max_utilization > self.scale_up_threshold:
            # Scale up
            action = 'SCALE_UP'
            new_instances = min(self.max_instances, 
                              int(np.ceil(current_instances * self.scale_factor)))
            reason = f"High utilization ({max_utilization*100:.1f}%) detected. Scaling up to handle increased load."
        
        elif max_utilization < self.scale_down_threshold and current_instances > self.min_instances:
            # Scale down
            action = 'SCALE_DOWN'
            new_instances = max(self.min_instances, 
                              int(np.floor(current_instances / self.scale_factor)))
            reason = f"Low utilization ({max_utilization*100:.1f}%) detected. Scaling down to reduce costs."
        
        else:
            # No action needed
            action = 'NO_ACTION'
            new_instances = current_instances
            reason = f"Utilization ({max_utilization*100:.1f}%) is within acceptable range. No scaling needed."
        
        return {
            'action': action,
            'new_instances': new_instances,
            'current_instances': current_instances,
            'reason': reason,
            'cpu_utilization': cpu_utilization,
            'ram_utilization': ram_utilization,
            'max_utilization': max_utilization,
            'predicted_cpu': predicted_cpu,
            'predicted_ram': predicted_ram
        }
    
    def calculate_resource_allocation(self, total_cpu_needed, total_ram_needed, 
                                     num_instances, cpu_per_instance=None, 
                                     ram_per_instance=None):
        """
        Calculate resource allocation per instance.
        
        Parameters:
        -----------
        total_cpu_needed : float
            Total CPU needed
        total_ram_needed : float
            Total RAM needed
        num_instances : int
            Number of instances
        cpu_per_instance : float, optional
            Fixed CPU per instance (if None, calculated)
        ram_per_instance : float, optional
            Fixed RAM per instance (if None, calculated)
        
        Returns:
        --------
        dict : Allocation details
        """
        if cpu_per_instance is None:
            cpu_per_instance = total_cpu_needed / num_instances if num_instances > 0 else 0
        
        if ram_per_instance is None:
            ram_per_instance = total_ram_needed / num_instances if num_instances > 0 else 0
        
        return {
            'cpu_per_instance': cpu_per_instance,
            'ram_per_instance': ram_per_instance,
            'total_cpu': cpu_per_instance * num_instances,
            'total_ram': ram_per_instance * num_instances,
            'num_instances': num_instances
        }


if __name__ == "__main__":
    # Test auto-scaler
    print("🔄 Testing auto-scaler...")
    
    autoscaler = AutoScaler(
        scale_up_threshold=0.75,
        scale_down_threshold=0.40,
        min_instances=1,
        max_instances=10
    )
    
    # Test scenarios
    scenarios = [
        {
            'name': 'High Load',
            'predicted_cpu': 85,
            'predicted_ram': 80,
            'current_cpu': 50,
            'current_ram': 50,
            'instances': 2
        },
        {
            'name': 'Low Load',
            'predicted_cpu': 25,
            'predicted_ram': 30,
            'current_cpu': 50,
            'current_ram': 50,
            'instances': 3
        },
        {
            'name': 'Normal Load',
            'predicted_cpu': 55,
            'predicted_ram': 50,
            'current_cpu': 50,
            'current_ram': 50,
            'instances': 2
        }
    ]
    
    for scenario in scenarios:
        decision = autoscaler.decide_scaling_action(
            predicted_cpu=scenario['predicted_cpu'],
            predicted_ram=scenario['predicted_ram'],
            current_cpu_allocation=scenario['current_cpu'],
            current_ram_allocation=scenario['current_ram'],
            current_instances=scenario['instances']
        )
        
        print(f"\n📊 Scenario: {scenario['name']}")
        print(f"   Action: {decision['action']}")
        print(f"   Instances: {decision['current_instances']} → {decision['new_instances']}")
        print(f"   CPU Utilization: {decision['cpu_utilization']*100:.1f}%")
        print(f"   RAM Utilization: {decision['ram_utilization']*100:.1f}%")
        print(f"   Reason: {decision['reason']}")
    
    print("\n✅ Auto-scaler test completed!")




