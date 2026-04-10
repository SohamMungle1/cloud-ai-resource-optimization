"""
Resource Optimization Engine
============================
Optimizes cloud resource allocation based on predicted demand.
"""

import numpy as np


def format_indian_currency(amount):
    """
    Format number in Indian numbering system (lakhs/crores).
    
    Examples:
        554426 → "5,54,426"
        1000000 → "10,00,000"
        1234567 → "12,34,567"
    
    Parameters:
    -----------
    amount : float
        Amount to format
    
    Returns:
    --------
    str : Formatted string with Indian numbering
    """
    # Convert to integer for formatting
    amount_int = int(round(amount))
    amount_str = str(abs(amount_int))
    
    # Handle negative numbers
    negative = amount_int < 0
    
    # Apply Indian numbering system (first 3 digits from right, then groups of 2)
    if len(amount_str) <= 3:
        result = amount_str
    else:
        # Last 3 digits
        result = amount_str[-3:]
        # Remaining digits in groups of 2 from right to left
        remaining = amount_str[:-3]
        # Process remaining digits in groups of 2
        i = len(remaining)
        while i > 0:
            start = max(0, i - 2)
            result = remaining[start:i] + ',' + result
            i = start
    
    if negative:
        result = '-' + result
    
    return result


class ResourceOptimizer:
    """
    Optimizes CPU and RAM allocation based on predicted workload.
    """
    
    def __init__(self, 
                 cpu_overhead=0.1,      # 10% overhead buffer
                 ram_overhead=0.1,      # 10% overhead buffer
                 min_utilization=0.5,   # Minimum 50% utilization target
                 max_utilization=0.85,  # Maximum 85% utilization (avoid overload)
                 cost_cpu_per_unit=89.83, # Cost per CPU unit
                 cost_ram_per_unit=44.92, # Cost per RAM unit
                 base_cost_per_instance=20.0, # Minimum running cost per instance
                 idle_penalty_factor=0.5):   # Penalty for idle resources (50% of unit cost)
        """
        Initialize optimizer with configuration parameters.
        """
        self.cpu_overhead = cpu_overhead
        self.ram_overhead = ram_overhead
        self.min_utilization = min_utilization
        self.max_utilization = max_utilization
        self.cost_cpu_per_unit = cost_cpu_per_unit
        self.cost_ram_per_unit = cost_ram_per_unit
        self.base_cost_per_instance = base_cost_per_instance
        self.idle_penalty_factor = idle_penalty_factor
    
    def calculate_cost(self, allocated_cpu, allocated_ram, used_cpu, used_ram, instance_count=1):
        """
        Calculate cost with base cost and idle penalty.
        """
        # Base running cost
        base_cost = instance_count * self.base_cost_per_instance
        
        # Resource cost (allocated)
        resource_cost = (allocated_cpu * self.cost_cpu_per_unit + 
                        allocated_ram * self.cost_ram_per_unit)
        
        # Idle penalty
        idle_cpu = max(0, allocated_cpu - used_cpu)
        idle_ram = max(0, allocated_ram - used_ram)
        
        penalty = (idle_cpu * self.cost_cpu_per_unit * self.idle_penalty_factor + 
                  idle_ram * self.cost_ram_per_unit * self.idle_penalty_factor)
        
        return base_cost + resource_cost + penalty

    def optimize_allocation(self, predicted_cpu, predicted_ram, 
                          current_cpu=100, current_ram=100):
        """
        Calculate optimized resource allocation.
        """
        # Convert percentage to actual usage
        predicted_cpu_usage = predicted_cpu
        predicted_ram_usage = predicted_ram
        
        # Calculate optimal allocation with overhead buffer
        target_utilization = (self.min_utilization + self.max_utilization) / 2
        
        # Optimal CPU allocation
        optimal_cpu = (predicted_cpu_usage / target_utilization) * (1 + self.cpu_overhead)
        optimal_cpu = max(optimal_cpu, 0.1)  # Minimum 0.1 units
        
        # Optimal RAM allocation
        optimal_ram = (predicted_ram_usage / target_utilization) * (1 + self.ram_overhead)
        optimal_ram = max(optimal_ram, 0.1)  # Minimum 0.1 units
        
        # Round to reasonable precision
        optimal_cpu = round(optimal_cpu, 2)
        optimal_ram = round(optimal_ram, 2)
        
        # Calculate expected utilization
        cpu_utilization = predicted_cpu_usage / optimal_cpu if optimal_cpu > 0 else 0
        ram_utilization = predicted_ram_usage / optimal_ram if optimal_ram > 0 else 0
        
        # Clamp utilization to 0-1
        cpu_utilization = min(1.0, max(0.0, cpu_utilization))
        ram_utilization = min(1.0, max(0.0, ram_utilization))
        
        # Calculate cost using new formula
        # Note: optimize_allocation assumes 1 'logical' instance usually, or aggregate
        # For simplicity, we assume 1 aggregate instance for this calculation if not specified, 
        # but logically scaling splits this.
        # However, to be strict, we should probably pass instance count.
        # For now, let's assume 1 logical instance for the optimization step estimate.
        cost = self.calculate_cost(optimal_cpu, optimal_ram, predicted_cpu_usage, predicted_ram_usage, instance_count=1)
        
        # Calculate efficiency score
        utilization_score = (cpu_utilization + ram_utilization) / 2
        cost_efficiency = 1.0 / (1.0 + cost / 100.0)
        efficiency_score = (utilization_score * 0.7 + cost_efficiency * 0.3)
        
        return {
            'optimal_cpu': optimal_cpu,
            'optimal_ram': optimal_ram,
            'cpu_utilization': cpu_utilization,
            'ram_utilization': ram_utilization,
            'cost': cost,
            'efficiency_score': efficiency_score,
            'predicted_cpu': predicted_cpu,
            'predicted_ram': predicted_ram
        }
    
    def compare_allocations(self, before_allocation, after_allocation):
        """
        Compare two allocation strategies.
        
        Parameters:
        -----------
        before_allocation : dict
            Allocation before optimization
        after_allocation : dict
            Allocation after optimization
        
        Returns:
        --------
        dict : Comparison metrics
        """
        # Calculate improvements
        cpu_improvement = ((after_allocation['cpu_utilization'] - 
                           before_allocation['cpu_utilization']) / 
                          max(before_allocation['cpu_utilization'], 0.01)) * 100
        
        ram_improvement = ((after_allocation['ram_utilization'] - 
                           before_allocation['ram_utilization']) / 
                          max(before_allocation['ram_utilization'], 0.01)) * 100
        
        cost_savings = ((before_allocation['cost'] - after_allocation['cost']) / 
                       max(before_allocation['cost'], 0.01)) * 100
        
        efficiency_improvement = ((after_allocation['efficiency_score'] - 
                                  before_allocation['efficiency_score']) / 
                                 max(before_allocation['efficiency_score'], 0.01)) * 100
        
        # Calculate wasted resources
        before_wasted_cpu = max(0, before_allocation['optimal_cpu'] - 
                               before_allocation['predicted_cpu'])
        after_wasted_cpu = max(0, after_allocation['optimal_cpu'] - 
                              after_allocation['predicted_cpu'])
        
        before_wasted_ram = max(0, before_allocation['optimal_ram'] - 
                               before_allocation['predicted_ram'])
        after_wasted_ram = max(0, after_allocation['optimal_ram'] - 
                              after_allocation['predicted_ram'])
        
        return {
            'cpu_utilization_improvement': cpu_improvement,
            'ram_utilization_improvement': ram_improvement,
            'cost_savings_percent': cost_savings,
            'efficiency_improvement': efficiency_improvement,
            'before_wasted_cpu': before_wasted_cpu,
            'after_wasted_cpu': after_wasted_cpu,
            'before_wasted_ram': before_wasted_ram,
            'after_wasted_ram': after_wasted_ram,
            'wasted_cpu_reduction': ((before_wasted_cpu - after_wasted_cpu) / 
                                    max(before_wasted_cpu, 0.01)) * 100,
            'wasted_ram_reduction': ((before_wasted_ram - after_wasted_ram) / 
                                    max(before_wasted_ram, 0.01)) * 100
        }


if __name__ == "__main__":
    # Test optimization engine
    print("🔄 Testing optimization engine...")
    
    optimizer = ResourceOptimizer()
    
    # Example: Predict 60% CPU and 70% RAM usage
    predicted_cpu = 60.0
    predicted_ram = 70.0
    
    # Optimize allocation
    result = optimizer.optimize_allocation(
        predicted_cpu=predicted_cpu,
        predicted_ram=predicted_ram,
        current_cpu=100,
        current_ram=100
    )
    
    print("\n📊 Optimization Results:")
    print(f"   Predicted CPU: {predicted_cpu}%")
    print(f"   Predicted RAM: {predicted_ram}%")
    print(f"   Optimal CPU: {result['optimal_cpu']} units")
    print(f"   Optimal RAM: {result['optimal_ram']} units")
    print(f"   CPU Utilization: {result['cpu_utilization']*100:.2f}%")
    print(f"   RAM Utilization: {result['ram_utilization']*100:.2f}%")
    formatted_cost = format_indian_currency(result['cost'])
    print(f"   Cost: Rs {formatted_cost}")
    print(f"   Efficiency Score: {result['efficiency_score']:.4f}")
    
    print("\n✅ Optimization engine test completed!")


