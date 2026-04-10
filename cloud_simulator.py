"""
Cloud Simulation Layer
======================
Simulates cloud environment and compares resource utilization before/after AI optimization.
"""

import numpy as np
import pandas as pd
from optimization_engine import ResourceOptimizer, format_indian_currency
from autoscaler import AutoScaler
from prediction import WorkloadPredictor
from preprocessing import DataPreprocessor
from decision_logger import DecisionLogger


class CloudSimulator:
    """
    Simulates cloud infrastructure and evaluates optimization strategies.
    """
    
    def __init__(self, optimizer=None, autoscaler=None):
        """
        Initialize cloud simulator.
        
        Parameters:
        -----------
        optimizer : ResourceOptimizer
            Resource optimization engine
        autoscaler : AutoScaler
            Auto-scaling decision engine
        """
        self.optimizer = optimizer or ResourceOptimizer()
        self.autoscaler = autoscaler or AutoScaler()
        self.results = []
    
    def simulate_baseline(self, data, fixed_cpu=100, fixed_ram=100, sla_threshold=0.9):
        """
        Simulate baseline scenario (fixed allocation, no AI optimization).
         SLA Violation = CPU Usage > SLA Threshold
        """
        results = {
            'cpu_utilizations': [],
            'ram_utilizations': [],
            'wasted_cpu': [],
            'wasted_ram': [],
            'costs': [],
            'efficiency_scores': [],
            'sla_violations': 0
        }
        
        for _, row in data.iterrows():
            cpu_usage = row['cpu_usage']
            ram_usage = row['ram_usage']
            
            # Calculate utilization
            cpu_util = cpu_usage / fixed_cpu if fixed_cpu > 0 else 0
            ram_util = ram_usage / fixed_ram if fixed_ram > 0 else 0
            
            cpu_util = min(1.0, max(0.0, cpu_util))
            ram_util = min(1.0, max(0.0, ram_util))
            
            # Track SLA violation
            if cpu_util > sla_threshold:
                results['sla_violations'] += 1
            
            # Calculate wasted resources
            wasted_cpu = max(0, fixed_cpu - cpu_usage)
            wasted_ram = max(0, fixed_ram - ram_usage)
            
            # Calculate cost
            # Baseline assumes 1 big fixed instance or multiple fixed instances summing to fixed_cpu
            instance_count = max(1, int(fixed_cpu / 60.0)) # Estimating instances based on new base unit
            
            cost = self.optimizer.calculate_cost(
                allocated_cpu=fixed_cpu,
                allocated_ram=fixed_ram,
                used_cpu=cpu_usage,
                used_ram=ram_usage,
                instance_count=instance_count
            )
            
            # Efficiency score
            avg_util = (cpu_util + ram_util) / 2
            cost_eff = 1.0 / (1.0 + cost / 100.0)
            efficiency = (avg_util * 0.7 + cost_eff * 0.3)
            
            results['cpu_utilizations'].append(cpu_util)
            results['ram_utilizations'].append(ram_util)
            results['wasted_cpu'].append(wasted_cpu)
            results['wasted_ram'].append(wasted_ram)
            results['costs'].append(cost)
            results['efficiency_scores'].append(efficiency)
        
        # Calculate summary statistics
        results['avg_cpu_utilization'] = np.mean(results['cpu_utilizations'])
        results['avg_ram_utilization'] = np.mean(results['ram_utilizations'])
        results['avg_wasted_cpu'] = np.mean(results['wasted_cpu'])
        results['avg_wasted_ram'] = np.mean(results['wasted_ram'])
        results['total_cost'] = np.sum(results['costs'])
        results['avg_efficiency'] = np.mean(results['efficiency_scores'])
        results['sla_violation_rate'] = (results['sla_violations'] / len(data)) * 100
        
        return results
    
    def simulate_threshold_scaling(self, data, scale_up_threshold=None, scale_down_threshold=0.40, sla_threshold=0.9):
        """
        Simulate threshold-based scaling (non-AI simple reactive logic).
        """
        results = {
            'cpu_utilizations': [],
            'ram_utilizations': [],
            'wasted_cpu': [],
            'wasted_ram': [],
            'costs': [],
            'efficiency_scores': [],
            'allocations': [],
            'sla_violations': 0
        }
        
        # Use autoscaler defaults if not provided
        if scale_up_threshold is None:
            scale_up_threshold = self.autoscaler.scale_up_threshold
            
        # Start with 1 instance
        current_instances = 1
        base_cpu_unit = 60.0
        base_ram_unit = 60.0
        
        for _, row in data.iterrows():
            # Get current demand
            cpu_demand = row['cpu_usage']
            ram_demand = row['ram_usage']
            
            # 1. Calculate current utilization (before scaling)
            current_capacity_cpu = current_instances * base_cpu_unit
            current_capacity_ram = current_instances * base_ram_unit
            
            util_cpu = cpu_demand / current_capacity_cpu
            util_ram = ram_demand / current_capacity_ram
            max_util = max(util_cpu, util_ram)
            
            # 2. Apply scaling rules for NEXT step (or immediate reaction depending on model)
            # Here we assume reactive: capacity adjusts for current load if it violates constraints,
            # or we simulate the "reaction" to the previous step. 
            # For simplicity and fairness with AI (which predicts ahead), 
            # we'll implement this as "Reactive": Check utilization -> Scale -> Recalculate cost
            
            # Simple Threshold Logic
            if max_util > scale_up_threshold:
                # Scale up
                current_instances = min(self.autoscaler.max_instances, current_instances + 1)
            elif max_util < scale_down_threshold and current_instances > self.autoscaler.min_instances:
                # Scale down
                current_instances = max(self.autoscaler.min_instances, current_instances - 1)
            
            # 3. Calculate metrics with (potentially new) instance count
            final_capacity_cpu = current_instances * base_cpu_unit
            final_capacity_ram = current_instances * base_ram_unit
            
            final_util_cpu = min(1.0, max(0.0, cpu_demand / final_capacity_cpu))
            final_util_ram = min(1.0, max(0.0, ram_demand / final_capacity_ram))
            
            wasted_cpu = max(0, final_capacity_cpu - cpu_demand)
            wasted_ram = max(0, final_capacity_ram - ram_demand)
            
            cost = self.optimizer.calculate_cost(
                allocated_cpu=final_capacity_cpu,
                allocated_ram=final_capacity_ram,
                used_cpu=cpu_demand,
                used_ram=ram_demand,
                instance_count=current_instances
            )
            
            avg_util = (final_util_cpu + final_util_ram) / 2
            cost_eff = 1.0 / (1.0 + cost / 100.0)
            efficiency = (avg_util * 0.7 + cost_eff * 0.3)
            
            # Track SLA violation
            if final_util_cpu > sla_threshold:
                results['sla_violations'] += 1
                
            results['cpu_utilizations'].append(final_util_cpu)
            results['ram_utilizations'].append(final_util_ram)
            results['wasted_cpu'].append(wasted_cpu)
            results['wasted_ram'].append(wasted_ram)
            results['costs'].append(cost)
            results['efficiency_scores'].append(efficiency)
            results['allocations'].append({
                'cpu': final_capacity_cpu,
                'ram': final_capacity_ram,
                'instances': current_instances
            })
            
        # Calculate summary statistics
        results['avg_cpu_utilization'] = np.mean(results['cpu_utilizations'])
        results['avg_ram_utilization'] = np.mean(results['ram_utilizations'])
        results['avg_wasted_cpu'] = np.mean(results['wasted_cpu'])
        results['avg_wasted_ram'] = np.mean(results['wasted_ram'])
        results['total_cost'] = np.sum(results['costs'])
        results['avg_efficiency'] = np.mean(results['efficiency_scores'])
        results['sla_violation_rate'] = (results['sla_violations'] / len(data)) * 100
        
        return results
    
    def simulate_ai_optimized(self, data, predictor, preprocessor, 
                            lookback_window=24, decision_logger=None, sla_threshold=0.9):
        """
        Simulate AI-optimized scenario (dynamic allocation based on predictions).
        """
        results = {
            'cpu_utilizations': [],
            'ram_utilizations': [],
            'wasted_cpu': [],
            'wasted_ram': [],
            'costs': [],
            'efficiency_scores': [],
            'allocations': [],
            'sla_violations': 0
        }
        
        # Track current instance count for autoscaling
        current_instances = 1
        current_cpu_per_instance = 100.0
        current_ram_per_instance = 100.0
        
        # Start from lookback_window to have enough history
        for i in range(lookback_window, len(data)):
            # Get recent history
            recent_data = data.iloc[i-lookback_window:i]
            
            try:
                # Predict next step
                prediction = predictor.predict_next_step(recent_data, preprocessor)
                
                # Make scaling decision using autoscaler
                scaling_decision = self.autoscaler.decide_scaling_action(
                    predicted_cpu=prediction['cpu_usage'],
                    predicted_ram=prediction['ram_usage'],
                    current_cpu_allocation=current_cpu_per_instance,
                    current_ram_allocation=current_ram_per_instance,
                    current_instances=current_instances
                )
                
                # Update instance count based on decision
                current_instances = scaling_decision['new_instances']
                
                # Log decision if logger provided
                if decision_logger is not None:
                    timestamp = data.iloc[i]['timestamp'] if 'timestamp' in data.columns else None
                    decision_logger.log_decision(
                        step=i - lookback_window + 1,  # Step number starting from 1
                        predicted_cpu=prediction['cpu_usage'],
                        predicted_ram=prediction['ram_usage'],
                        sla_threshold=self.autoscaler.scale_up_threshold,
                        decision_dict=scaling_decision,
                        timestamp=timestamp
                    )
                
                # Optimize allocation based on predicted demand
                optimization = self.optimizer.optimize_allocation(
                    predicted_cpu=prediction['cpu_usage'],
                    predicted_ram=prediction['ram_usage']
                )
                
                # Adjust allocation per instance based on scaling
                if current_instances > 0:
                    current_cpu_per_instance = optimization['optimal_cpu'] / current_instances
                    current_ram_per_instance = optimization['optimal_ram'] / current_instances
                
                # Get actual usage
                actual_cpu = data.iloc[i]['cpu_usage']
                actual_ram = data.iloc[i]['ram_usage']
                
                # Calculate utilization with optimized allocation
                cpu_util = actual_cpu / optimization['optimal_cpu'] if optimization['optimal_cpu'] > 0 else 0
                ram_util = actual_ram / optimization['optimal_ram'] if optimization['optimal_ram'] > 0 else 0
                
                cpu_util = min(1.0, max(0.0, cpu_util))
                ram_util = min(1.0, max(0.0, ram_util))
                
                # Calculate wasted resources
                wasted_cpu = max(0, optimization['optimal_cpu'] - actual_cpu)
                wasted_ram = max(0, optimization['optimal_ram'] - actual_ram)
                
                # Track SLA violation
                if cpu_util > sla_threshold:
                    results['sla_violations'] += 1
                
                results['cpu_utilizations'].append(cpu_util)
                results['ram_utilizations'].append(ram_util)
                results['wasted_cpu'].append(wasted_cpu)
                results['wasted_ram'].append(wasted_ram)
                # Calculate real cost (using actual usage for idle penalty)
                real_cost = self.optimizer.calculate_cost(
                    allocated_cpu=optimization['optimal_cpu'],
                    allocated_ram=optimization['optimal_ram'],
                    used_cpu=actual_cpu,
                    used_ram=actual_ram,
                    instance_count=current_instances if current_instances > 0 else 1
                )
                
                results['costs'].append(real_cost)
                results['efficiency_scores'].append(optimization['efficiency_score'])
                results['allocations'].append({
                    'cpu': optimization['optimal_cpu'],
                    'ram': optimization['optimal_ram']
                })
            
            except Exception as e:
                # If prediction fails, use baseline
                print(f"⚠️  Prediction failed at step {i}: {e}")
                # Use simple heuristic
                actual_cpu = data.iloc[i]['cpu_usage']
                actual_ram = data.iloc[i]['ram_usage']
                optimization = self.optimizer.optimize_allocation(
                    predicted_cpu=actual_cpu,
                    predicted_ram=actual_ram
                )
                
                cpu_util = (actual_cpu / 100.0) / optimization['optimal_cpu'] if optimization['optimal_cpu'] > 0 else 0
                ram_util = (actual_ram / 100.0) / optimization['optimal_ram'] if optimization['optimal_ram'] > 0 else 0
                
                results['cpu_utilizations'].append(min(1.0, max(0.0, cpu_util)))
                results['ram_utilizations'].append(min(1.0, max(0.0, ram_util)))
                results['wasted_cpu'].append(max(0, optimization['optimal_cpu'] - actual_cpu/100.0))
                results['wasted_ram'].append(max(0, optimization['optimal_ram'] - actual_ram/100.0))
                results['costs'].append(optimization['cost'])
                results['efficiency_scores'].append(optimization['efficiency_score'])
                results['allocations'].append({
                    'cpu': optimization['optimal_cpu'],
                    'ram': optimization['optimal_ram']
                })
        
        # Calculate summary statistics
        results['avg_cpu_utilization'] = np.mean(results['cpu_utilizations'])
        results['avg_ram_utilization'] = np.mean(results['ram_utilizations'])
        results['avg_wasted_cpu'] = np.mean(results['wasted_cpu'])
        results['avg_wasted_ram'] = np.mean(results['wasted_ram'])
        results['total_cost'] = np.sum(results['costs'])
        results['avg_efficiency'] = np.mean(results['efficiency_scores'])
        results['sla_violation_rate'] = (results['sla_violations'] / len(data)) * 100
        
        return results
    
    def compare_strategies(self, baseline_results, ai_results):
        """
        Compare baseline and AI-optimized strategies.
        
        Parameters:
        -----------
        baseline_results : dict
            Baseline simulation results
        ai_results : dict
            AI-optimized simulation results
        
        Returns:
        --------
        dict : Comparison metrics
        """
        comparison = {
            'cpu_utilization_improvement': (
                (ai_results['avg_cpu_utilization'] - baseline_results['avg_cpu_utilization']) /
                max(baseline_results['avg_cpu_utilization'], 0.01) * 100
            ),
            'ram_utilization_improvement': (
                (ai_results['avg_ram_utilization'] - baseline_results['avg_ram_utilization']) /
                max(baseline_results['avg_ram_utilization'], 0.01) * 100
            ),
            'wasted_cpu_reduction': (
                (baseline_results['avg_wasted_cpu'] - ai_results['avg_wasted_cpu']) /
                max(baseline_results['avg_wasted_cpu'], 0.01) * 100
            ),
            'wasted_ram_reduction': (
                (baseline_results['avg_wasted_ram'] - ai_results['avg_wasted_ram']) /
                max(baseline_results['avg_wasted_ram'], 0.01) * 100
            ),
            'cost_reduction': (
                (baseline_results['total_cost'] - ai_results['total_cost']) /
                max(baseline_results['total_cost'], 0.01) * 100
            ),
            'efficiency_improvement': (
                (ai_results['avg_efficiency'] - baseline_results['avg_efficiency']) /
                max(baseline_results['avg_efficiency'], 0.01) * 100
            )
        }
        
        return comparison
    
    def generate_report(self, baseline_results, ai_results, comparison):
        """
        Generate a text report of simulation results.
        
        Parameters:
        -----------
        baseline_results : dict
            Baseline simulation results
        ai_results : dict
            AI-optimized simulation results
        comparison : dict
            Comparison metrics
        
        Returns:
        --------
        str : Formatted report
        """
        report = "\n" + "="*70 + "\n"
        report += "CLOUD RESOURCE OPTIMIZATION - EXPERIMENTAL EVALUATION REPORT\n"
        report += "="*70 + "\n\n"
        
        report += "📊 BASELINE STRATEGY (Fixed Allocation)\n"
        report += "-"*70 + "\n"
        report += f"Average CPU Utilization: {baseline_results['avg_cpu_utilization']*100:.2f}%\n"
        report += f"Average RAM Utilization: {baseline_results['avg_ram_utilization']*100:.2f}%\n"
        report += f"Average Wasted CPU: {baseline_results['avg_wasted_cpu']:.2f} units\n"
        report += f"Average Wasted RAM: {baseline_results['avg_wasted_ram']:.2f} units\n"
        formatted_baseline_cost = format_indian_currency(baseline_results['total_cost'])
        report += f"Total Cost: Rs {formatted_baseline_cost}\n"
        report += f"Average Efficiency Score: {baseline_results['avg_efficiency']:.4f}\n"
        report += f"SLA Violation Rate: {baseline_results.get('sla_violation_rate', 0):.2f}%\n\n"
        
        report += "🤖 AI-OPTIMIZED STRATEGY (Dynamic Allocation)\n"
        report += "-"*70 + "\n"
        report += f"Average CPU Utilization: {ai_results['avg_cpu_utilization']*100:.2f}%\n"
        report += f"Average RAM Utilization: {ai_results['avg_ram_utilization']*100:.2f}%\n"
        report += f"Average Wasted CPU: {ai_results['avg_wasted_cpu']:.2f} units\n"
        report += f"Average Wasted RAM: {ai_results['avg_wasted_ram']:.2f} units\n"
        formatted_ai_cost = format_indian_currency(ai_results['total_cost'])
        report += f"Total Cost: Rs {formatted_ai_cost}\n"
        report += f"Average Efficiency Score: {ai_results['avg_efficiency']:.4f}\n"
        report += f"SLA Violation Rate: {ai_results.get('sla_violation_rate', 0):.2f}%\n\n"
        
        report += "📈 IMPROVEMENT METRICS\n"
        report += "-"*70 + "\n"
        report += f"CPU Utilization Improvement: {comparison['cpu_utilization_improvement']:+.2f}%\n"
        report += f"RAM Utilization Improvement: {comparison['ram_utilization_improvement']:+.2f}%\n"
        report += f"Wasted CPU Reduction: {comparison['wasted_cpu_reduction']:+.2f}%\n"
        report += f"Wasted RAM Reduction: {comparison['wasted_ram_reduction']:+.2f}%\n"
        report += f"Cost Reduction: {comparison['cost_reduction']:+.2f}%\n"
        report += f"Efficiency Improvement: {comparison['efficiency_improvement']:+.2f}%\n"
        report += "="*70 + "\n"
        
        return report


if __name__ == "__main__":
    # Test cloud simulator
    print("🔄 Testing cloud simulator...")
    
    simulator = CloudSimulator()
    
    # Create sample data
    sample_data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=100, freq='15min'),
        'cpu_usage': np.random.uniform(20, 80, 100),
        'ram_usage': np.random.uniform(25, 75, 100),
        'requests': np.random.randint(50, 500, 100)
    })
    
    # Simulate baseline
    baseline = simulator.simulate_baseline(sample_data, fixed_cpu=100, fixed_ram=100)
    
    print("\n📊 Baseline Results:")
    print(f"   Avg CPU Utilization: {baseline['avg_cpu_utilization']*100:.2f}%")
    print(f"   Avg RAM Utilization: {baseline['avg_ram_utilization']*100:.2f}%")
    formatted_cost = format_indian_currency(baseline['total_cost'])
    print(f"   Total Cost: Rs {formatted_cost}")
    
    print("\n✅ Cloud simulator test completed!")
    print("⚠️  Note: Full AI simulation requires trained model.")


