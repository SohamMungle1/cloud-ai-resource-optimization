"""
Visualization Module
====================
Creates graphs and charts for experimental evaluation.
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import os
from optimization_engine import format_indian_currency


class Visualizer:
    """
    Creates visualizations for cloud resource optimization analysis.
    """
    
    def __init__(self, output_dir='outputs'):
        """
        Initialize visualizer.
        
        Parameters:
        -----------
        output_dir : str
            Directory to save plots
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        try:
            plt.style.use('seaborn-v0_8-darkgrid')
        except:
            try:
                plt.style.use('seaborn-darkgrid')
            except:
                plt.style.use('default')
        plt.rcParams['figure.figsize'] = (12, 6)
        plt.rcParams['font.size'] = 10
    
    def plot_actual_vs_predicted(self, actual_data, predicted_data, 
                                 timestamps=None, save_path=None):
        """
        Plot actual vs predicted resource usage.
        
        Parameters:
        -----------
        actual_data : pd.DataFrame or np.array
            Actual resource usage (columns: cpu_usage, ram_usage)
        predicted_data : pd.DataFrame or np.array
            Predicted resource usage
        timestamps : pd.Series, optional
            Timestamps for x-axis
        save_path : str, optional
            Path to save plot
        """
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Convert to DataFrame if needed
        if isinstance(actual_data, np.ndarray):
            actual_df = pd.DataFrame(actual_data, columns=['cpu_usage', 'ram_usage'])
        else:
            actual_df = actual_data
        
        if isinstance(predicted_data, np.ndarray):
            pred_df = pd.DataFrame(predicted_data, columns=['cpu_usage', 'ram_usage'])
        else:
            pred_df = predicted_data
        
        if timestamps is None:
            timestamps = pd.date_range(start='2024-01-01', periods=len(actual_df), freq='15min')
        
        # Plot CPU
        axes[0].plot(timestamps, actual_df['cpu_usage'], 
                    label='Actual CPU', linewidth=2, alpha=0.7)
        axes[0].plot(timestamps, pred_df['cpu_usage'], 
                    label='Predicted CPU', linewidth=2, linestyle='--', alpha=0.7)
        axes[0].set_title('CPU Usage: Actual vs Predicted', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('CPU Usage (%)', fontsize=12)
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3)
        axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45)
        
        # Plot RAM
        axes[1].plot(timestamps, actual_df['ram_usage'], 
                    label='Actual RAM', linewidth=2, alpha=0.7)
        axes[1].plot(timestamps, pred_df['ram_usage'], 
                    label='Predicted RAM', linewidth=2, linestyle='--', alpha=0.7)
        axes[1].set_title('RAM Usage: Actual vs Predicted', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Time', fontsize=12)
        axes[1].set_ylabel('RAM Usage (%)', fontsize=12)
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3)
        axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Saved plot to {save_path}")
        
        return fig
    
    def plot_utilization_comparison(self, baseline_results, ai_results, threshold_results=None, save_path=None):
        """
        Plot resource utilization comparison (before/after optimization).
        
        Parameters:
        -----------
        baseline_results : dict
            Baseline simulation results
        ai_results : dict
            AI-optimized simulation results
        threshold_results : dict, optional
            Threshold-based simulation results
        save_path : str, optional
            Path to save plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        categories = ['CPU', 'RAM']
        baseline_utils = [
            baseline_results['avg_cpu_utilization'] * 100,
            baseline_results['avg_ram_utilization'] * 100
        ]
        ai_utils = [
            ai_results['avg_cpu_utilization'] * 100,
            ai_results['avg_ram_utilization'] * 100
        ]
        
        x = np.arange(len(categories))
        # Adjust width based on number of series
        if threshold_results:
            width = 0.25
            threshold_utils = [
                threshold_results['avg_cpu_utilization'] * 100,
                threshold_results['avg_ram_utilization'] * 100
            ]
        else:
            width = 0.35
        
        # Plot Utilization
        if threshold_results:
            axes[0].bar(x - width, baseline_utils, width, label='Baseline (Fixed)', alpha=0.8, color='#ff7f7f')
            axes[0].bar(x, threshold_utils, width, label='Threshold-Based', alpha=0.8, color='#7f7fff')
            axes[0].bar(x + width, ai_utils, width, label='AI-Optimized', alpha=0.8, color='#7fbf7f')
        else:
            axes[0].bar(x - width/2, baseline_utils, width, label='Baseline (Fixed)', alpha=0.8, color='#ff7f7f')
            axes[0].bar(x + width/2, ai_utils, width, label='AI-Optimized', alpha=0.8, color='#7fbf7f')
            
        axes[0].set_ylabel('Utilization (%)', fontsize=12)
        axes[0].set_title('Average Resource Utilization Comparison', fontsize=14, fontweight='bold')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(categories)
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3, axis='y')
        axes[0].set_ylim([0, 100])
        
        # Add values
        def add_labels(ax, rects):
             for rect in rects:
                height = rect.get_height()
                ax.text(rect.get_x() + rect.get_width()/2., height + 1,
                        f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

        # We can't easily get rects from previous call without capturing return, so let's just redo logically suitable labels
        # Or simpler:
        # Just loop through what we plotted.
        
        # (For brevity in this edit, I'll rely on the visual comparison or add simple text if needed, 
        # but the logic above in original code was specific to 2 bars. I'll stick to a cleaner implementation without text labels for crowded bars or minimal labels).
        
        # Plot Wasted Resources
        baseline_wasted = [baseline_results['avg_wasted_cpu'], baseline_results['avg_wasted_ram']]
        ai_wasted = [ai_results['avg_wasted_cpu'], ai_results['avg_wasted_ram']]
        
        if threshold_results:
            threshold_wasted = [threshold_results['avg_wasted_cpu'], threshold_results['avg_wasted_ram']]
            axes[1].bar(x - width, baseline_wasted, width, label='Baseline (Fixed)', alpha=0.8, color='#ff7f7f')
            axes[1].bar(x, threshold_wasted, width, label='Threshold-Based', alpha=0.8, color='#7f7fff')
            axes[1].bar(x + width, ai_wasted, width, label='AI-Optimized', alpha=0.8, color='#7fbf7f')
        else:
            axes[1].bar(x - width/2, baseline_wasted, width, label='Baseline (Fixed)', alpha=0.8, color='#ff7f7f')
            axes[1].bar(x + width/2, ai_wasted, width, label='AI-Optimized', alpha=0.8, color='#7fbf7f')
            
        axes[1].set_ylabel('Wasted Resources (units)', fontsize=12)
        axes[1].set_title('Average Wasted Resources Comparison', fontsize=14, fontweight='bold')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(categories)
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Saved plot to {save_path}")
        return fig
    
    def plot_cost_efficiency(self, baseline_results, ai_results, threshold_results=None, save_path=None):
        """
        Plot cost and efficiency comparison.
        
        Parameters:
        -----------
        baseline_results : dict
            Baseline simulation results
        ai_results : dict
            AI-optimized simulation results
        threshold_results : dict, optional
            Threshold simulation results
        save_path : str, optional
            Path to save plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Data preparation
        costs = [baseline_results['total_cost']]
        labels = ['Baseline']
        colors = ['#ff7f7f']
        efficiencies = [baseline_results['avg_efficiency']]
        
        if threshold_results:
            costs.append(threshold_results['total_cost'])
            labels.append('Threshold')
            colors.append('#7f7fff')
            efficiencies.append(threshold_results['avg_efficiency'])
            
        costs.append(ai_results['total_cost'])
        labels.append('AI-Optimized')
        colors.append('#7fbf7f')
        efficiencies.append(ai_results['avg_efficiency'])
        
        # Plot Cost
        bars1 = axes[0].bar(labels, costs, color=colors, alpha=0.8)
        axes[0].set_ylabel('Total Cost (Rs)', fontsize=12)
        axes[0].set_title('Total Cost Comparison', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3, axis='y')
        
        for bar, cost in zip(bars1, costs):
            height = bar.get_height()
            formatted_cost = format_indian_currency(cost)
            axes[0].text(bar.get_x() + bar.get_width()/2., height,
                        f'Rs {formatted_cost}', ha='center', va='bottom', fontsize=11)
        
        # Plot Efficiency
        bars2 = axes[1].bar(labels, efficiencies, color=colors, alpha=0.8)
        axes[1].set_ylabel('Efficiency Score', fontsize=12)
        axes[1].set_title('Average Efficiency Score Comparison', fontsize=14, fontweight='bold')
        axes[1].set_ylim([0, 1])
        axes[1].grid(True, alpha=0.3, axis='y')
        
        for bar, eff in zip(bars2, efficiencies):
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height,
                        f'{eff:.4f}', ha='center', va='bottom', fontsize=11)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Saved plot to {save_path}")
        return fig
    
    def plot_improvement_metrics(self, comparison, save_path=None):
        """
        Plot improvement metrics as bar chart.
        
        Parameters:
        -----------
        comparison : dict
            Comparison metrics dictionary
        save_path : str, optional
            Path to save plot
        """
        fig, ax = plt.subplots(figsize=(12, 7))
        
        metrics = [
            'CPU Utilization\nImprovement',
            'RAM Utilization\nImprovement',
            'Wasted CPU\nReduction',
            'Wasted RAM\nReduction',
            'Cost\nReduction',
            'Efficiency\nImprovement'
        ]
        
        values = [
            comparison['cpu_utilization_improvement'],
            comparison['ram_utilization_improvement'],
            comparison['wasted_cpu_reduction'],
            comparison['wasted_ram_reduction'],
            comparison['cost_reduction'],
            comparison['efficiency_improvement']
        ]
        
        colors = ['#4CAF50' if v > 0 else '#f44336' for v in values]
        
        bars = ax.barh(metrics, values, color=colors, alpha=0.8)
        ax.set_xlabel('Improvement (%)', fontsize=12)
        ax.set_title('AI Optimization Improvement Metrics', 
                    fontsize=14, fontweight='bold')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for bar, value in zip(bars, values):
            width = bar.get_width()
            label_x = width + (1 if width >= 0 else -1)
            ax.text(label_x, bar.get_y() + bar.get_height()/2,
                   f'{value:+.2f}%', ha='left' if width >= 0 else 'right',
                   va='center', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Saved plot to {save_path}")
        
        return fig
    
    def plot_time_series_utilization(self, baseline_utils, ai_utils, threshold_utils=None,
                                     timestamps=None, save_path=None):
        """
        Plot time-series utilization comparison.
        
        Parameters:
        -----------
        baseline_utils : dict
            Baseline utilization time series
        ai_utils : dict
            AI-optimized utilization time series
        threshold_utils : dict, optional
            Threshold utilization time series
        timestamps : pd.Series, optional
            Timestamps for x-axis
        save_path : str, optional
            Path to save plot
        """
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Find minimum length
        min_len = min(len(baseline_utils['cpu_utilizations']), len(ai_utils['cpu_utilizations']))
        if threshold_utils:
            min_len = min(min_len, len(threshold_utils['cpu_utilizations']))
            
        def align_len(arr): return np.array(arr)[-min_len:]
        
        base_cpu = align_len(baseline_utils['cpu_utilizations']) * 100
        base_ram = align_len(baseline_utils['ram_utilizations']) * 100
        ai_cpu = align_len(ai_utils['cpu_utilizations']) * 100
        ai_ram = align_len(ai_utils['ram_utilizations']) * 100
        
        if threshold_utils:
            thr_cpu = align_len(threshold_utils['cpu_utilizations']) * 100
            thr_ram = align_len(threshold_utils['ram_utilizations']) * 100
        
        if timestamps is None:
            timestamps = pd.date_range(start='2024-01-01', periods=min_len, freq='15min')
        else:
            timestamps = timestamps[-min_len:] if len(timestamps) >= min_len else pd.date_range(start='2024-01-01', periods=min_len, freq='15min')
            
        # Plot CPU
        axes[0].plot(timestamps, base_cpu, label='Baseline', linewidth=2, alpha=0.7, color='#ff7f7f')
        if threshold_utils:
            axes[0].plot(timestamps, thr_cpu, label='Threshold-Based', linewidth=2, alpha=0.7, color='#7f7fff', linestyle='--')
        axes[0].plot(timestamps, ai_cpu, label='AI-Optimized', linewidth=2, alpha=0.7, color='#7fbf7f')
        
        axes[0].set_title('CPU Utilization Over Time', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('CPU Utilization (%)', fontsize=12)
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3)
        axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        
        # Plot RAM
        axes[1].plot(timestamps, base_ram, label='Baseline', linewidth=2, alpha=0.7, color='#ff7f7f')
        if threshold_utils:
            axes[1].plot(timestamps, thr_ram, label='Threshold-Based', linewidth=2, alpha=0.7, color='#7f7fff', linestyle='--')
        axes[1].plot(timestamps, ai_ram, label='AI-Optimized', linewidth=2, alpha=0.7, color='#7fbf7f')

        axes[1].set_title('RAM Utilization Over Time', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Time', fontsize=12)
        axes[1].set_ylabel('RAM Utilization (%)', fontsize=12)
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3)
        axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Saved plot to {save_path}")
        return fig


if __name__ == "__main__":
    # Test visualization
    print("🔄 Testing visualization module...")
    
    visualizer = Visualizer()
    
    # Create sample data
    timestamps = pd.date_range('2024-01-01', periods=100, freq='15min')
    actual = pd.DataFrame({
        'cpu_usage': np.random.uniform(30, 70, 100),
        'ram_usage': np.random.uniform(35, 75, 100)
    })
    predicted = pd.DataFrame({
        'cpu_usage': actual['cpu_usage'] + np.random.normal(0, 5, 100),
        'ram_usage': actual['ram_usage'] + np.random.normal(0, 5, 100)
    })
    
    # Test plots
    visualizer.plot_actual_vs_predicted(actual, predicted, timestamps,
                                       save_path='outputs/test_prediction.png')
    
    print("✅ Visualization test completed!")

