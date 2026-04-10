"""
Decision Logger Module
======================
Logs and explains each auto-scaling decision for AI transparency and explainability.
"""

import os
from datetime import datetime


class DecisionLogger:
    """
    Logs auto-scaling decisions with explanations for transparency.
    """
    
    def __init__(self, output_file="outputs/decision_timeline.txt"):
        """
        Initialize decision logger.
        
        Parameters:
        -----------
        output_file : str
            Path to save decision log file
        """
        self.output_file = output_file
        self.decisions = []
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Initialize log file with header
        self._write_header()
    
    def _write_header(self):
        """
        Write header to log file.
        """
        header = "="*80 + "\n"
        header += "AI AUTO-SCALING DECISION TIMELINE\n"
        header += "="*80 + "\n"
        header += "This log explains each scaling decision made by the AI system.\n"
        header += "Each entry shows: Step | Predicted CPU | SLA Threshold | Decision | Reason\n"
        header += "="*80 + "\n\n"
        
        with open(self.output_file, 'w', encoding='utf-8') as f:
            f.write(header)
    
    def log_decision(self, step, predicted_cpu, predicted_ram, 
                    sla_threshold, decision_dict, timestamp=None):
        """
        Log a scaling decision with explanation.
        
        Parameters:
        -----------
        step : int
            Time step number
        predicted_cpu : float
            Predicted CPU usage percentage
        predicted_ram : float
            Predicted RAM usage percentage
        sla_threshold : float
            SLA threshold (scale up threshold) in percentage
        decision_dict : dict
            Decision dictionary from AutoScaler.decide_scaling_action()
            Must contain: 'action', 'reason', 'max_utilization', etc.
        timestamp : datetime, optional
            Timestamp for this decision
        """
        # Extract decision information
        action = decision_dict.get('action', 'UNKNOWN')
        reason = decision_dict.get('reason', 'No reason provided')
        max_utilization = decision_dict.get('max_utilization', 0.0)
        current_instances = decision_dict.get('current_instances', 0)
        new_instances = decision_dict.get('new_instances', 0)
        
        # Format timestamp
        if timestamp is None:
            timestamp_str = f"Step {step}"
        else:
            timestamp_str = f"{timestamp.strftime('%Y-%m-%d %H:%M:%S')} (Step {step})"
        
        # Create log entry
        log_entry = f"[{timestamp_str}]\n"
        log_entry += f"  Predicted CPU: {predicted_cpu:.1f}%\n"
        log_entry += f"  Predicted RAM: {predicted_ram:.1f}%\n"
        log_entry += f"  Max Utilization: {max_utilization*100:.1f}%\n"
        log_entry += f"  SLA Threshold: {sla_threshold*100:.1f}%\n"
        log_entry += f"  Current Instances: {current_instances}\n"
        log_entry += f"  Decision: {action}\n"
        
        if action != 'NO_ACTION':
            log_entry += f"  New Instances: {new_instances}\n"
        
        log_entry += f"  Reason: {reason}\n"
        log_entry += "-"*80 + "\n"
        
        # Store decision for summary
        self.decisions.append({
            'step': step,
            'timestamp': timestamp,
            'predicted_cpu': predicted_cpu,
            'predicted_ram': predicted_ram,
            'max_utilization': max_utilization,
            'sla_threshold': sla_threshold,
            'action': action,
            'current_instances': current_instances,
            'new_instances': new_instances,
            'reason': reason
        })
        
        # Append to file
        with open(self.output_file, 'a', encoding='utf-8') as f:
            f.write(log_entry)
    
    def get_summary(self):
        """
        Generate summary statistics of decisions.
        
        Returns:
        --------
        dict : Summary statistics
        """
        if not self.decisions:
            return {
                'total_decisions': 0,
                'scale_ups': 0,
                'scale_downs': 0,
                'no_actions': 0,
                'scale_up_percentage': 0.0,
                'scale_down_percentage': 0.0,
                'no_action_percentage': 0.0
            }
        
        total = len(self.decisions)
        scale_ups = sum(1 for d in self.decisions if d['action'] == 'SCALE_UP')
        scale_downs = sum(1 for d in self.decisions if d['action'] == 'SCALE_DOWN')
        no_actions = sum(1 for d in self.decisions if d['action'] == 'NO_ACTION')
        
        return {
            'total_decisions': total,
            'scale_ups': scale_ups,
            'scale_downs': scale_downs,
            'no_actions': no_actions,
            'scale_up_percentage': (scale_ups / total) * 100,
            'scale_down_percentage': (scale_downs / total) * 100,
            'no_action_percentage': (no_actions / total) * 100
        }
    
    def print_summary(self):
        """
        Print summary statistics to console.
        """
        summary = self.get_summary()
        
        print("\n" + "="*70)
        print("DECISION TIMELINE SUMMARY")
        print("="*70)
        print(f"Total Decisions Logged: {summary['total_decisions']}")
        print(f"  - SCALE UP:   {summary['scale_ups']:3d} ({summary['scale_up_percentage']:5.1f}%)")
        print(f"  - SCALE DOWN: {summary['scale_downs']:3d} ({summary['scale_down_percentage']:5.1f}%)")
        print(f"  - NO ACTION:  {summary['no_actions']:3d} ({summary['no_action_percentage']:5.1f}%)")
        print(f"\n📄 Full decision log saved to: {self.output_file}")
        print("="*70 + "\n")
    
    def write_summary_to_file(self):
        """
        Append summary to log file.
        """
        summary = self.get_summary()
        
        summary_text = "\n" + "="*80 + "\n"
        summary_text += "DECISION SUMMARY\n"
        summary_text += "="*80 + "\n"
        summary_text += f"Total Decisions: {summary['total_decisions']}\n"
        summary_text += f"  SCALE UP:   {summary['scale_ups']:3d} ({summary['scale_up_percentage']:5.1f}%)\n"
        summary_text += f"  SCALE DOWN: {summary['scale_downs']:3d} ({summary['scale_down_percentage']:5.1f}%)\n"
        summary_text += f"  NO ACTION:  {summary['no_actions']:3d} ({summary['no_action_percentage']:5.1f}%)\n"
        summary_text += "="*80 + "\n"
        
        with open(self.output_file, 'a', encoding='utf-8') as f:
            f.write(summary_text)


if __name__ == "__main__":
    # Test decision logger
    print("🔄 Testing decision logger...")
    
    logger = DecisionLogger("outputs/test_decision_log.txt")
    
    # Test logging some decisions
    from autoscaler import AutoScaler
    
    autoscaler = AutoScaler()
    
    test_decisions = [
        {'predicted_cpu': 82, 'predicted_ram': 78, 'sla': 0.75},
        {'predicted_cpu': 45, 'predicted_ram': 50, 'sla': 0.75},
        {'predicted_cpu': 60, 'predicted_ram': 55, 'sla': 0.75},
    ]
    
    for i, test in enumerate(test_decisions, 1):
        decision = autoscaler.decide_scaling_action(
            predicted_cpu=test['predicted_cpu'],
            predicted_ram=test['predicted_ram'],
            current_cpu_allocation=50,
            current_ram_allocation=50,
            current_instances=2
        )
        
        logger.log_decision(
            step=i,
            predicted_cpu=test['predicted_cpu'],
            predicted_ram=test['predicted_ram'],
            sla_threshold=test['sla'],
            decision_dict=decision
        )
    
    logger.write_summary_to_file()
    logger.print_summary()
    
    print("✅ Decision logger test completed!")



