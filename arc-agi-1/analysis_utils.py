#!/usr/bin/env python3
"""
Analysis utilities for the Active Inference ARC-AGI Agent.

This module provides tools for analyzing agent performance, visualizing
learning curves, and understanding the effectiveness of different primitives.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
import seaborn as sns
from collections import defaultdict, Counter

# Set up plotting style
try:
    plt.style.use('seaborn-v0_8')
except OSError:
    try:
        plt.style.use('seaborn')
    except OSError:
        pass  # Use default style
        
try:
    sns.set_palette("husl")
except:
    pass  # Use default palette


def plot_learning_curves(agent_stats: Dict[str, Any], 
                        save_path: Optional[str] = None,
                        show_plot: bool = True) -> None:
    """
    Plot learning curves for the Active Inference agent.
    
    Args:
        agent_stats: Statistics from agent.get_stats()
        save_path: Optional path to save the plot
        show_plot: Whether to display the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Prediction errors over time
    if agent_stats.get('recent_prediction_errors'):
        errors = agent_stats['recent_prediction_errors']
        axes[0, 0].plot(errors, 'b-', alpha=0.7, linewidth=2)
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Prediction Error')
        axes[0, 0].set_title('Prediction Error Over Time')
        axes[0, 0].grid(True, alpha=0.3)
    
    # Free energy over time
    if agent_stats.get('recent_free_energies'):
        free_energies = agent_stats['recent_free_energies']
        axes[0, 1].plot(free_energies, 'r-', alpha=0.7, linewidth=2)
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Expected Free Energy')
        axes[0, 1].set_title('Expected Free Energy Over Time')
        axes[0, 1].grid(True, alpha=0.3)
    
    # Epistemic vs Pragmatic values
    if (agent_stats.get('recent_epistemic_values') and 
        agent_stats.get('recent_pragmatic_values')):
        epistemic = agent_stats['recent_epistemic_values']
        pragmatic = agent_stats['recent_pragmatic_values']
        
        steps = range(min(len(epistemic), len(pragmatic)))
        axes[1, 0].plot(steps, epistemic[:len(steps)], 'g-', 
                       label='Epistemic', alpha=0.7, linewidth=2)
        axes[1, 0].plot(steps, pragmatic[:len(steps)], 'orange', 
                       label='Pragmatic', alpha=0.7, linewidth=2)
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Value')
        axes[1, 0].set_title('Epistemic vs Pragmatic Value')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Primitive success rates
    if agent_stats.get('primitive_beliefs'):
        beliefs = agent_stats['primitive_beliefs']
        primitives = list(beliefs.keys())
        success_rates = [beliefs[p]['success_rate'] for p in primitives]
        
        # Truncate primitive names for better display
        display_names = [p.replace('_', ' ').title()[:8] for p in primitives]
        
        bars = axes[1, 1].bar(display_names, success_rates, alpha=0.7)
        axes[1, 1].set_xlabel('Primitive')
        axes[1, 1].set_ylabel('Success Rate')
        axes[1, 1].set_title('Primitive Success Rates')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        # Color bars based on success rate
        for bar, rate in zip(bars, success_rates):
            if rate > 0.7:
                bar.set_color('green')
            elif rate > 0.4:
                bar.set_color('orange')
            else:
                bar.set_color('red')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Learning curves saved to: {save_path}")
    
    if show_plot:
        plt.show()


def plot_primitive_analysis(agent_stats: Dict[str, Any],
                          save_path: Optional[str] = None,
                          show_plot: bool = True) -> None:
    """
    Plot detailed analysis of primitive usage and effectiveness.
    
    Args:
        agent_stats: Statistics from agent.get_stats()
        save_path: Optional path to save the plot
        show_plot: Whether to display the plot
    """
    if not agent_stats.get('primitive_beliefs'):
        print("No primitive beliefs data available for analysis")
        return
    
    beliefs = agent_stats['primitive_beliefs']
    primitives = list(beliefs.keys())
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Success rate vs Experience
    success_rates = [beliefs[p]['success_rate'] for p in primitives]
    experiences = [beliefs[p]['experience'] for p in primitives]
    
    scatter = axes[0, 0].scatter(experiences, success_rates, 
                               c=success_rates, cmap='RdYlGn', 
                               s=100, alpha=0.7)
    axes[0, 0].set_xlabel('Experience (Total Count)')
    axes[0, 0].set_ylabel('Success Rate')
    axes[0, 0].set_title('Success Rate vs Experience')
    axes[0, 0].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[0, 0])
    
    # Add labels for each point
    for i, primitive in enumerate(primitives):
        short_name = primitive.replace('_', ' ')[:6]
        axes[0, 0].annotate(short_name, 
                          (experiences[i], success_rates[i]),
                          xytext=(5, 5), textcoords='offset points',
                          fontsize=8, alpha=0.8)
    
    # Precision distribution
    precisions = [beliefs[p]['precision'] for p in primitives]
    axes[0, 1].hist(precisions, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 1].set_xlabel('Precision')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Precision Distribution')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Top and bottom performers
    primitive_performance = [(p, beliefs[p]['success_rate']) for p in primitives]
    primitive_performance.sort(key=lambda x: x[1], reverse=True)
    
    top_5 = primitive_performance[:5]
    bottom_5 = primitive_performance[-5:]
    
    top_names = [p[0].replace('_', ' ').title()[:10] for p in top_5]
    top_rates = [p[1] for p in top_5]
    
    bottom_names = [p[0].replace('_', ' ').title()[:10] for p in bottom_5]
    bottom_rates = [p[1] for p in bottom_5]
    
    axes[1, 0].barh(top_names, top_rates, color='green', alpha=0.7)
    axes[1, 0].set_xlabel('Success Rate')
    axes[1, 0].set_title('Top 5 Performing Primitives')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].barh(bottom_names, bottom_rates, color='red', alpha=0.7)
    axes[1, 1].set_xlabel('Success Rate')
    axes[1, 1].set_title('Bottom 5 Performing Primitives')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Primitive analysis saved to: {save_path}")
    
    if show_plot:
        plt.show()


def analyze_benchmark_results(results_file: Path) -> Dict[str, Any]:
    """
    Analyze benchmark results and generate comprehensive statistics.
    
    Args:
        results_file: Path to benchmark results JSON file
        
    Returns:
        Analysis dictionary
    """
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    results = data['results']
    
    if not results:
        return {'error': 'No results to analyze'}
    
    analysis = {
        'total_tasks': len(results),
        'correctly_solved': sum(1 for r in results if r['accuracy'] == 1.0),
        'partially_solved': sum(1 for r in results if 0 < r['accuracy'] < 1.0),
        'failed': sum(1 for r in results if r['accuracy'] == 0.0),
    }
    
    analysis['solve_rate'] = analysis['correctly_solved'] / analysis['total_tasks']
    analysis['partial_rate'] = analysis['partially_solved'] / analysis['total_tasks']
    
    # Performance statistics
    accuracies = [r['accuracy'] for r in results]
    errors = [r['mean_error'] for r in results]
    times = [r['solve_time'] for r in results]
    
    analysis['accuracy_stats'] = {
        'mean': np.mean(accuracies),
        'std': np.std(accuracies),
        'median': np.median(accuracies),
        'min': np.min(accuracies),
        'max': np.max(accuracies)
    }
    
    analysis['error_stats'] = {
        'mean': np.mean(errors),
        'std': np.std(errors),
        'median': np.median(errors),
        'min': np.min(errors),
        'max': np.max(errors)
    }
    
    analysis['time_stats'] = {
        'mean': np.mean(times),
        'std': np.std(times),
        'median': np.median(times),
        'min': np.min(times),
        'max': np.max(times)
    }
    
    # Program characteristics
    program_lengths = []
    primitive_usage = Counter()
    
    for result in results:
        if 'program' in result:
            program = result['program']
            program_lengths.append(len(program['primitives']))
            
            for primitive in program['primitives']:
                primitive_usage[primitive['type']] += 1
    
    analysis['program_length_stats'] = {
        'mean': np.mean(program_lengths) if program_lengths else 0,
        'std': np.std(program_lengths) if program_lengths else 0,
        'median': np.median(program_lengths) if program_lengths else 0,
        'distribution': dict(Counter(program_lengths))
    }
    
    analysis['primitive_usage'] = dict(primitive_usage)
    analysis['most_used_primitives'] = primitive_usage.most_common(5)
    
    return analysis


def plot_benchmark_results(results_file: Path,
                          save_path: Optional[str] = None,
                          show_plot: bool = True) -> None:
    """
    Plot comprehensive benchmark results analysis.
    
    Args:
        results_file: Path to benchmark results JSON file
        save_path: Optional path to save the plot
        show_plot: Whether to display the plot
    """
    analysis = analyze_benchmark_results(results_file)
    
    if 'error' in analysis:
        print(f"Error: {analysis['error']}")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Overall performance pie chart
    labels = ['Correctly Solved', 'Partially Solved', 'Failed']
    sizes = [analysis['correctly_solved'], analysis['partially_solved'], analysis['failed']]
    colors = ['green', 'orange', 'red']
    
    axes[0, 0].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    axes[0, 0].set_title(f'Task Solving Performance\n(Total: {analysis["total_tasks"]} tasks)')
    
    # Accuracy distribution
    with open(results_file, 'r') as f:
        data = json.load(f)
    accuracies = [r['accuracy'] for r in data['results']]
    
    axes[0, 1].hist(accuracies, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 1].set_xlabel('Accuracy')
    axes[0, 1].set_ylabel('Number of Tasks')
    axes[0, 1].set_title('Accuracy Distribution')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Solve time distribution
    times = [r['solve_time'] for r in data['results']]
    axes[0, 2].hist(times, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
    axes[0, 2].set_xlabel('Solve Time (seconds)')
    axes[0, 2].set_ylabel('Number of Tasks')
    axes[0, 2].set_title('Solve Time Distribution')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Program length distribution
    if analysis['program_length_stats']['distribution']:
        lengths = list(analysis['program_length_stats']['distribution'].keys())
        counts = list(analysis['program_length_stats']['distribution'].values())
        
        axes[1, 0].bar(lengths, counts, alpha=0.7, color='lightgreen')
        axes[1, 0].set_xlabel('Program Length')
        axes[1, 0].set_ylabel('Number of Programs')
        axes[1, 0].set_title('Program Length Distribution')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Most used primitives
    if analysis['most_used_primitives']:
        primitives, counts = zip(*analysis['most_used_primitives'])
        display_names = [p.replace('_', ' ').title()[:8] for p in primitives]
        
        axes[1, 1].bar(display_names, counts, alpha=0.7, color='gold')
        axes[1, 1].set_xlabel('Primitive')
        axes[1, 1].set_ylabel('Usage Count')
        axes[1, 1].set_title('Most Used Primitives')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
    
    # Performance correlation (accuracy vs solve time)
    axes[1, 2].scatter(times, accuracies, alpha=0.6, color='purple')
    axes[1, 2].set_xlabel('Solve Time (seconds)')
    axes[1, 2].set_ylabel('Accuracy')
    axes[1, 2].set_title('Accuracy vs Solve Time')
    axes[1, 2].grid(True, alpha=0.3)
    
    # Add correlation coefficient
    if len(times) > 1 and len(accuracies) > 1:
        correlation = np.corrcoef(times, accuracies)[0, 1]
        axes[1, 2].text(0.05, 0.95, f'Correlation: {correlation:.3f}',
                       transform=axes[1, 2].transAxes, fontsize=10,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Benchmark analysis saved to: {save_path}")
    
    if show_plot:
        plt.show()


def print_benchmark_summary(results_file: Path) -> None:
    """
    Print a comprehensive summary of benchmark results.
    
    Args:
        results_file: Path to benchmark results JSON file
    """
    analysis = analyze_benchmark_results(results_file)
    
    if 'error' in analysis:
        print(f"Error: {analysis['error']}")
        return
    
    print("\n" + "="*60)
    print("ACTIVE INFERENCE ARC-AGI BENCHMARK ANALYSIS")
    print("="*60)
    
    print(f"\nOVERALL PERFORMANCE:")
    print(f"  Total tasks: {analysis['total_tasks']}")
    print(f"  Correctly solved: {analysis['correctly_solved']} ({analysis['solve_rate']:.2%})")
    print(f"  Partially solved: {analysis['partially_solved']} ({analysis['partial_rate']:.2%})")
    print(f"  Failed: {analysis['failed']} ({(analysis['failed']/analysis['total_tasks']):.2%})")
    
    print(f"\nACCURACY STATISTICS:")
    acc_stats = analysis['accuracy_stats']
    print(f"  Mean: {acc_stats['mean']:.3f} ± {acc_stats['std']:.3f}")
    print(f"  Median: {acc_stats['median']:.3f}")
    print(f"  Range: {acc_stats['min']:.3f} - {acc_stats['max']:.3f}")
    
    print(f"\nERROR STATISTICS:")
    err_stats = analysis['error_stats']
    print(f"  Mean: {err_stats['mean']:.4f} ± {err_stats['std']:.4f}")
    print(f"  Median: {err_stats['median']:.4f}")
    print(f"  Range: {err_stats['min']:.4f} - {err_stats['max']:.4f}")
    
    print(f"\nSOLVE TIME STATISTICS:")
    time_stats = analysis['time_stats']
    print(f"  Mean: {time_stats['mean']:.2f}s ± {time_stats['std']:.2f}s")
    print(f"  Median: {time_stats['median']:.2f}s")
    print(f"  Range: {time_stats['min']:.2f}s - {time_stats['max']:.2f}s")
    
    print(f"\nPROGRAM CHARACTERISTICS:")
    prog_stats = analysis['program_length_stats']
    print(f"  Mean length: {prog_stats['mean']:.1f} ± {prog_stats['std']:.1f}")
    print(f"  Median length: {prog_stats['median']:.1f}")
    
    print(f"\nMOST USED PRIMITIVES:")
    for primitive, count in analysis['most_used_primitives']:
        percentage = (count / sum(analysis['primitive_usage'].values())) * 100
        print(f"  {primitive.replace('_', ' ').title()}: {count} ({percentage:.1f}%)")


def compare_configurations(results_files: List[Path],
                         config_names: List[str] = None,
                         save_path: Optional[str] = None,
                         show_plot: bool = True) -> None:
    """
    Compare results from different agent configurations.
    
    Args:
        results_files: List of result file paths
        config_names: Optional names for configurations
        save_path: Optional path to save the plot
        show_plot: Whether to display the plot
    """
    if config_names is None:
        config_names = [f"Config {i+1}" for i in range(len(results_files))]
    
    analyses = []
    for file in results_files:
        analyses.append(analyze_benchmark_results(file))
    
    # Extract metrics for comparison
    solve_rates = [a['solve_rate'] for a in analyses]
    mean_accuracies = [a['accuracy_stats']['mean'] for a in analyses]
    mean_errors = [a['error_stats']['mean'] for a in analyses]
    mean_times = [a['time_stats']['mean'] for a in analyses]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Solve rates comparison
    bars1 = axes[0, 0].bar(config_names, solve_rates, alpha=0.7, color='green')
    axes[0, 0].set_ylabel('Solve Rate')
    axes[0, 0].set_title('Task Solve Rate Comparison')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, rate in zip(bars1, solve_rates):
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{rate:.2%}', ha='center', va='bottom')
    
    # Mean accuracy comparison
    bars2 = axes[0, 1].bar(config_names, mean_accuracies, alpha=0.7, color='blue')
    axes[0, 1].set_ylabel('Mean Accuracy')
    axes[0, 1].set_title('Mean Accuracy Comparison')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(True, alpha=0.3)
    
    for bar, acc in zip(bars2, mean_accuracies):
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{acc:.3f}', ha='center', va='bottom')
    
    # Mean error comparison
    bars3 = axes[1, 0].bar(config_names, mean_errors, alpha=0.7, color='red')
    axes[1, 0].set_ylabel('Mean Error')
    axes[1, 0].set_title('Mean Error Comparison')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(True, alpha=0.3)
    
    for bar, err in zip(bars3, mean_errors):
        height = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.001,
                       f'{err:.3f}', ha='center', va='bottom')
    
    # Mean solve time comparison
    bars4 = axes[1, 1].bar(config_names, mean_times, alpha=0.7, color='orange')
    axes[1, 1].set_ylabel('Mean Solve Time (s)')
    axes[1, 1].set_title('Mean Solve Time Comparison')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(True, alpha=0.3)
    
    for bar, time in zip(bars4, mean_times):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                       f'{time:.1f}s', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Configuration comparison saved to: {save_path}")
    
    if show_plot:
        plt.show()


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze Active Inference ARC-AGI results")
    parser.add_argument('--results-file', type=str, required=True,
                       help='Path to benchmark results JSON file')
    parser.add_argument('--analysis-type', choices=['summary', 'plots', 'both'],
                       default='both', help='Type of analysis to perform')
    parser.add_argument('--save-plots', action='store_true',
                       help='Save plots to files')
    
    args = parser.parse_args()
    
    results_path = Path(args.results_file)
    if not results_path.exists():
        print(f"Error: Results file {results_path} not found")
        exit(1)
    
    if args.analysis_type in ['summary', 'both']:
        print_benchmark_summary(results_path)
    
    if args.analysis_type in ['plots', 'both']:
        save_path = f"benchmark_analysis_{results_path.stem}.png" if args.save_plots else None
        plot_benchmark_results(results_path, save_path=save_path)
