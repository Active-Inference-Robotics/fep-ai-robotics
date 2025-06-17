#!/usr/bin/env python3
"""
Main script for running the Active Inference ARC-AGI Agent.

This script demonstrates how to use the Active Inference agent to solve
ARC-AGI tasks through program synthesis and free energy minimization.
"""

import json
import numpy as np
import argparse
import sys
import os
from pathlib import Path
from typing import List, Dict, Any
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import time

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from active_inference_arc_agent import ActiveInferenceARCAgent, Program, Primitive, PrimitiveType


def load_arc_task(task_file: Path) -> Dict[str, Any]:
    """Load a single ARC task from JSON file."""
    with open(task_file, 'r') as f:
        return json.load(f)


def evaluate_program_on_task(agent: ActiveInferenceARCAgent, 
                           task_data: Dict[str, Any],
                           program: Program) -> Dict[str, Any]:
    """Evaluate a program on all test examples of a task."""
    results = {
        'correct_predictions': 0,
        'total_test_examples': len(task_data['test']),
        'prediction_errors': [],
        'predictions': []
    }
    
    for test_example in task_data['test']:
        input_matrix = np.array(test_example['input'])
        expected_output = np.array(test_example['output'])
        
        # Generate prediction
        predicted_output = agent.predict(input_matrix, program)
        
        # Calculate error
        error = agent.prediction_error(predicted_output, expected_output)
        results['prediction_errors'].append(error)
        
        # Check if prediction is correct (exact match)
        if error < 1e-6:  # Nearly exact match
            results['correct_predictions'] += 1
        
        results['predictions'].append({
            'input': input_matrix.tolist(),
            'predicted': predicted_output.tolist(),
            'expected': expected_output.tolist(),
            'error': error
        })
    
    results['accuracy'] = results['correct_predictions'] / results['total_test_examples']
    results['mean_error'] = np.mean(results['prediction_errors'])
    
    return results


def visualize_program_performance(agent: ActiveInferenceARCAgent, 
                                task_data: Dict[str, Any],
                                program: Program,
                                save_path: str = None):
    """Visualize the performance of a program on task examples."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Show first training example
    if task_data['train']:
        train_example = task_data['train'][0]
        input_matrix = np.array(train_example['input'])
        expected_output = np.array(train_example['output'])
        predicted_output = agent.predict(input_matrix, program)
        
        axes[0, 0].imshow(input_matrix, cmap='tab10', vmin=0, vmax=9)
        axes[0, 0].set_title('Training Input')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(expected_output, cmap='tab10', vmin=0, vmax=9)
        axes[0, 1].set_title('Training Expected')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(predicted_output, cmap='tab10', vmin=0, vmax=9)
        axes[0, 2].set_title('Training Predicted')
        axes[0, 2].axis('off')
    
    # Show first test example
    if task_data['test']:
        test_example = task_data['test'][0]
        input_matrix = np.array(test_example['input'])
        expected_output = np.array(test_example['output'])
        predicted_output = agent.predict(input_matrix, program)
        
        axes[1, 0].imshow(input_matrix, cmap='tab10', vmin=0, vmax=9)
        axes[1, 0].set_title('Test Input')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(expected_output, cmap='tab10', vmin=0, vmax=9)
        axes[1, 1].set_title('Test Expected')
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(predicted_output, cmap='tab10', vmin=0, vmax=9)
        axes[1, 2].set_title('Test Predicted')
        axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    
    plt.show()


def print_program(program: Program):
    """Print a human-readable representation of a program."""
    print("Program:")
    for i, primitive in enumerate(program.primitives):
        params_str = ""
        if primitive.params:
            params_str = f" (params: {primitive.params})"
        region_str = ""
        if primitive.region:
            region_str = f" [region: {primitive.region}]"
        print(f"  {i+1}. {primitive.type.value}{params_str}{region_str}")
    print(f"Fitness: {program.fitness:.4f}")


def run_single_task(task_file: Path, agent_config: Dict[str, Any]) -> Dict[str, Any]:
    """Run the agent on a single task."""
    print(f"\nProcessing task: {task_file.name}")
    
    # Load task
    task_data = load_arc_task(task_file)
    
    # Create agent
    agent = ActiveInferenceARCAgent(**agent_config)
    
    # Solve task using training examples
    start_time = time.time()
    program = agent.solve_task(task_data['train'])
    solve_time = time.time() - start_time
    
    print_program(program)
    
    # Evaluate on test examples
    results = evaluate_program_on_task(agent, task_data, program)
    results['solve_time'] = solve_time
    results['task_file'] = task_file.name
    results['program'] = {
        'primitives': [{'type': p.type.value, 'params': p.params, 'region': p.region} 
                      for p in program.primitives],
        'fitness': program.fitness
    }
    
    print(f"Accuracy: {results['accuracy']:.2%} ({results['correct_predictions']}/{results['total_test_examples']})")
    print(f"Mean error: {results['mean_error']:.4f}")
    print(f"Solve time: {solve_time:.2f}s")
    
    return results


def run_benchmark(data_dir: Path, 
                 max_tasks: int = None, 
                 agent_config: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    """Run the agent on multiple tasks for benchmarking."""
    if agent_config is None:
        agent_config = {
            'memory_size': 50,
            'max_program_length': 4,
            'learning_rate': 0.1,
            'epistemic_weight': 0.3,
            'pragmatic_weight': 0.7
        }
    
    # Get task files
    task_files = list(data_dir.glob("*.json"))
    if max_tasks:
        task_files = task_files[:max_tasks]
    
    print(f"Running benchmark on {len(task_files)} tasks...")
    
    results = []
    correct_tasks = 0
    
    for i, task_file in enumerate(task_files):
        print(f"\n--- Task {i+1}/{len(task_files)} ---")
        
        try:
            task_result = run_single_task(task_file, agent_config)
            results.append(task_result)
            
            if task_result['accuracy'] == 1.0:
                correct_tasks += 1
                
        except Exception as e:
            print(f"Error processing {task_file}: {e}")
            continue
    
    # Summary statistics
    print(f"\n{'='*50}")
    print("BENCHMARK RESULTS")
    print(f"{'='*50}")
    print(f"Tasks solved correctly: {correct_tasks}/{len(results)} ({correct_tasks/len(results):.2%})")
    
    if results:
        accuracies = [r['accuracy'] for r in results]
        errors = [r['mean_error'] for r in results]
        times = [r['solve_time'] for r in results]
        
        print(f"Mean accuracy: {np.mean(accuracies):.3f} ± {np.std(accuracies):.3f}")
        print(f"Mean error: {np.mean(errors):.4f} ± {np.std(errors):.4f}")
        print(f"Mean solve time: {np.mean(times):.2f}s ± {np.std(times):.2f}s")
    
    return results


def demonstrate_agent_principles():
    """Demonstrate key Active Inference principles with a simple example."""
    print("\n" + "="*60)
    print("ACTIVE INFERENCE ARC AGENT DEMONSTRATION")
    print("="*60)
    
    # Create a simple synthetic task
    print("\n1. CREATING SYNTHETIC TASK:")
    print("   Input: 3x3 matrix with a pattern")
    print("   Goal: Rotate the pattern 90 degrees")
    
    input_matrix = np.array([
        [1, 0, 0],
        [1, 1, 0],
        [0, 0, 0]
    ])
    
    expected_output = np.array([
        [0, 1, 1],
        [0, 1, 0],
        [0, 0, 0]
    ])
    
    print("   Input matrix:")
    print(input_matrix)
    print("   Expected output:")
    print(expected_output)
    
    # Create agent
    agent = ActiveInferenceARCAgent(
        memory_size=20,
        max_program_length=3,
        learning_rate=0.2
    )
    
    print("\n2. AGENT CONFIGURATION:")
    print(f"   Memory size: {agent.memory_size}")
    print(f"   Max program length: {agent.max_program_length}")
    print(f"   Available primitives: {len(PrimitiveType)}")
    
    # Demonstrate expected free energy calculation
    print("\n3. EXPECTED FREE ENERGY CALCULATION:")
    
    # Test a few programs
    programs = [
        Program([Primitive(PrimitiveType.ROTATE_90)]),
        Program([Primitive(PrimitiveType.FLIP_HORIZONTAL)]),
        Program([Primitive(PrimitiveType.IDENTITY)]),
    ]
    
    for i, program in enumerate(programs):
        efe = agent.expected_free_energy(program, input_matrix, expected_output)
        print(f"   Program {i+1} ({program.primitives[0].type.value}): EFE = {efe:.4f}")
    
    print(f"   Lower Expected Free Energy = Better program")
    
    # Solve the task
    print("\n4. SOLVING TASK WITH ACTIVE INFERENCE:")
    training_data = [{'input': input_matrix.tolist(), 'output': expected_output.tolist()}]
    
    start_time = time.time()
    best_program = agent.solve_task(training_data, num_generations=20, population_size=10)
    solve_time = time.time() - start_time
    
    print(f"   Solved in {solve_time:.2f}s")
    print_program(best_program)
    
    # Test the solution
    predicted = agent.predict(input_matrix, best_program)
    error = agent.prediction_error(predicted, expected_output)
    
    print("\n5. SOLUTION VERIFICATION:")
    print("   Predicted output:")
    print(predicted)
    print(f"   Prediction error: {error:.6f}")
    print(f"   Success: {'YES' if error < 1e-6 else 'NO'}")


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description="Active Inference ARC-AGI Agent"
    )
    parser.add_argument(
        '--mode',
        choices=['demo', 'single', 'benchmark'],
        default='demo',
        help='Run mode'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/training',
        help='Directory containing ARC task files'
    )
    parser.add_argument(
        '--task-file',
        type=str,
        help='Specific task file to run (for single mode)'
    )
    parser.add_argument(
        '--max-tasks',
        type=int,
        help='Maximum number of tasks to run (for benchmark mode)'
    )
    parser.add_argument(
        '--memory-size',
        type=int,
        default=50,
        help='Agent memory size'
    )
    parser.add_argument(
        '--max-program-length',
        type=int,
        default=4,
        help='Maximum program length'
    )
    parser.add_argument(
        '--generations',
        type=int,
        default=50,
        help='Number of generations for evolution'
    )
    parser.add_argument(
        '--population-size',
        type=int,
        default=20,
        help='Population size for evolution'
    )
    parser.add_argument(
        '--epistemic-weight',
        type=float,
        default=0.3,
        help='Weight for epistemic (exploration) value'
    )
    parser.add_argument(
        '--pragmatic-weight',
        type=float,
        default=0.7,
        help='Weight for pragmatic (goal-seeking) value'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Show visualizations'
    )
    
    args = parser.parse_args()
    
    print("Active Inference ARC-AGI Agent")
    print("="*50)
    
    if args.mode == 'demo':
        demonstrate_agent_principles()
        
    elif args.mode == 'single':
        if not args.task_file:
            print("Error: --task-file required for single mode")
            return
        
        task_file = Path(args.task_file)
        if not task_file.exists():
            print(f"Error: Task file {task_file} not found")
            return
        
        agent_config = {
            'memory_size': args.memory_size,
            'max_program_length': args.max_program_length,
            'learning_rate': 0.1,
            'epistemic_weight': args.epistemic_weight,
            'pragmatic_weight': args.pragmatic_weight
        }
        
        result = run_single_task(task_file, agent_config)
        
        if args.visualize:
            task_data = load_arc_task(task_file)
            agent = ActiveInferenceARCAgent(**agent_config)
            program = Program([
                Primitive(PrimitiveType(p['type']), p['params'], p['region'])
                for p in result['program']['primitives']
            ])
            visualize_program_performance(agent, task_data, program)
        
    elif args.mode == 'benchmark':
        data_dir = Path(args.data_dir)
        if not data_dir.exists():
            print(f"Error: Data directory {data_dir} not found")
            return
        
        agent_config = {
            'memory_size': args.memory_size,
            'max_program_length': args.max_program_length,
            'learning_rate': 0.1,
            'epistemic_weight': args.epistemic_weight,
            'pragmatic_weight': args.pragmatic_weight
        }
        
        results = run_benchmark(data_dir, args.max_tasks, agent_config)
        
        # Save results
        output_file = f"arc_benchmark_results_{int(time.time())}.json"
        with open(output_file, 'w') as f:
            json.dump({
                'config': agent_config,
                'args': vars(args),
                'results': results
            }, f, indent=2)
        print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
