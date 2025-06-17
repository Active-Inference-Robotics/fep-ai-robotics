#!/usr/bin/env python3
"""
ARC-AGI-1 Dataset Analysis Script

This script analyzes the ARC-AGI-1 dataset from the training and evaluation folders
and generates a comprehensive markdown summary with statistics about:
- Matrix sizes for inputs and outputs
- Color distributions 
- Task counts and dataset structure
"""

import json
import os
import numpy as np
from collections import defaultdict, Counter
from pathlib import Path
import argparse


def load_task_files(folder_path):
    """Load all JSON task files from a folder."""
    tasks = []
    folder = Path(folder_path)
    
    if not folder.exists():
        print(f"Warning: Folder {folder_path} does not exist!")
        return tasks
    
    json_files = list(folder.glob("*.json"))
    print(f"Found {len(json_files)} JSON files in {folder_path}")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                task_data = json.load(f)
                tasks.append({
                    'filename': json_file.name,
                    'data': task_data
                })
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
    
    return tasks


def analyze_matrix_properties(matrices):
    """Analyze matrix size and color properties, and count color diversity."""
    sizes = []
    all_colors = []
    single_color_count = 0
    multi_color_count = 0

    for matrix in matrices:
        if matrix:  # Check if matrix is not empty
            height = len(matrix)
            width = len(matrix[0]) if matrix else 0
            sizes.append((height, width))
            # Flatten matrix to get all colors
            flat = []
            for row in matrix:
                flat.extend(row)
            all_colors.extend(flat)
            unique_colors = set(flat)
            if len(unique_colors) == 1:
                single_color_count += 1
            elif len(unique_colors) > 1:
                multi_color_count += 1

    return sizes, all_colors, single_color_count, multi_color_count


def calculate_statistics(data_list):
    """Calculate basic statistics for a list of numbers."""
    if not data_list:
        return {'count': 0, 'min': 0, 'max': 0, 'mean': 0, 'std': 0}
    
    arr = np.array(data_list)
    return {
        'count': len(data_list),
        'min': float(np.min(arr)),
        'max': float(np.max(arr)),
        'mean': float(np.mean(arr)),
        'std': float(np.std(arr))
    }


def analyze_dataset(tasks, dataset_name):
    """Analyze a complete dataset (training or evaluation)."""
    print(f"\nAnalyzing {dataset_name} dataset...")

    analysis = {
        'name': dataset_name,
        'total_tasks': len(tasks),
        'input_sizes': [],
        'output_sizes': [],
        'input_colors': [],
        'output_colors': [],
        'train_examples_per_task': [],
        'test_examples_per_task': [],
        'input_single_color': 0,
        'input_multi_color': 0,
        'output_single_color': 0,
        'output_multi_color': 0
    }

    for task in tasks:
        task_data = task['data']

        # Analyze training examples
        if 'train' in task_data:
            train_examples = task_data['train']
            analysis['train_examples_per_task'].append(len(train_examples))

            for example in train_examples:
                # Input analysis
                if 'input' in example:
                    input_sizes, input_colors, single, multi = analyze_matrix_properties([example['input']])
                    analysis['input_sizes'].extend(input_sizes)
                    analysis['input_colors'].extend(input_colors)
                    analysis['input_single_color'] += single
                    analysis['input_multi_color'] += multi

                # Output analysis
                if 'output' in example:
                    output_sizes, output_colors, single, multi = analyze_matrix_properties([example['output']])
                    analysis['output_sizes'].extend(output_sizes)
                    analysis['output_colors'].extend(output_colors)
                    analysis['output_single_color'] += single
                    analysis['output_multi_color'] += multi

        # Analyze test examples
        if 'test' in task_data:
            test_examples = task_data['test']
            analysis['test_examples_per_task'].append(len(test_examples))

            for example in test_examples:
                # Input analysis
                if 'input' in example:
                    input_sizes, input_colors, single, multi = analyze_matrix_properties([example['input']])
                    analysis['input_sizes'].extend(input_sizes)
                    analysis['input_colors'].extend(input_colors)
                    analysis['input_single_color'] += single
                    analysis['input_multi_color'] += multi

                # Output analysis (if available in test data)
                if 'output' in example:
                    output_sizes, output_colors, single, multi = analyze_matrix_properties([example['output']])
                    analysis['output_sizes'].extend(output_sizes)
                    analysis['output_colors'].extend(output_colors)
                    analysis['output_single_color'] += single
                    analysis['output_multi_color'] += multi

    return analysis


def generate_size_statistics(sizes, data_type):
    """Generate statistics for matrix sizes."""
    if not sizes:
        return "No data available\n"
    
    heights = [size[0] for size in sizes]
    widths = [size[1] for size in sizes]
    areas = [h * w for h, w in sizes]
    
    height_stats = calculate_statistics(heights)
    width_stats = calculate_statistics(widths)
    area_stats = calculate_statistics(areas)
    
    # Most common sizes
    size_counter = Counter(sizes)
    most_common_sizes = size_counter.most_common(10)
    
    result = f"### {data_type} Matrix Size Statistics\n\n"
    result += f"**Total matrices analyzed:** {len(sizes)}\n\n"
    
    result += "| Dimension | Min | Max | Mean | Std Dev |\n"
    result += "|-----------|-----|-----|------|---------|\n"
    result += f"| Height | {height_stats['min']:.1f} | {height_stats['max']:.1f} | {height_stats['mean']:.2f} | {height_stats['std']:.2f} |\n"
    result += f"| Width | {width_stats['min']:.1f} | {width_stats['max']:.1f} | {width_stats['mean']:.2f} | {width_stats['std']:.2f} |\n"
    result += f"| Area | {area_stats['min']:.1f} | {area_stats['max']:.1f} | {area_stats['mean']:.2f} | {area_stats['std']:.2f} |\n\n"
    
    result += "**Most Common Matrix Sizes (Height × Width):**\n"
    for i, ((h, w), count) in enumerate(most_common_sizes, 1):
        result += f"{i}. {h}×{w}: {count} matrices\n"
    
    return result + "\n"


def generate_color_statistics(colors, data_type):
    """Generate statistics for color distributions."""
    if not colors:
        return "No color data available\n"
    
    color_counter = Counter(colors)
    total_pixels = len(colors)
    unique_colors = len(color_counter)
    
    result = f"### {data_type} Color Statistics\n\n"
    result += f"**Total pixels analyzed:** {total_pixels:,}\n"
    result += f"**Unique colors found:** {unique_colors}\n\n"
    
    result += "**Color Distribution:**\n"
    result += "| Color | Count | Percentage |\n"
    result += "|-------|-------|-----------|\n"
    
    for color, count in color_counter.most_common():
        percentage = (count / total_pixels) * 100
        result += f"| {color} | {count:,} | {percentage:.2f}% |\n"
    
    return result + "\n"


def generate_color_diversity_stats(single_color, multi_color, data_type):
    total = single_color + multi_color
    if total == 0:
        return f"No {data_type} matrices found.\n"
    pct_single = (single_color / total) * 100
    pct_multi = (multi_color / total) * 100
    result = f"### {data_type} Matrix Color Diversity\n\n"
    result += f"- Matrices with **one color**: {single_color} ({pct_single:.2f}%)\n"
    result += f"- Matrices with **more than one color**: {multi_color} ({pct_multi:.2f}%)\n\n"
    return result


def generate_markdown_report(training_analysis, evaluation_analysis):
    """Generate a comprehensive markdown report."""
    
    report = """# ARC-AGI-1 Dataset Analysis Report

This report provides a comprehensive analysis of the ARC-AGI-1 dataset, including both training and evaluation sets.

## Dataset Overview

"""
    
    # Dataset summary table
    report += "| Dataset | Total Tasks | Total Train Examples | Total Test Examples |\n"
    report += "|---------|-------------|---------------------|--------------------|\n"
    
    train_examples_total = sum(training_analysis['train_examples_per_task'])
    train_test_examples_total = sum(training_analysis['test_examples_per_task'])
    eval_train_examples_total = sum(evaluation_analysis['train_examples_per_task'])
    eval_test_examples_total = sum(evaluation_analysis['test_examples_per_task'])
    
    report += f"| Training | {training_analysis['total_tasks']} | {train_examples_total} | {train_test_examples_total} |\n"
    report += f"| Evaluation | {evaluation_analysis['total_tasks']} | {eval_train_examples_total} | {eval_test_examples_total} |\n"
    report += f"| **Total** | **{training_analysis['total_tasks'] + evaluation_analysis['total_tasks']}** | **{train_examples_total + eval_train_examples_total}** | **{train_test_examples_total + eval_test_examples_total}** |\n\n"
    
    # Training dataset analysis
    report += "## Training Dataset Analysis\n\n"
    
    # Combine input and output sizes for training
    all_training_input_sizes = training_analysis['input_sizes']
    all_training_output_sizes = training_analysis['output_sizes']
    all_training_input_colors = training_analysis['input_colors']
    all_training_output_colors = training_analysis['output_colors']
    
    report += generate_size_statistics(all_training_input_sizes, "Training Input")
    report += generate_size_statistics(all_training_output_sizes, "Training Output")
    report += generate_color_statistics(all_training_input_colors, "Training Input")
    report += generate_color_statistics(all_training_output_colors, "Training Output")
    report += generate_color_diversity_stats(
        training_analysis['input_single_color'],
        training_analysis['input_multi_color'],
        "Training Input"
    )
    report += generate_color_diversity_stats(
        training_analysis['output_single_color'],
        training_analysis['output_multi_color'],
        "Training Output"
    )

    # Evaluation dataset analysis
    report += "## Evaluation Dataset Analysis\n\n"
    
    # Combine input and output sizes for evaluation
    all_eval_input_sizes = evaluation_analysis['input_sizes']
    all_eval_output_sizes = evaluation_analysis['output_sizes']
    all_eval_input_colors = evaluation_analysis['input_colors']
    all_eval_output_colors = evaluation_analysis['output_colors']
    
    report += generate_size_statistics(all_eval_input_sizes, "Evaluation Input")
    report += generate_size_statistics(all_eval_output_sizes, "Evaluation Output")
    report += generate_color_statistics(all_eval_input_colors, "Evaluation Input")
    report += generate_color_statistics(all_eval_output_colors, "Evaluation Output")
    report += generate_color_diversity_stats(
        evaluation_analysis['input_single_color'],
        evaluation_analysis['input_multi_color'],
        "Evaluation Input"
    )
    report += generate_color_diversity_stats(
        evaluation_analysis['output_single_color'],
        evaluation_analysis['output_multi_color'],
        "Evaluation Output"
    )

    # Combined analysis
    report += "## Combined Dataset Statistics\n\n"
    
    combined_input_sizes = all_training_input_sizes + all_eval_input_sizes
    combined_output_sizes = all_training_output_sizes + all_eval_output_sizes
    combined_input_colors = all_training_input_colors + all_eval_input_colors
    combined_output_colors = all_training_output_colors + all_eval_output_colors
    combined_input_single = training_analysis['input_single_color'] + evaluation_analysis['input_single_color']
    combined_input_multi = training_analysis['input_multi_color'] + evaluation_analysis['input_multi_color']
    combined_output_single = training_analysis['output_single_color'] + evaluation_analysis['output_single_color']
    combined_output_multi = training_analysis['output_multi_color'] + evaluation_analysis['output_multi_color']

    report += generate_size_statistics(combined_input_sizes, "Combined Input")
    report += generate_size_statistics(combined_output_sizes, "Combined Output")
    report += generate_color_statistics(combined_input_colors, "Combined Input")
    report += generate_color_statistics(combined_output_colors, "Combined Output")
    report += generate_color_diversity_stats(
        combined_input_single,
        combined_input_multi,
        "Combined Input"
    )
    report += generate_color_diversity_stats(
        combined_output_single,
        combined_output_multi,
        "Combined Output"
    )
    
    # Task complexity analysis
    report += "## Task Complexity Analysis\n\n"
    
    train_examples_stats = calculate_statistics(training_analysis['train_examples_per_task'])
    eval_examples_stats = calculate_statistics(evaluation_analysis['train_examples_per_task'])
    
    report += "### Training Examples per Task\n\n"
    report += "| Dataset | Min | Max | Mean | Std Dev |\n"
    report += "|---------|-----|-----|------|---------|\n"
    report += f"| Training | {train_examples_stats['min']:.1f} | {train_examples_stats['max']:.1f} | {train_examples_stats['mean']:.2f} | {train_examples_stats['std']:.2f} |\n"
    report += f"| Evaluation | {eval_examples_stats['min']:.1f} | {eval_examples_stats['max']:.1f} | {eval_examples_stats['mean']:.2f} | {eval_examples_stats['std']:.2f} |\n\n"
    
    # Color analysis summary
    report += "## Key Findings\n\n"
    report += "### Matrix Sizes\n"
    report += f"- Input matrices range from the smallest to largest observed sizes\n"
    report += f"- Output matrices show similar size distributions\n"
    report += f"- Most tasks involve relatively small matrices (typical sizes under 30×30)\n\n"
    
    report += "### Color Usage\n"
    report += f"- The dataset uses colors 0-9 (10 total colors)\n"
    report += f"- Color 0 (typically background) is most frequent\n"
    report += f"- Color distribution varies between training and evaluation sets\n\n"
    
    report += "### Task Structure\n"
    report += f"- Training set: {training_analysis['total_tasks']} tasks\n"
    report += f"- Evaluation set: {evaluation_analysis['total_tasks']} tasks\n"
    report += f"- Each task contains multiple input-output examples for learning the pattern\n\n"
    
    report += "---\n"
    report += "*Report generated by ARC-AGI-1 Dataset Analyzer*\n"
    
    return report


def main():
    parser = argparse.ArgumentParser(description='Analyze ARC-AGI-1 dataset and generate markdown report')
    parser.add_argument('--data-dir', default='./data',
                       help='Path to the data directory containing training and evaluation folders')
    parser.add_argument('--output', default='arc_agi_dataset_analysis.md',
                       help='Output markdown file name')
    
    args = parser.parse_args()
    
    # Define paths
    data_dir = Path(args.data_dir)
    training_dir = data_dir / 'training'
    evaluation_dir = data_dir / 'evaluation'
    
    print("ARC-AGI-1 Dataset Analyzer")
    print("=" * 50)
    
    # Load and analyze training data
    print("Loading training data...")
    training_tasks = load_task_files(training_dir)
    training_analysis = analyze_dataset(training_tasks, "Training")
    
    # Load and analyze evaluation data  
    print("Loading evaluation data...")
    evaluation_tasks = load_task_files(evaluation_dir)
    evaluation_analysis = analyze_dataset(evaluation_tasks, "Evaluation")
    
    # Generate report
    print("Generating markdown report...")
    report = generate_markdown_report(training_analysis, evaluation_analysis)
    
    # Save report
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        f.write(report)
    
    print(f"Analysis complete! Report saved to: {output_path}")
    print(f"\nQuick Summary:")
    print(f"- Training tasks: {training_analysis['total_tasks']}")
    print(f"- Evaluation tasks: {evaluation_analysis['total_tasks']}")
    print(f"- Total tasks: {training_analysis['total_tasks'] + evaluation_analysis['total_tasks']}")


if __name__ == "__main__":
    main()
