#!/usr/bin/env python3
"""
Minimal Active Inference ARC Demo

This script demonstrates the core Active Inference concepts with a simple example
that doesn't rely on complex imports or external dependencies.
"""

import numpy as np
import random
from enum import Enum
from dataclasses import dataclass
from typing import List, Dict


class PrimitiveType(Enum):
    ROTATE_90 = "rotate_90"
    FLIP_HORIZONTAL = "flip_horizontal"
    IDENTITY = "identity"


@dataclass
class Primitive:
    type: PrimitiveType


@dataclass
class Program:
    primitives: List[Primitive]
    fitness: float = 0.0


class MinimalARCAgent:
    """Minimal Active Inference agent for demonstration."""
    
    def __init__(self):
        # Beliefs about primitive effectiveness
        self.primitive_beliefs = {
            PrimitiveType.ROTATE_90: 0.5,
            PrimitiveType.FLIP_HORIZONTAL: 0.5,
            PrimitiveType.IDENTITY: 0.8
        }
        
    def apply_primitive(self, matrix: np.ndarray, primitive: Primitive) -> np.ndarray:
        """Apply a primitive operation to the matrix."""
        if primitive.type == PrimitiveType.ROTATE_90:
            return np.rot90(matrix, k=-1)  # 90 degrees clockwise
        elif primitive.type == PrimitiveType.FLIP_HORIZONTAL:
            return np.fliplr(matrix)
        else:  # IDENTITY
            return matrix.copy()
    
    def execute_program(self, matrix: np.ndarray, program: Program) -> np.ndarray:
        """Execute a program on the input matrix."""
        result = matrix.copy()
        for primitive in program.primitives:
            result = self.apply_primitive(result, primitive)
        return result
    
    def prediction_error(self, predicted: np.ndarray, target: np.ndarray) -> float:
        """Calculate prediction error."""
        if predicted.shape != target.shape:
            return 1.0
        return np.mean((predicted - target) ** 2) / np.mean(target ** 2 + 1e-8)
    
    def epistemic_value(self, program: Program) -> float:
        """Calculate epistemic value (exploration)."""
        total_uncertainty = 0.0
        for primitive in program.primitives:
            # Higher uncertainty for less-believed primitives
            belief = self.primitive_beliefs[primitive.type]
            uncertainty = 1.0 - belief
            total_uncertainty += uncertainty
        return total_uncertainty / len(program.primitives)
    
    def pragmatic_value(self, program: Program, input_matrix: np.ndarray, 
                       target_matrix: np.ndarray) -> float:
        """Calculate pragmatic value (goal achievement)."""
        predicted = self.execute_program(input_matrix, program)
        error = self.prediction_error(predicted, target_matrix)
        return -error  # Negative error is positive value
    
    def expected_free_energy(self, program: Program, input_matrix: np.ndarray,
                           target_matrix: np.ndarray) -> float:
        """Calculate expected free energy."""
        epistemic = self.epistemic_value(program)
        pragmatic = self.pragmatic_value(program, input_matrix, target_matrix)
        
        # Balance exploration (epistemic) and exploitation (pragmatic)
        efe = -(0.3 * epistemic + 0.7 * pragmatic)
        return efe
    
    def solve_simple_task(self, input_matrix: np.ndarray, target_matrix: np.ndarray):
        """Solve a simple task using Active Inference."""
        
        print("=== ACTIVE INFERENCE ARC DEMONSTRATION ===")
        print()
        print("Input matrix:")
        print(input_matrix)
        print("Target matrix:")
        print(target_matrix)
        print()
        
        # Generate candidate programs
        programs = [
            Program([Primitive(PrimitiveType.ROTATE_90)]),
            Program([Primitive(PrimitiveType.FLIP_HORIZONTAL)]),
            Program([Primitive(PrimitiveType.IDENTITY)]),
            Program([Primitive(PrimitiveType.ROTATE_90), Primitive(PrimitiveType.FLIP_HORIZONTAL)]),
            Program([Primitive(PrimitiveType.FLIP_HORIZONTAL), Primitive(PrimitiveType.ROTATE_90)])
        ]
        
        print("Evaluating programs using Expected Free Energy:")
        print()
        
        best_program = None
        best_efe = float('inf')
        
        for i, program in enumerate(programs):
            efe = self.expected_free_energy(program, input_matrix, target_matrix)
            epistemic = self.epistemic_value(program)
            pragmatic = self.pragmatic_value(program, input_matrix, target_matrix)
            predicted = self.execute_program(input_matrix, program)
            error = self.prediction_error(predicted, target_matrix)
            
            program_desc = " → ".join([p.type.value for p in program.primitives])
            
            print(f"Program {i+1}: {program_desc}")
            print(f"  Epistemic Value (exploration): {epistemic:.3f}")
            print(f"  Pragmatic Value (goal-seeking): {pragmatic:.3f}")
            print(f"  Expected Free Energy: {efe:.3f}")
            print(f"  Prediction Error: {error:.3f}")
            print(f"  Predicted output:")
            print(f"  {predicted}")
            print()
            
            if efe < best_efe:
                best_efe = efe
                best_program = program
        
        print(f"BEST PROGRAM (lowest Expected Free Energy):")
        best_desc = " → ".join([p.type.value for p in best_program.primitives])
        print(f"  {best_desc}")
        print(f"  Expected Free Energy: {best_efe:.3f}")
        
        # Test the best program
        final_prediction = self.execute_program(input_matrix, best_program)
        final_error = self.prediction_error(final_prediction, target_matrix)
        
        print()
        print("FINAL RESULT:")
        print("Predicted:")
        print(final_prediction)
        print("Target:")
        print(target_matrix)
        print(f"Success: {'YES' if final_error < 0.01 else 'NO'} (error: {final_error:.6f})")
        
        return best_program


def main():
    """Demonstrate Active Inference principles."""
    
    # Create agent
    agent = MinimalARCAgent()
    
    # Test Case 1: Rotation task
    print("TEST CASE 1: ROTATION TASK")
    print("=" * 50)
    
    input_matrix = np.array([
        [1, 0, 0],
        [1, 1, 0],
        [0, 0, 0]
    ])
    
    target_matrix = np.array([
        [0, 1, 1],
        [0, 1, 0],
        [0, 0, 0]
    ])
    
    best_program = agent.solve_simple_task(input_matrix, target_matrix)
    
    print("\n" + "=" * 50)
    print("TEST CASE 2: FLIP TASK")
    print("=" * 50)
    
    # Test Case 2: Flip task
    input_matrix2 = np.array([
        [1, 0, 1],
        [0, 1, 0],
        [1, 0, 1]
    ])
    
    target_matrix2 = np.array([
        [1, 0, 1],
        [0, 1, 0],
        [1, 0, 1]
    ])
    
    best_program2 = agent.solve_simple_task(input_matrix2, target_matrix2)
    
    print("\n" + "=" * 70)
    print("ACTIVE INFERENCE PRINCIPLES DEMONSTRATED:")
    print("=" * 70)
    print("1. EXPECTED FREE ENERGY: Agent minimizes EFE to select actions")
    print("2. EPISTEMIC VALUE: Explores uncertain/unfamiliar operations")  
    print("3. PRAGMATIC VALUE: Seeks to achieve goals (minimize prediction error)")
    print("4. GENERATIVE MODEL: Predicts outcomes of primitive operations")
    print("5. BELIEF-BASED BEHAVIOR: Decisions based on beliefs about primitives")
    print()
    print("This demonstrates how biological intelligence might work:")
    print("- Balance exploration vs exploitation")
    print("- Learn through prediction error minimization") 
    print("- Compose simple operations into complex behaviors")
    print("- Use uncertainty to guide exploration")


if __name__ == "__main__":
    main()
