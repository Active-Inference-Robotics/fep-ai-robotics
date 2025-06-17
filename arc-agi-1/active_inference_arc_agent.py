#!/usr/bin/env python3
"""
Active Inference ARC-AGI Agent

This module implements an Active Inference agent for the ARC-AGI dataset using 
Karl Friston's Free Energy Principle. The agent learns to compose primitive 
matrix operations to solve abstract reasoning tasks.

Key Components:
1. Generative Model: Predicts outcomes of primitive operations
2. Expected Free Energy: Balances epistemic (exploration) and pragmatic (goal-seeking) value
3. Program Synthesis: Composes sequences of primitive operations
4. Belief Updating: Updates model parameters based on prediction errors

References:
- Friston, K. (2010). The free-energy principle: a unified brain theory?
- Friston, K., et al. (2016). Active inference and learning
- Da Costa, L., et al. (2020). Active inference on discrete state-spaces
"""

import numpy as np
import json
import copy
from typing import List, Dict, Tuple, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import random
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PrimitiveType(Enum):
    """Available primitive operations for matrix manipulation."""
    ROTATE_90 = "rotate_90"
    ROTATE_180 = "rotate_180" 
    ROTATE_270 = "rotate_270"
    FLIP_HORIZONTAL = "flip_horizontal"
    FLIP_VERTICAL = "flip_vertical"
    TRANSLATE_UP = "translate_up"
    TRANSLATE_DOWN = "translate_down"
    TRANSLATE_LEFT = "translate_left"
    TRANSLATE_RIGHT = "translate_right"
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    COLOR_INVERT = "color_invert"
    EXTRACT_PATTERN = "extract_pattern"
    FILL_BACKGROUND = "fill_background"
    CONNECT_COMPONENTS = "connect_components"
    MIRROR_PATTERN = "mirror_pattern"
    IDENTITY = "identity"


@dataclass
class Primitive:
    """Represents a primitive operation with parameters."""
    type: PrimitiveType
    params: Dict[str, Any] = None
    region: Tuple[int, int, int, int] = None  # (x, y, width, height) for sub-region


@dataclass
class Program:
    """Represents a sequence of primitive operations."""
    primitives: List[Primitive]
    fitness: float = 0.0
    prediction_error: float = float('inf')


class MatrixOperations:
    """Implementation of primitive matrix operations."""
    
    @staticmethod
    def rotate_90(matrix: np.ndarray) -> np.ndarray:
        """Rotate matrix 90 degrees clockwise."""
        return np.rot90(matrix, k=-1)
    
    @staticmethod
    def rotate_180(matrix: np.ndarray) -> np.ndarray:
        """Rotate matrix 180 degrees."""
        return np.rot90(matrix, k=2)
    
    @staticmethod
    def rotate_270(matrix: np.ndarray) -> np.ndarray:
        """Rotate matrix 270 degrees clockwise."""
        return np.rot90(matrix, k=1)
    
    @staticmethod
    def flip_horizontal(matrix: np.ndarray) -> np.ndarray:
        """Flip matrix horizontally."""
        return np.fliplr(matrix)
    
    @staticmethod
    def flip_vertical(matrix: np.ndarray) -> np.ndarray:
        """Flip matrix vertically."""
        return np.flipud(matrix)
    
    @staticmethod
    def translate(matrix: np.ndarray, dx: int, dy: int) -> np.ndarray:
        """Translate matrix by dx, dy with wrap-around."""
        if dx == 0 and dy == 0:
            return matrix.copy()
        return np.roll(np.roll(matrix, dx, axis=1), dy, axis=0)
    
    @staticmethod
    def scale_up(matrix: np.ndarray, factor: int = 2) -> np.ndarray:
        """Scale matrix up by repeating pixels."""
        return np.repeat(np.repeat(matrix, factor, axis=0), factor, axis=1)
    
    @staticmethod
    def scale_down(matrix: np.ndarray, factor: int = 2) -> np.ndarray:
        """Scale matrix down by sampling."""
        if matrix.shape[0] < factor or matrix.shape[1] < factor:
            return matrix
        return matrix[::factor, ::factor]
    
    @staticmethod
    def color_invert(matrix: np.ndarray, mapping: Dict[int, int] = None) -> np.ndarray:
        """Invert colors based on mapping or default 0<->non-zero."""
        if mapping is None:
            # Simple inversion: 0 becomes 1, non-zero becomes 0
            result = matrix.copy()
            result[matrix == 0] = 1
            result[matrix != 0] = 0
            return result
        else:
            result = matrix.copy()
            for old_color, new_color in mapping.items():
                result[matrix == old_color] = new_color
            return result
    
    @staticmethod
    def extract_pattern(matrix: np.ndarray, background_color: int = 0) -> np.ndarray:
        """Extract non-background pattern."""
        return matrix != background_color
    
    @staticmethod
    def fill_background(matrix: np.ndarray, color: int) -> np.ndarray:
        """Fill background (0s) with specified color."""
        result = matrix.copy()
        result[matrix == 0] = color
        return result
    
    @staticmethod
    def apply_to_region(matrix: np.ndarray, operation: Callable, 
                       region: Tuple[int, int, int, int]) -> np.ndarray:
        """Apply operation to a specific region of the matrix."""
        x, y, w, h = region
        if x < 0 or y < 0 or x + w > matrix.shape[1] or y + h > matrix.shape[0]:
            return matrix  # Invalid region
        
        result = matrix.copy()
        sub_matrix = matrix[y:y+h, x:x+w]
        transformed = operation(sub_matrix)
        
        # Ensure transformed matrix fits back into region
        min_h = min(h, transformed.shape[0])
        min_w = min(w, transformed.shape[1])
        result[y:y+min_h, x:x+min_w] = transformed[:min_h, :min_w]
        
        return result


class ActiveInferenceARCAgent:
    """
    Active Inference agent for ARC-AGI tasks using program synthesis.
    
    This agent implements the Free Energy Principle by:
    1. Maintaining beliefs about which primitive operations are effective
    2. Minimizing expected free energy through exploration and exploitation
    3. Learning to compose primitives into effective programs
    4. Updating beliefs based on prediction errors
    """
    
    def __init__(self, 
                 memory_size: int = 100,
                 max_program_length: int = 5,
                 learning_rate: float = 0.1,
                 epistemic_weight: float = 0.3,
                 pragmatic_weight: float = 0.7):
        """
        Initialize the Active Inference ARC agent.
        
        Args:
            memory_size: Maximum number of programs to remember
            max_program_length: Maximum length of synthesized programs
            learning_rate: Rate of belief updating
            epistemic_weight: Weight for epistemic (exploration) value
            pragmatic_weight: Weight for pragmatic (goal-seeking) value
        """
        self.memory_size = memory_size
        self.max_program_length = max_program_length
        self.learning_rate = learning_rate
        self.epistemic_weight = epistemic_weight
        self.pragmatic_weight = pragmatic_weight
        
        # Initialize beliefs about primitive effectiveness
        self.primitive_beliefs = {
            primitive: {'success_count': 1.0, 'total_count': 2.0, 'precision': 1.0}
            for primitive in PrimitiveType
        }
        
        # Memory of successful programs
        self.program_memory: List[Program] = []
        
        # Statistics tracking
        self.prediction_errors: List[float] = []
        self.free_energies: List[float] = []
        self.epistemic_values: List[float] = []
        self.pragmatic_values: List[float] = []
        
        # Matrix operations handler
        self.ops = MatrixOperations()
        
        logger.info(f"Initialized Active Inference ARC Agent with {len(PrimitiveType)} primitives")
    
    def generative_model(self, input_matrix: np.ndarray, program: Program) -> np.ndarray:
        """
        Generative model that predicts the output given input and program.
        
        Args:
            input_matrix: Input matrix
            program: Program to execute
            
        Returns:
            Predicted output matrix
        """
        current_matrix = input_matrix.copy()
        
        for primitive in program.primitives:
            try:
                current_matrix = self._apply_primitive(current_matrix, primitive)
            except Exception as e:
                logger.warning(f"Error applying primitive {primitive.type}: {e}")
                # Return current state if operation fails
                break
                
        return current_matrix
    
    def _apply_primitive(self, matrix: np.ndarray, primitive: Primitive) -> np.ndarray:
        """Apply a single primitive operation to the matrix."""
        if primitive.type == PrimitiveType.ROTATE_90:
            result = self.ops.rotate_90(matrix)
        elif primitive.type == PrimitiveType.ROTATE_180:
            result = self.ops.rotate_180(matrix)
        elif primitive.type == PrimitiveType.ROTATE_270:
            result = self.ops.rotate_270(matrix)
        elif primitive.type == PrimitiveType.FLIP_HORIZONTAL:
            result = self.ops.flip_horizontal(matrix)
        elif primitive.type == PrimitiveType.FLIP_VERTICAL:
            result = self.ops.flip_vertical(matrix)
        elif primitive.type == PrimitiveType.TRANSLATE_UP:
            result = self.ops.translate(matrix, 0, -1)
        elif primitive.type == PrimitiveType.TRANSLATE_DOWN:
            result = self.ops.translate(matrix, 0, 1)
        elif primitive.type == PrimitiveType.TRANSLATE_LEFT:
            result = self.ops.translate(matrix, -1, 0)
        elif primitive.type == PrimitiveType.TRANSLATE_RIGHT:
            result = self.ops.translate(matrix, 1, 0)
        elif primitive.type == PrimitiveType.SCALE_UP:
            result = self.ops.scale_up(matrix, primitive.params.get('factor', 2) if primitive.params else 2)
        elif primitive.type == PrimitiveType.SCALE_DOWN:
            result = self.ops.scale_down(matrix, primitive.params.get('factor', 2) if primitive.params else 2)
        elif primitive.type == PrimitiveType.COLOR_INVERT:
            result = self.ops.color_invert(matrix, primitive.params.get('mapping') if primitive.params else None)
        elif primitive.type == PrimitiveType.FILL_BACKGROUND:
            color = primitive.params.get('color', 1) if primitive.params else 1
            result = self.ops.fill_background(matrix, color)
        elif primitive.type == PrimitiveType.IDENTITY:
            result = matrix.copy()
        else:
            result = matrix.copy()
        
        # Apply to region if specified
        if primitive.region is not None:
            result = self.ops.apply_to_region(matrix, 
                                            lambda m: self._apply_primitive(m, 
                                                Primitive(primitive.type, primitive.params)),
                                            primitive.region)
        
        return result
    
    def prediction_error(self, predicted: np.ndarray, target: np.ndarray) -> float:
        """
        Calculate prediction error between predicted and target matrices.
        
        Args:
            predicted: Predicted output matrix
            target: Target output matrix
            
        Returns:
            Prediction error (normalized)
        """
        if predicted.shape != target.shape:
            # Severe penalty for shape mismatch
            return 1.0 + abs(predicted.size - target.size) / max(predicted.size, target.size)
        
        # Normalized mean squared error
        diff = predicted.astype(float) - target.astype(float)
        mse = np.mean(diff ** 2)
        max_possible_error = max(np.max(predicted)**2, np.max(target)**2)
        
        if max_possible_error == 0:
            return 0.0 if mse == 0 else 1.0
        
        return min(mse / max_possible_error, 1.0)
    
    def epistemic_value(self, program: Program) -> float:
        """
        Calculate epistemic value (information gain potential) of a program.
        
        Args:
            program: Program to evaluate
            
        Returns:
            Epistemic value
        """
        # Information gain is higher for less-explored primitive combinations
        total_uncertainty = 0.0
        
        for primitive in program.primitives:
            belief = self.primitive_beliefs[primitive.type]
            # Uncertainty is higher when we have less experience
            uncertainty = 1.0 / (1.0 + belief['total_count'])
            total_uncertainty += uncertainty
        
        # Normalize by program length
        epistemic_val = total_uncertainty / max(len(program.primitives), 1)
        self.epistemic_values.append(epistemic_val)
        
        return epistemic_val
    
    def pragmatic_value(self, program: Program, input_matrix: np.ndarray, 
                       target_matrix: np.ndarray) -> float:
        """
        Calculate pragmatic value (goal achievement potential) of a program.
        
        Args:
            program: Program to evaluate
            input_matrix: Input matrix
            target_matrix: Target matrix
            
        Returns:
            Pragmatic value (negative prediction error)
        """
        predicted = self.generative_model(input_matrix, program)
        error = self.prediction_error(predicted, target_matrix)
        
        # Pragmatic value is negative error (higher value for lower error)
        pragmatic_val = -error
        self.pragmatic_values.append(pragmatic_val)
        
        return pragmatic_val
    
    def expected_free_energy(self, program: Program, input_matrix: np.ndarray,
                           target_matrix: np.ndarray) -> float:
        """
        Calculate expected free energy for a program.
        
        Expected Free Energy = -Epistemic Value - Pragmatic Value
        
        Args:
            program: Program to evaluate
            input_matrix: Input matrix
            target_matrix: Target matrix
            
        Returns:
            Expected free energy (to be minimized)
        """
        epistemic = self.epistemic_value(program)
        pragmatic = self.pragmatic_value(program, input_matrix, target_matrix)
        
        # Expected free energy (negative because we want to maximize value)
        efe = -(self.epistemic_weight * epistemic + self.pragmatic_weight * pragmatic)
        
        self.free_energies.append(efe)
        return efe
    
    def generate_program(self, length: Optional[int] = None) -> Program:
        """
        Generate a random program of specified length.
        
        Args:
            length: Program length (random if None)
            
        Returns:
            Generated program
        """
        if length is None:
            length = random.randint(1, self.max_program_length)
        
        primitives = []
        for _ in range(length):
            primitive_type = random.choice(list(PrimitiveType))
            
            # Add parameters for certain primitives
            params = {}
            if primitive_type in [PrimitiveType.SCALE_UP, PrimitiveType.SCALE_DOWN]:
                params['factor'] = random.choice([2, 3])
            elif primitive_type == PrimitiveType.COLOR_INVERT:
                # Sometimes add color mapping
                if random.random() < 0.3:
                    params['mapping'] = {0: 1, 1: 0}  # Simple swap
            elif primitive_type == PrimitiveType.FILL_BACKGROUND:
                params['color'] = random.randint(1, 9)
            
            primitive = Primitive(primitive_type, params if params else None)
            primitives.append(primitive)
        
        return Program(primitives)
    
    def mutate_program(self, program: Program) -> Program:
        """
        Create a mutated version of a program.
        
        Args:
            program: Program to mutate
            
        Returns:
            Mutated program
        """
        new_primitives = program.primitives.copy()
        
        if not new_primitives:
            return self.generate_program(1)
        
        mutation_type = random.choice(['add', 'remove', 'replace', 'swap'])
        
        if mutation_type == 'add' and len(new_primitives) < self.max_program_length:
            # Add a new primitive
            new_primitive_type = random.choice(list(PrimitiveType))
            new_primitive = Primitive(new_primitive_type)
            insert_pos = random.randint(0, len(new_primitives))
            new_primitives.insert(insert_pos, new_primitive)
            
        elif mutation_type == 'remove' and len(new_primitives) > 1:
            # Remove a primitive
            remove_pos = random.randint(0, len(new_primitives) - 1)
            del new_primitives[remove_pos]
            
        elif mutation_type == 'replace':
            # Replace a primitive
            replace_pos = random.randint(0, len(new_primitives) - 1)
            new_primitive_type = random.choice(list(PrimitiveType))
            new_primitives[replace_pos] = Primitive(new_primitive_type)
            
        elif mutation_type == 'swap' and len(new_primitives) > 1:
            # Swap two primitives
            pos1, pos2 = random.sample(range(len(new_primitives)), 2)
            new_primitives[pos1], new_primitives[pos2] = new_primitives[pos2], new_primitives[pos1]
        
        return Program(new_primitives)
    
    def update_beliefs(self, program: Program, success: bool):
        """
        Update beliefs about primitive effectiveness based on program performance.
        
        Args:
            program: Executed program
            success: Whether the program was successful
        """
        for primitive in program.primitives:
            belief = self.primitive_beliefs[primitive.type]
            
            if success:
                belief['success_count'] += self.learning_rate
            belief['total_count'] += self.learning_rate
            
            # Update precision based on consistency
            success_rate = belief['success_count'] / belief['total_count']
            belief['precision'] = belief['total_count'] * (success_rate * (1 - success_rate) + 0.01)
    
    def solve_task(self, training_examples: List[Dict], 
                   num_generations: int = 50,
                   population_size: int = 20) -> Program:
        """
        Solve a task using active inference and program synthesis.
        
        Args:
            training_examples: List of input-output example pairs
            num_generations: Number of generations for evolution
            population_size: Size of program population
            
        Returns:
            Best program found
        """
        if not training_examples:
            return Program([Primitive(PrimitiveType.IDENTITY)])
        
        logger.info(f"Solving task with {len(training_examples)} examples")
        
        # Initialize population
        population = [self.generate_program() for _ in range(population_size)]
        
        best_program = None
        best_fitness = float('inf')
        
        for generation in range(num_generations):
            # Evaluate population
            for program in population:
                total_error = 0.0
                
                for example in training_examples:
                    input_matrix = np.array(example['input'])
                    target_matrix = np.array(example['output'])
                    
                    # Calculate expected free energy
                    efe = self.expected_free_energy(program, input_matrix, target_matrix)
                    total_error += efe
                
                program.fitness = total_error / len(training_examples)
                
                # Track best program
                if program.fitness < best_fitness:
                    best_fitness = program.fitness
                    best_program = copy.deepcopy(program)
            
            # Update beliefs for best programs
            sorted_population = sorted(population, key=lambda p: p.fitness)
            for i, program in enumerate(sorted_population[:population_size//4]):
                success = i < population_size//8  # Top performers are "successful"
                self.update_beliefs(program, success)
            
            # Generate next generation using active inference
            new_population = []
            
            # Keep best programs (elitism)
            elite_count = population_size // 4
            new_population.extend(sorted_population[:elite_count])
            
            # Generate new programs with bias towards successful primitives
            while len(new_population) < population_size:
                if random.random() < 0.3:
                    # Pure exploration: random program
                    new_program = self.generate_program()
                else:
                    # Exploitation: mutate good programs
                    parent = random.choice(sorted_population[:population_size//2])
                    new_program = self.mutate_program(parent)
                
                new_population.append(new_program)
            
            population = new_population
            
            if generation % 10 == 0:
                logger.info(f"Generation {generation}: Best fitness = {best_fitness:.4f}")
        
        # Store successful program in memory
        if best_program and self.memory_size > 0:
            if len(self.program_memory) < self.memory_size:
                self.program_memory.append(copy.deepcopy(best_program))
            else:
                # Replace worst program in memory
                worst_idx = max(range(len(self.program_memory)), 
                              key=lambda i: self.program_memory[i].fitness)
                if best_program.fitness < self.program_memory[worst_idx].fitness:
                    self.program_memory[worst_idx] = copy.deepcopy(best_program)
        
        logger.info(f"Task solved with fitness {best_fitness:.4f}")
        return best_program or Program([Primitive(PrimitiveType.IDENTITY)])
    
    def predict(self, input_matrix: np.ndarray, program: Program) -> np.ndarray:
        """
        Predict output for given input using the specified program.
        
        Args:
            input_matrix: Input matrix
            program: Program to execute
            
        Returns:
            Predicted output matrix
        """
        return self.generative_model(input_matrix, program)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        return {
            'memory_programs': len(self.program_memory),
            'primitive_beliefs': {
                prim.value: {
                    'success_rate': belief['success_count'] / belief['total_count'],
                    'precision': belief['precision'],
                    'experience': belief['total_count']
                }
                for prim, belief in self.primitive_beliefs.items()
            },
            'recent_prediction_errors': self.prediction_errors[-10:] if self.prediction_errors else [],
            'recent_free_energies': self.free_energies[-10:] if self.free_energies else [],
            'recent_epistemic_values': self.epistemic_values[-10:] if self.epistemic_values else [],
            'recent_pragmatic_values': self.pragmatic_values[-10:] if self.pragmatic_values else []
        }
