"""
Tutorial 1: Bayesian Classification with pgmpy
===============================================

This tutorial demonstrates Bayesian inference using the pgmpy library with a 
frog vs apple classification example. It includes:

1. Creating a generative model with variables x (object type) and y (behavior)
2. Performing exact Bayesian inference
3. Calculating surprise values using the negative log probability formula
4. Visualizing results and comparing with manual calculations

Key concepts:
- Prior beliefs P(x)
- Likelihood model P(y|x) 
- Posterior inference P(x|y)
- Surprise function ℑ = -ln P
- Information gain from observations

Reference:
----------
This tutorial is inspired by concepts from the book:
"Active Inference: The Free Energy Principle in Mind, Brain, and Behavior"  
by Thomas Parr, Giovanni Pezzulo, and Karl J. Friston (MIT Press, 2022),  
especially Chapter 2, which introduces Bayesian inference, surprise, and information gain  
in the context of active inference and the free energy principle.
See: https://direct.mit.edu/books/oa-monograph/5299/Active-InferenceThe-Free-Energy-Principle-in-Mind

Author: FEP AI Robotics Team
Date: May 2025
"""

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.inference import VariableElimination
from pgmpy.factors.discrete import TabularCPD
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gammaln

def surprise_function(probability, distribution_type='categorical'):
    """
    Calculate surprise (negative log probability) for different distributions
    
    What is "surprise" and what are "nats"?
    =====================================
    
    In information theory and active inference, "surprise" quantifies how unexpected 
    an observation is. It's measured using the negative logarithm of probability:
    
    ℑ = -ln P(event)
    
    The unit of surprise depends on the logarithm base:
    - Natural logarithm (ln): measured in "nats" (natural units)
    - Base-2 logarithm (log₂): measured in "bits" 
    - Base-10 logarithm (log₁₀): measured in "dits" or "bans"
    
    Intuitive interpretation:
    - Higher probability → Lower surprise (expected events are less surprising)
    - Lower probability → Higher surprise (rare events are more surprising)
    - P = 1.0 (certain event) → ℑ = 0 nats (no surprise)
    - P = 0.37 (≈ 1/e) → ℑ = 1 nat (one natural unit of surprise)
    
    Example: If P(y=jumps) = 0.09, we would only expect to observe jumping 
    behavior 9 out of 100 times. The surprise is:
    ℑ = -ln(0.09) ≈ 2.4 nats
    
    This means the jumping event contains about 2.4 natural units of information.
    For comparison:
    - A coin flip (P=0.5) has ℑ = 0.69 nats ≈ 1 bit of surprise
    - Rolling a 6 on a die (P=1/6≈0.167) has ℑ = 1.79 nats
    - A rare event (P=0.01) has ℑ = 4.61 nats
    
    In active inference, agents try to minimize surprise by forming better 
    predictions about their environment. The surprise function is central to 
    the free energy principle.
    
    Parameters:
    -----------
    probability : float or array-like
        Probability value(s) for which to calculate surprise
    distribution_type : str
        Type of distribution ('categorical', 'gaussian', 'multinomial', 'dirichlet', 'gamma')
    
    Returns:
    --------
    float or array-like
        Surprise value(s) in nats (natural units)
    """
    
    if distribution_type == 'categorical':
        # For categorical/discrete distributions: ℑ = -ln P
        return -np.log(probability)
    
    elif distribution_type == 'gaussian':
        # For Gaussian: ℑ = ½(x - μ) · Π(x - μ) where Π is precision matrix
        # For univariate case: ℑ = ½(x - μ)²/σ² + ½ln(2πσ²)
        # Here we assume probability is already the density value
        return -np.log(probability)
    
    elif distribution_type == 'multinomial':
        # For multinomial: ℑ = -∑ᵢ xᵢ ln dᵢ where dᵢ are the probabilities
        # probability should be array of probabilities, xi should be counts
        if isinstance(probability, (list, np.ndarray)):
            return -np.sum(probability * np.log(probability + 1e-10))  # Add small epsilon to avoid log(0)
        else:
            return -np.log(probability)
    
    elif distribution_type == 'dirichlet':
        # For Dirichlet: ℑ = ∑ᵢ(1 - αᵢ) ln xᵢ
        # probability should be array of concentration parameters α
        if isinstance(probability, (list, np.ndarray)):
            # Assuming uniform Dirichlet for simplicity
            return np.sum((1 - np.array(probability)) * np.log(np.array(probability) + 1e-10))
        else:
            return -np.log(probability)
    
    elif distribution_type == 'gamma':
        # For Gamma: ℑ = (bx + (1-a) ln x) where a=shape, b=rate
        # For simplicity, treating as -ln P(x)
        return -np.log(probability)
    
    else:
        # Default case: negative log probability
        return -np.log(probability)

def calculate_surprises(model, results):
    """
    Calculate surprise values for different events in the frog/apple model
    """
    print("6. Surprise Calculations:")
    print("=" * 30)
    
    # Get marginal probability of jumping
    p_jumps = results['prior_y'].values[1]  # P(y=jumps)
    p_no_jumps = results['prior_y'].values[0]  # P(y=doesn't jump)
    
    # Calculate surprises for observations
    surprise_jumps = surprise_function(p_jumps, 'categorical')
    surprise_no_jumps = surprise_function(p_no_jumps, 'categorical')
    
    print(f"P(y = jumps) = {p_jumps:.4f}")
    print(f"ℑ(y = jumps) = -ln P(y = jumps) = -ln({p_jumps:.4f}) = {surprise_jumps:.2f} nats")
    print()
    print(f"P(y = doesn't jump) = {p_no_jumps:.4f}")
    print(f"ℑ(y = doesn't jump) = -ln P(y = doesn't jump) = -ln({p_no_jumps:.4f}) = {surprise_no_jumps:.2f} nats")
    print()
    
    # Calculate surprises for posterior beliefs
    p_frog_given_jumps = results['posterior_jumps'].values[0]
    p_apple_given_jumps = results['posterior_jumps'].values[1]
    
    surprise_frog_given_jumps = surprise_function(p_frog_given_jumps, 'categorical')
    surprise_apple_given_jumps = surprise_function(p_apple_given_jumps, 'categorical')
    
    print("Posterior surprises given jumping observation:")
    print(f"P(x = frog | y = jumps) = {p_frog_given_jumps:.4f}")
    print(f"ℑ(x = frog | y = jumps) = {surprise_frog_given_jumps:.2f} nats")
    print(f"P(x = apple | y = jumps) = {p_apple_given_jumps:.4f}")
    print(f"ℑ(x = apple | y = jumps) = {surprise_apple_given_jumps:.2f} nats")
    print()
    
    # Calculate joint surprises
    print("Joint event surprises:")
    
    # P(x=frog, y=jumps) = P(y=jumps|x=frog) * P(x=frog)
    p_frog_and_jumps = 0.81 * 0.1
    surprise_frog_and_jumps = surprise_function(p_frog_and_jumps, 'categorical')
    
    # P(x=apple, y=jumps) = P(y=jumps|x=apple) * P(x=apple)
    p_apple_and_jumps = 0.01 * 0.9
    surprise_apple_and_jumps = surprise_function(p_apple_and_jumps, 'categorical')
    
    print(f"P(x = frog, y = jumps) = {p_frog_and_jumps:.4f}")
    print(f"ℑ(x = frog, y = jumps) = {surprise_frog_and_jumps:.2f} nats")
    print(f"P(x = apple, y = jumps) = {p_apple_and_jumps:.4f}")
    print(f"ℑ(x = apple, y = jumps) = {surprise_apple_and_jumps:.2f} nats")
    print()
    
    # Information gain (reduction in surprise)
    prior_frog_surprise = surprise_function(0.1, 'categorical')  # Prior surprise about being frog
    posterior_frog_surprise = surprise_frog_given_jumps  # Posterior surprise given jumping
    
    information_gain = prior_frog_surprise - posterior_frog_surprise
    print(f"Information gain about 'frog' from observing jumping:")
    print(f"Prior surprise: ℑ(x = frog) = {prior_frog_surprise:.2f} nats")
    print(f"Posterior surprise: ℑ(x = frog | y = jumps) = {posterior_frog_surprise:.2f} nats")
    print(f"Information gain: {information_gain:.2f} nats")
    
    return {
        'surprise_jumps': surprise_jumps,
        'surprise_no_jumps': surprise_no_jumps,
        'surprise_frog_given_jumps': surprise_frog_given_jumps,
        'surprise_apple_given_jumps': surprise_apple_given_jumps,
        'surprise_frog_and_jumps': surprise_frog_and_jumps,
        'surprise_apple_and_jumps': surprise_apple_and_jumps,
        'information_gain': information_gain
    }

def create_frog_apple_generative_model():
    """
    Create generative model for frog vs apple classification
    Variables:
    - x: object type (frog=0, apple=1)
    - y: behavior (doesn't jump=0, jumps=1)
    """
    
    # Create Bayesian Network structure: x -> y
    model = DiscreteBayesianNetwork([('x', 'y')])
    
    # Prior P(x): P(x=frog)=0.1, P(x=apple)=0.9
    cpd_x = TabularCPD(
        variable='x', 
        variable_card=2,
        values=[[0.1],   # P(x=frog)
                [0.9]],  # P(x=apple)
        state_names={'x': ['frog', 'apple']}
    )
    
    # Likelihood P(y|x): P(behavior | object_type)
    cpd_y = TabularCPD(
        variable='y',
        variable_card=2,
        values=[[0.19, 0.99],   # P(y=doesn't jump | x=frog, x=apple)
                [0.81, 0.01]],  # P(y=jumps | x=frog, x=apple)
        evidence=['x'],
        evidence_card=[2],
        state_names={
            'y': ['doesnt_jump', 'jumps'],
            'x': ['frog', 'apple']
        }
    )
    
    # Add CPDs to model
    model.add_cpds(cpd_x, cpd_y)
    
    # Check model validity
    assert model.check_model()
    
    return model

def perform_exact_inference(model):
    """
    Perform exact Bayesian inference using Variable Elimination
    """
    print("Frog vs Apple Generative Model with pgmpy")
    print("=" * 45)
    
    # Create inference object
    inference = VariableElimination(model)
    
    # Print model structure
    print("Model Structure:")
    print(f"Nodes: {model.nodes()}")
    print(f"Edges: {model.edges()}")
    print()
    
    # Display CPDs
    print("Conditional Probability Distributions:")
    print("\nPrior P(x):")
    print(model.get_cpds('x'))
    print("\nLikelihood P(y|x):")
    print(model.get_cpds('y'))
    print()
    
    # Query 1: Prior marginal P(y) - probability of jumping
    print("1. Prior marginal P(y):")
    prior_y = inference.query(['y'])
    print(prior_y)
    print()
    
    # Query 2: Posterior P(x|y=jumps) - what we want to find
    print("2. Posterior P(x | y=jumps):")
    posterior_x_given_jumps = inference.query(['x'], evidence={'y': 'jumps'})
    print(posterior_x_given_jumps)
    print()
    
    # Query 3: Posterior P(x|y=doesn't jump)
    print("3. Posterior P(x | y=doesn't jump):")
    posterior_x_given_no_jumps = inference.query(['x'], evidence={'y': 'doesnt_jump'})
    print(posterior_x_given_no_jumps)
    print()
    
    # Manual verification using Bayes' theorem
    print("4. Manual Bayes' theorem verification:")
    
    # P(x=frog), P(x=apple)
    p_frog = 0.1
    p_apple = 0.9
    
    # P(y=jumps | x=frog), P(y=jumps | x=apple)
    p_jumps_given_frog = 0.81
    p_jumps_given_apple = 0.01
    
    # P(y=jumps) = P(y=jumps|x=frog)*P(x=frog) + P(y=jumps|x=apple)*P(x=apple)
    p_jumps = p_jumps_given_frog * p_frog + p_jumps_given_apple * p_apple
    
    # P(x=frog | y=jumps) = P(y=jumps | x=frog) * P(x=frog) / P(y=jumps)
    p_frog_given_jumps = (p_jumps_given_frog * p_frog) / p_jumps
    p_apple_given_jumps = (p_jumps_given_apple * p_apple) / p_jumps
    
    print(f"P(y=jumps) = {p_jumps:.4f}")
    print(f"P(x=frog | y=jumps) = {p_frog_given_jumps:.4f}")
    print(f"P(x=apple | y=jumps) = {p_apple_given_jumps:.4f}")
    print()
    
    return {
        'prior_y': prior_y,
        'posterior_jumps': posterior_x_given_jumps,
        'posterior_no_jumps': posterior_x_given_no_jumps,
        'manual_verification': {
            'p_frog_given_jumps': p_frog_given_jumps,
            'p_apple_given_jumps': p_apple_given_jumps
        }
    }

def simulate_generative_process(model, n_samples=1000):
    """
    Simulate the generative process: sample from P(x)P(y|x)
    """
    print("5. Generative Model Simulation:")
    print(f"Generating {n_samples} samples from the joint distribution P(x,y)")
    
    # Sample from the model
    samples = model.simulate(n_samples, seed=42)
    
    # Count frequencies
    sample_counts = samples.groupby(['x', 'y'], observed=False).size().unstack(fill_value=0)
    sample_probs = sample_counts / n_samples
    
    print("\nSample frequencies:")
    print(sample_counts)
    print("\nSample probabilities:")
    print(sample_probs)
    
    # Compare with theoretical joint probabilities
    print("\nTheoretical joint probabilities:")
    p_frog_doesnt_jump = 0.1 * 0.19
    p_frog_jumps = 0.1 * 0.81
    p_apple_doesnt_jump = 0.9 * 0.99
    p_apple_jumps = 0.9 * 0.01
    
    theoretical = {
        ('frog', 'doesnt_jump'): p_frog_doesnt_jump,
        ('frog', 'jumps'): p_frog_jumps,
        ('apple', 'doesnt_jump'): p_apple_doesnt_jump,
        ('apple', 'jumps'): p_apple_jumps
    }
    
    for (x_val, y_val), prob in theoretical.items():
        print(f"P(x={x_val}, y={y_val}) = {prob:.4f}")
    
    return samples, theoretical

def visualize_results(results, samples, theoretical, surprises):
    """
    Create visualizations of the Bayesian inference results including surprise
    """
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    # 1. Prior P(x)
    prior_x = [0.1, 0.9]
    axes[0, 0].bar(['Frog', 'Apple'], prior_x, color=['green', 'red'], alpha=0.7)
    axes[0, 0].set_title('Prior P(x)')
    axes[0, 0].set_ylabel('Probability')
    axes[0, 0].set_ylim(0, 1)
    for i, v in enumerate(prior_x):
        axes[0, 0].text(i, v + 0.02, f'{v:.1f}', ha='center', fontweight='bold')
    
    # 2. Likelihood P(y|x) as heatmap
    likelihood_matrix = np.array([[0.19, 0.81],   # frog: [doesn't jump, jumps]
                                 [0.99, 0.01]])   # apple: [doesn't jump, jumps]
    im = axes[0, 1].imshow(likelihood_matrix, cmap='Blues', aspect='auto')
    axes[0, 1].set_title('Likelihood P(y|x)')
    axes[0, 1].set_xticks([0, 1])
    axes[0, 1].set_xticklabels(["Doesn't Jump", 'Jumps'])
    axes[0, 1].set_yticks([0, 1])
    axes[0, 1].set_yticklabels(['Frog', 'Apple'])
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            axes[0, 1].text(j, i, f'{likelihood_matrix[i, j]:.2f}', 
                           ha='center', va='center', fontweight='bold', color='white')
    
    # 3. Prior P(y)
    prior_y_values = results['prior_y'].values
    axes[0, 2].bar(["Doesn't Jump", 'Jumps'], prior_y_values, 
                   color=['orange', 'blue'], alpha=0.7)
    axes[0, 2].set_title('Prior P(y)')
    axes[0, 2].set_ylabel('Probability')
    axes[0, 2].set_ylim(0, 1)
    for i, v in enumerate(prior_y_values):
        axes[0, 2].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
    
    # 4. Surprise for P(y)
    surprise_y_values = [surprises['surprise_no_jumps'], surprises['surprise_jumps']]
    axes[0, 3].bar(["Doesn't Jump", 'Jumps'], surprise_y_values, 
                   color=['orange', 'blue'], alpha=0.7)
    axes[0, 3].set_title('Surprise ℑ(y) [nats]')
    axes[0, 3].set_ylabel('Surprise (nats)')
    for i, v in enumerate(surprise_y_values):
        axes[0, 3].text(i, v + 0.1, f'{v:.2f}', ha='center', fontweight='bold')
    
    # 5. Posterior P(x|y=jumps)
    posterior_jumps_values = results['posterior_jumps'].values
    axes[1, 0].bar(['Frog', 'Apple'], posterior_jumps_values, 
                   color=['green', 'red'], alpha=0.7)
    axes[1, 0].set_title('Posterior P(x | y=jumps)')
    axes[1, 0].set_ylabel('Probability')
    axes[1, 0].set_ylim(0, 1)
    for i, v in enumerate(posterior_jumps_values):
        axes[1, 0].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
    
    # 6. Posterior P(x|y=doesn't jump)
    posterior_no_jumps_values = results['posterior_no_jumps'].values
    axes[1, 1].bar(['Frog', 'Apple'], posterior_no_jumps_values, 
                   color=['green', 'red'], alpha=0.7)
    axes[1, 1].set_title("Posterior P(x | y=doesn't jump)")
    axes[1, 1].set_ylabel('Probability')
    axes[1, 1].set_ylim(0, 1)
    for i, v in enumerate(posterior_no_jumps_values):
        axes[1, 1].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
    
    # 7. Joint distribution P(x,y) from samples
    sample_counts = samples.groupby(['x', 'y'],observed=False).size().unstack(fill_value=0)
    sample_probs = sample_counts / len(samples)
    
    joint_matrix = sample_probs.values
    im2 = axes[1, 2].imshow(joint_matrix, cmap='Greens', aspect='auto')
    axes[1, 2].set_title('Empirical Joint P(x,y)')
    axes[1, 2].set_xticks([0, 1])
    axes[1, 2].set_xticklabels(["Doesn't Jump", 'Jumps'])
    axes[1, 2].set_yticks([0, 1])
    axes[1, 2].set_yticklabels(['Frog', 'Apple'])
    
    # Add text annotations for joint probabilities
    for i in range(2):
        for j in range(2):
            axes[1, 2].text(j, i, f'{joint_matrix[i, j]:.3f}', 
                           ha='center', va='center', fontweight='bold')
    
    # 8. Information gain visualization
    prior_surprise = surprise_function(0.1, 'categorical')
    posterior_surprise = surprises['surprise_frog_given_jumps']
    info_gain = surprises['information_gain']
    
    categories = ['Prior\nℑ(x=frog)', 'Posterior\nℑ(x=frog|y=jumps)', 'Information\nGain']
    values = [prior_surprise, posterior_surprise, info_gain]
    colors = ['red', 'green', 'blue']
    
    bars = axes[1, 3].bar(categories, values, color=colors, alpha=0.7)
    axes[1, 3].set_title('Information Gain Analysis')
    axes[1, 3].set_ylabel('Surprise/Information (nats)')
    for i, v in enumerate(values):
        axes[1, 3].text(i, v + 0.1, f'{v:.2f}', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('tutorial1_frog_apple_bayesian_analysis.png', dpi=150, bbox_inches='tight')

def main():
    """
    Main tutorial function demonstrating Bayesian classification with pgmpy
    """
    print("Tutorial 1: Bayesian Classification with pgmpy")
    print("=" * 55)
    print()
    print("This tutorial demonstrates:")
    print("1. Creating a generative model with pgmpy")
    print("2. Performing exact Bayesian inference")
    print("3. Calculating surprise values")
    print("4. Analyzing information gain")
    print("=" * 55)
    print()
    
    # Create the generative model
    model = create_frog_apple_generative_model()
    
    # Perform exact inference
    results = perform_exact_inference(model)
    
    # Simulate the generative process
    samples, theoretical = simulate_generative_process(model)
    
    # Calculate surprise values
    surprises = calculate_surprises(model, results)
    
    # Create visualizations
    visualize_results(results, samples, theoretical, surprises)
    
    print("\n" + "=" * 55)
    print("Tutorial completed successfully!")
    print("Key takeaways:")
    print("- Bayesian inference provides exact probabilistic reasoning")
    print("- Surprise quantifies how unexpected observations are")
    print("- Information gain measures learning from evidence")
    print("- pgmpy enables exact inference for discrete models")
    print("=" * 55)
    
    return model, results, samples, surprises

if __name__ == "__main__":
    # Run the tutorial
    model, results, samples, surprises = main()
    
    # Additional interactive examples
    print("\nInteractive Examples:")
    print("=" * 25)
    
    # Example 1: What if we observed no jumping?
    print("\nExample 1: What if we observed NO jumping?")
    inference = VariableElimination(model)
    posterior_no_jump = inference.query(['x'], evidence={'y': 'doesnt_jump'})
    print("Posterior P(x | y=doesn't jump):")
    print(posterior_no_jump)
    
    # Calculate surprise for this scenario
    p_frog_given_no_jump = posterior_no_jump.values[0]
    surprise_frog_no_jump = surprise_function(p_frog_given_no_jump, 'categorical')
    print(f"Surprise ℑ(x=frog | y=doesn't jump) = {surprise_frog_no_jump:.2f} nats")
    
    # Example 2: Information theory insight
    print("\nExample 2: Information Theory Insight")
    print("The more surprising an event, the more information it conveys!")
    print(f"Jumping surprise: {surprises['surprise_jumps']:.2f} nats")
    print(f"Not jumping surprise: {surprises['surprise_no_jumps']:.2f} nats")
    print(f"Jumping is {'more' if surprises['surprise_jumps'] > surprises['surprise_no_jumps'] else 'less'} surprising than not jumping")
    
    print("\nTutorial 1 complete! 🐸🍎")