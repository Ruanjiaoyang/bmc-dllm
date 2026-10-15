"""
Loader for optimized BMC metric weights.

Reads the weights from the result file produced by eval_bmc/optimize_bmc_weights.py.
"""

import json
from pathlib import Path
from typing import Dict, Optional
import warnings


def load_optimized_weights(
    weights_path: Optional[str] = None,
    fallback_to_default: bool = True
) -> Optional[Dict[str, float]]:
    """
    Load BMC weights from an optimization result file.
    
    Args:
        weights_path: Path to the weight file. One of:
            - the full path to an optimization_results.json file
            - a directory that contains optimization_results.json
            - None (use the default location)
        fallback_to_default: Whether to return the default weights when loading fails
    
    Returns:
        A dict of weights, or None when loading fails and fallback_to_default is False
    """
    
    # Default weights (obtained by optimizing on GSM8K)
    default_weights = {
        "token_accuracy": 0.03382289409637451,
        "semantic_similarity": 0.02513952925801277,
        "number_retention": 0.048321787267923355,
        "final_answer_match": 0.8432632088661194,
        "char_similarity": 0.023441256955266,
        "avg_confidence": 0.026011332869529724
    }
    
    # No path given: fall back to the known default locations
    if weights_path is None:
        possible_paths = [
            # GSM8K weights shipped with the LLaDA experiments in this repo
            Path(__file__).resolve().parents[2] / "llada" / "weights" / "gsm8k.json",
        ]
        
        for path in possible_paths:
            if path.exists():
                weights_path = str(path)
                break
    
    # Still no usable path
    if weights_path is None:
        if fallback_to_default:
            warnings.warn(
                "No optimized weight file found; using the default weights. "
                "Pass --bmc_weights_path to use custom weights."
            )
            return default_weights
        else:
            return None
    
    # Normalize to a Path
    weights_path = Path(weights_path)
    
    # Resolve a directory to the optimization_results.json inside it
    if weights_path.is_dir():
        weights_path = weights_path / "optimization_results.json"
    
    # Check that the file exists
    if not weights_path.exists():
        if fallback_to_default:
            warnings.warn(
                f"Weight file not found: {weights_path}\n"
                f"Using the default weights."
            )
            return default_weights
        else:
            raise FileNotFoundError(f"Weight file not found: {weights_path}")
    
    try:
        # Load the JSON file
        with open(weights_path, 'r') as f:
            data = json.load(f)
        
        # Pick the best weights from the optimization results
        if 'optimized_results' in data and len(data['optimized_results']) > 0:
            # Select the result with the highest ROC-AUC
            best_result = max(
                data['optimized_results'],
                key=lambda x: x.get('metrics', {}).get('roc_auc', 0)
            )
            
            weights = best_result.get('weights', {})
            
            if weights:
                print(f"[OK] Loaded optimized weights: {weights_path}")
                print(f"   method: {best_result.get('method', 'unknown')}")
                print(f"   ROC-AUC: {best_result.get('metrics', {}).get('roc_auc', 0):.4f}")
                return weights
        
        # No usable optimization result in the file
        if fallback_to_default:
            warnings.warn(
                f"Malformed weight file, or no optimization results: {weights_path}\n"
                f"Using the default weights."
            )
            return default_weights
        else:
            raise ValueError(f"Malformed weight file: {weights_path}")
            
    except json.JSONDecodeError as e:
        if fallback_to_default:
            warnings.warn(
                f"Could not parse the weight file: {weights_path}\n"
                f"Error: {e}\n"
                f"Using the default weights."
            )
            return default_weights
        else:
            raise
    except Exception as e:
        if fallback_to_default:
            warnings.warn(
                f"Error while loading the weight file: {weights_path}\n"
                f"Error: {e}\n"
                f"Using the default weights."
            )
            return default_weights
        else:
            raise


def get_default_weights() -> Dict[str, float]:
    """
    Return the default BMC weights (obtained by optimizing on GSM8K).
    
    Returns:
        The default weight dict
    """
    return {
        "token_accuracy": 0.03382289409637451,
        "semantic_similarity": 0.02513952925801277,
        "number_retention": 0.048321787267923355,
        "final_answer_match": 0.8432632088661194,
        "char_similarity": 0.023441256955266,
        "avg_confidence": 0.026011332869529724
    }


if __name__ == "__main__":
    # Smoke-test the loader
    import sys
    
    if len(sys.argv) > 1:
        weights_path = sys.argv[1]
    else:
        weights_path = None
    
    print("=" * 60)
    print("BMC weight loader test")
    print("=" * 60)
    
    weights = load_optimized_weights(weights_path)
    
    if weights:
        print("\nLoaded weights:")
        for key, value in weights.items():
            print(f"  {key}: {value:.6f}")
    else:
        print("\nFailed to load weights")
