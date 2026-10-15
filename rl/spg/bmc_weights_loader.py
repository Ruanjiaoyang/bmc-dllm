"""
BMC优化权重加载工具

从eval_bmc/optimize_bmc_weights.py生成的结果文件中自动加载优化权重
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
    从优化结果文件中加载BMC权重
    
    Args:
        weights_path: 权重文件路径，可以是：
            - optimization_results.json 文件的完整路径
            - 包含optimization_results.json的目录路径
            - None（使用默认路径）
        fallback_to_default: 如果加载失败，是否返回默认权重
    
    Returns:
        权重字典，如果加载失败且fallback_to_default=False则返回None
    """
    
    # 默认权重（从GSM8K优化得到）
    default_weights = {
        "token_accuracy": 0.03382289409637451,
        "semantic_similarity": 0.02513952925801277,
        "number_retention": 0.048321787267923355,
        "final_answer_match": 0.8432632088661194,
        "char_similarity": 0.023441256955266,
        "avg_confidence": 0.026011332869529724
    }
    
    # 如果没有指定路径，尝试默认位置
    if weights_path is None:
        # 尝试几个可能的默认位置
        possible_paths = [
            # 仓库内随 LLaDA 实验发布的 GSM8K 权重
            Path(__file__).resolve().parents[2] / "llada" / "weights" / "gsm8k.json",
        ]
        
        for path in possible_paths:
            if path.exists():
                weights_path = str(path)
                break
    
    # 如果还是没有找到路径
    if weights_path is None:
        if fallback_to_default:
            warnings.warn(
                "未找到优化权重文件，使用默认权重。"
                "如需使用自定义权重，请通过--bmc_weights_path参数指定路径。"
            )
            return default_weights
        else:
            return None
    
    # 转换为Path对象
    weights_path = Path(weights_path)
    
    # 如果是目录，查找optimization_results.json
    if weights_path.is_dir():
        weights_path = weights_path / "optimization_results.json"
    
    # 检查文件是否存在
    if not weights_path.exists():
        if fallback_to_default:
            warnings.warn(
                f"权重文件不存在: {weights_path}\n"
                f"使用默认权重。"
            )
            return default_weights
        else:
            raise FileNotFoundError(f"权重文件不存在: {weights_path}")
    
    try:
        # 加载JSON文件
        with open(weights_path, 'r') as f:
            data = json.load(f)
        
        # 从优化结果中提取最佳权重
        if 'optimized_results' in data and len(data['optimized_results']) > 0:
            # 找到ROC-AUC最高的结果
            best_result = max(
                data['optimized_results'],
                key=lambda x: x.get('metrics', {}).get('roc_auc', 0)
            )
            
            weights = best_result.get('weights', {})
            
            if weights:
                print(f"✅ 成功加载优化权重: {weights_path}")
                print(f"   方法: {best_result.get('method', 'unknown')}")
                print(f"   ROC-AUC: {best_result.get('metrics', {}).get('roc_auc', 0):.4f}")
                return weights
        
        # 如果没有找到优化结果
        if fallback_to_default:
            warnings.warn(
                f"权重文件格式不正确或没有优化结果: {weights_path}\n"
                f"使用默认权重。"
            )
            return default_weights
        else:
            raise ValueError(f"权重文件格式不正确: {weights_path}")
            
    except json.JSONDecodeError as e:
        if fallback_to_default:
            warnings.warn(
                f"无法解析权重文件: {weights_path}\n"
                f"错误: {e}\n"
                f"使用默认权重。"
            )
            return default_weights
        else:
            raise
    except Exception as e:
        if fallback_to_default:
            warnings.warn(
                f"加载权重文件时出错: {weights_path}\n"
                f"错误: {e}\n"
                f"使用默认权重。"
            )
            return default_weights
        else:
            raise


def get_default_weights() -> Dict[str, float]:
    """
    获取默认的BMC权重（从GSM8K优化得到）
    
    Returns:
        默认权重字典
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
    # 测试加载功能
    import sys
    
    if len(sys.argv) > 1:
        weights_path = sys.argv[1]
    else:
        weights_path = None
    
    print("=" * 60)
    print("BMC权重加载测试")
    print("=" * 60)
    
    weights = load_optimized_weights(weights_path)
    
    if weights:
        print("\n加载的权重:")
        for key, value in weights.items():
            print(f"  {key}: {value:.6f}")
    else:
        print("\n未能加载权重")
