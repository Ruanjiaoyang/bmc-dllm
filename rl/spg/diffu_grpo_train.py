# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import wandb
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
from trl import TrlParser, ModelConfig
from peft import LoraConfig

# Custom imports
from diffu_grpo_trainer import DiffuGRPOTrainer
from spg_trainer import SPGTrainer
from diffu_grpo_config import DiffuGRPOConfig
from reward_func import (
    xmlcount_reward_func,
    soft_format_reward_func,
    strict_format_reward_func,
    int_reward_func,
    correctness_reward_func,
    countdown_reward_func,
    correctness_reward_func_math,
    sudoku_reward_func,
    boxed_and_answer_tags_format_reward,
    reward_len,
    bmc_reward_gsm8k,
    bmc_reward_math,
    bmc_reward_olympic,
    bmc_reward_gpqa,
    bmc_reward_arc,
    correctness_reward_func_olympic,
    correctness_reward_func_gpqa,
    correctness_reward_func_arc,
    initialize_bmc_probe,
)
from data_utils import (
    get_gsm8k_questions,
    get_countdown_questions,
    get_sudoku_questions,
    get_sudoku_questions_new,
    set_random_seed,
    get_math_questions,
    get_olympic_questions,
    get_gpqa_questions,
    get_arc_questions,
)


def main(grpo_config, model_config):
    import torch.distributed as dist

    # 🔥 添加进程标识
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    print(f"{'=' * 60}")
    print(f"[Rank {rank}/{world_size}] 🚀 Starting main function")
    print(f"{'=' * 60}")

    # Set seed for reproducibility
    set_random_seed(grpo_config.seed)
    print(f"[Rank {rank}] ✅ Random seed set: {grpo_config.seed}")

    # Load dataset based on configuration
    print(f"[Rank {rank}] 📊 Loading dataset: {grpo_config.dataset}")

    # Load dataset based on configuration
    if grpo_config.dataset == "gsm8k":
        dataset = get_gsm8k_questions("train")

        # 🔥 选择是否使用BMC增强的奖励函数
        # 通过grpo_config.use_bmc参数控制（需要在配置中添加）
        use_bmc = getattr(grpo_config, 'use_bmc', False)

        if use_bmc:
            print("✅ 使用BMC增强的奖励函数")
            reward_functions = [
                xmlcount_reward_func,
                soft_format_reward_func,
                strict_format_reward_func,
                int_reward_func,
                bmc_reward_gsm8k,  # 🔥 使用BMC增强的奖励函数
            ]
        else:
            print("使用标准奖励函数")
            reward_functions = [
                xmlcount_reward_func,
                soft_format_reward_func,
                strict_format_reward_func,
                int_reward_func,
                correctness_reward_func,
            ]
    elif grpo_config.dataset == "countdown":
        dataset = get_countdown_questions("train")
        reward_functions = [countdown_reward_func]
    # elif grpo_config.dataset == "sudoku":
    #     dataset = get_sudoku_questions()
    #     reward_functions = [sudoku_reward_func]
    elif grpo_config.dataset == "sudoku_new":
        dataset = get_sudoku_questions_new(few_shot=grpo_config.few_shot)
        reward_functions = [sudoku_reward_func]
    elif grpo_config.dataset == "math":
        dataset = get_math_questions("train")

        # 🔥 选择是否使用BMC增强的奖励函数
        use_bmc = getattr(grpo_config, 'use_bmc', False)

        if use_bmc:
            print("✅ 使用BMC增强的奖励函数 (MATH)")
            reward_functions = [
                bmc_reward_math,  # 🔥 使用BMC增强的MATH奖励函数
                boxed_and_answer_tags_format_reward,
            ]
        else:
            print("使用标准奖励函数 (MATH)")
            reward_functions = [
                correctness_reward_func_math,
                boxed_and_answer_tags_format_reward,
            ]
    elif grpo_config.dataset == "olympic":
        dataset = get_olympic_questions("train")

        # 🔥 选择是否使用BMC增强的奖励函数
        use_bmc = getattr(grpo_config, 'use_bmc', False)

        if use_bmc:
            print("✅ 使用BMC增强的奖励函数 (Olympic)")
            reward_functions = [
                bmc_reward_olympic,  # 🔥 使用BMC增强的Olympic奖励函数
                boxed_and_answer_tags_format_reward,
            ]
        else:
            print("使用标准奖励函数 (Olympic)")
            reward_functions = [
                correctness_reward_func_olympic,
                boxed_and_answer_tags_format_reward,
            ]
    elif grpo_config.dataset == "gpqa":
        dataset = get_gpqa_questions("train", subset="main")

        # 🔥 选择是否使用BMC增强的奖励函数
        use_bmc = getattr(grpo_config, 'use_bmc', False)

        if use_bmc:
            print("✅ 使用BMC增强的奖励函数 (GPQA)")
            reward_functions = [
                bmc_reward_gpqa,  # 🔥 使用BMC增强的GPQA奖励函数
            ]
        else:
            print("使用标准奖励函数 (GPQA)")
            reward_functions = [
                correctness_reward_func_gpqa,
            ]
    elif grpo_config.dataset == "arc":
        dataset = get_arc_questions("train")

        # 🔥 选择是否使用BMC增强的奖励函数
        use_bmc = getattr(grpo_config, 'use_bmc', False)

        if use_bmc:
            print("✅ 使用BMC增强的奖励函数 (ARC)")
            reward_functions = [
                bmc_reward_arc,  # 🔥 使用BMC增强的ARC奖励函数
            ]
        else:
            print("使用标准奖励函数 (ARC)")
            reward_functions = [
                correctness_reward_func_arc,
            ]

    # Shuffle dataset with fixed seed for reproducibility
    dataset = dataset.shuffle(seed=grpo_config.seed)
    print(f"[Rank {rank}] ✅ Dataset shuffled")

    # 🔥 同步所有进程
    if dist.is_initialized():
        dist.barrier()
        print(f"[Rank {rank}] ✅ Passed dataset shuffle barrier")

    # Split dataset if needed
    if grpo_config.dataset in ["countdown", "sudoku", "sudoku_new"]:
        train_set = dataset.select(range(0, len(dataset) - 500))  # Leave last 500 for evaluation
    else:
        train_set = dataset
    print(f"[Rank {rank}] ✅ Train set ready: {len(train_set)} samples")

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 4 bit quantization configuration
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # Load model and tokenizer
    model = AutoModel.from_pretrained(
        grpo_config.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(grpo_config.model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.use_cache = False

    # Configure LoRA for parameter-efficient fine-tuning
    peft_config = LoraConfig(
        r=model_config.lora_r,
        lora_alpha=model_config.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
        task_type="CAUSAL_LM",
        lora_dropout=model_config.lora_dropout,
    )

    # 🔥 初始化BMC探针（如果使用BMC奖励函数）
    use_bmc = getattr(grpo_config, 'use_bmc', False)
    if use_bmc and grpo_config.dataset in ["gsm8k", "math", "olympic", "gpqa", "arc"]:
        print("\n" + "=" * 60)
        print("初始化BMC探针...")
        print("=" * 60)

        # 计算总训练步数
        # total_steps = (dataset_size / batch_size) * num_epochs
        dataset_size = len(train_set)
        batch_size = grpo_config.per_device_train_batch_size * grpo_config.gradient_accumulation_steps
        if torch.cuda.device_count() > 1:
            batch_size *= torch.cuda.device_count()

        # 如果设置了max_steps，使用max_steps；否则根据num_epochs计算
        if hasattr(grpo_config, 'max_steps') and grpo_config.max_steps > 0:
            total_steps = grpo_config.max_steps
        else:
            steps_per_epoch = dataset_size // batch_size
            total_steps = steps_per_epoch * grpo_config.num_train_epochs

        print(f"数据集大小: {dataset_size}")
        print(f"有效批次大小: {batch_size}")
        print(f"预计总步数: {total_steps}")

        # 训练配置
        training_config = {
            'total_steps': total_steps,
            'num_epochs': grpo_config.num_train_epochs,
            'dataset_size': dataset_size,
            'dataset': grpo_config.dataset,  # 🔥 添加数据集名称，用于BMC任务类型推断
        }

        # BMC配置参数（可以通过grpo_config传入）
        bmc_kwargs = {
            'adaptive_t': getattr(grpo_config, 'bmc_adaptive_t', True),
            'num_samples': getattr(grpo_config, 'bmc_num_samples', 1),
            'aggressive_scoring': getattr(grpo_config, 'bmc_aggressive_scoring', False),
        }

        # 优先级：手动指定的权重 > 从文件加载的权重 > aggressive_scoring
        if hasattr(grpo_config, 'bmc_optimized_weights') and grpo_config.bmc_optimized_weights:
            # 优先使用手动指定的权重
            bmc_kwargs['optimized_weights'] = grpo_config.bmc_optimized_weights
            print("✅ 使用手动指定的优化权重")
        elif hasattr(grpo_config, 'bmc_weights_path') and grpo_config.bmc_weights_path:
            # 从文件自动加载权重
            from bmc_weights_loader import load_optimized_weights
            try:
                loaded_weights = load_optimized_weights(
                    weights_path=grpo_config.bmc_weights_path,
                    fallback_to_default=True
                )
                if loaded_weights:
                    bmc_kwargs['optimized_weights'] = loaded_weights
                    print("✅ 从文件自动加载优化权重")
                    print(f"   路径: {grpo_config.bmc_weights_path}")
            except Exception as e:
                print(f"⚠️  加载权重文件失败: {e}")
                print(f"   将使用aggressive_scoring策略")
        else:
            print("ℹ️  使用aggressive_scoring策略")

        # 初始化BMC探针
        initialize_bmc_probe(
            model=model,
            tokenizer=tokenizer,
            device=device,
            training_config=training_config,
            **bmc_kwargs
        )
        print("=" * 60 + "\n")

    if grpo_config.trainer == "diffu_grpo":
        # Initialize and run trainer
        trainer = DiffuGRPOTrainer(
            args=grpo_config,
            model=model,
            peft_config=peft_config,
            reward_funcs=reward_functions,
            train_dataset=train_set,
        )
    elif grpo_config.trainer == "spg":
        trainer = SPGTrainer(
            args=grpo_config,
            model=model,
            peft_config=peft_config,
            reward_funcs=reward_functions,
            train_dataset=train_set,
        )
    else:
        raise ValueError(f"Invalid trainer: {grpo_config.trainer}")

    trainer.train()


if __name__ == "__main__":
    parser = TrlParser((DiffuGRPOConfig, ModelConfig))
    grpo_config, model_config = parser.parse_args_and_config()
    main(grpo_config=grpo_config, model_config=model_config)