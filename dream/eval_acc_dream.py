#!/usr/bin/env python3
"""
Dream模型简化评估脚本 - 只评估准确率，无BMC
支持多GPU并行评估
🔥 修复版：使用与 eval_bmc_dream.py 完全一致的 prompt 格式
"""
import argparse
import json
import math
import os
import random
import sys

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset

# 添加当前目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# 导入答案提取和验证工具
from lm_eval.bmc.task_utils import AnswerExtractor, AnswerValidator


def init_seed(seed):
    """初始化随机种子"""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def setup_ddp():
    """设置分布式训练"""
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    """清理分布式训练"""
    dist.destroy_process_group()


class CustomDistributedSampler(DistributedSampler):
    """自定义分布式采样器"""

    def __init__(
        self,
        dataset,
        num_replicas=None,
        rank=None,
        shuffle=True,
        seed=0,
        drop_last=False,
    ) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(f"Invalid rank {rank}, rank should be in [0, {num_replicas - 1}]")

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last

        if self.drop_last and len(self.dataset) % self.num_replicas != 0:
            self.num_samples = math.ceil((len(self.dataset) - self.num_replicas) / self.num_replicas)
            self.total_size = self.num_samples * self.num_replicas
        else:
            self.total_size = len(self.dataset)
            self.num_samples = len(self.dataset) // self.num_replicas + int(
                rank < (self.total_size % self.num_replicas)
            )

        self.shuffle = shuffle
        self.seed = seed


# ============================================================
# 数据集定义 - 🔥 修复版：使用 chat template + system prompt
# ============================================================

class GSM8KDataset(Dataset):
    """GSM8K数据集 - 使用chat template和<reasoning>格式"""
    
    def __init__(self, tokenizer, subsample=-1):
        self.tokenizer = tokenizer
        self.data = load_dataset("gsm8k", "main", split="test")
        
        if subsample > 0:
            self.data = self.data.select(range(min(subsample, len(self.data))))
        
        # 🔥 System prompt（与 BMC 脚本完全一致）
        self.system_prompt = """You are a math expert. You will be given a question to solve. Solve it step by step. Wrap the final answer in a \\boxed{}. 
Respond in the following format:
<reasoning>
Your reasoning here
</reasoning>
<answer>
\\boxed{...}
</answer>"""
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        question = item["question"]
        answer = item["answer"].split("####")[-1].strip()
        
        # 🔥 使用 chat template（与 BMC 脚本完全一致）
        messages = [{
            "role": "user",
            "content": self.system_prompt + "\n\n" + question
        }]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        # 🔥 添加<reasoning>标签引导模型生成
        prompt += "<reasoning>"
        
        return {
            "question": question,
            "answer": answer,
            "prompt": prompt
        }
    
    def collate_fn(self, batch):
        questions = [item["question"] for item in batch]
        answers = [item["answer"] for item in batch]
        prompts = [item["prompt"] for item in batch]
        
        encoded = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "questions": questions,
            "answers": answers,
            "prompts": prompts
        }


class MATH500Dataset(Dataset):
    """MATH500数学推理数据集 - 使用chat template和<reasoning>格式"""
    
    def __init__(self, tokenizer, subsample=-1):
        self.tokenizer = tokenizer
        self.data = load_dataset("HuggingFaceH4/MATH-500", split="test")
        
        if subsample > 0:
            self.data = self.data.select(range(min(subsample, len(self.data))))
        
        # 🔥 System prompt（与 BMC 脚本完全一致）
        self.system_prompt = """You are a math expert. You will be given a question to solve. Solve it step by step. Wrap the final answer in a \\boxed{}.
Respond in the following format:
<reasoning>
Your reasoning here
</reasoning>
<answer>
\\boxed{...}
</answer>"""
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        question = item["problem"]
        answer = item["answer"]
        
        # 🔥 使用 chat template（与 BMC 脚本完全一致）
        messages = [{
            "role": "user",
            "content": self.system_prompt + "\n\n" + question
        }]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        # 🔥 添加<reasoning>标签引导模型生成
        prompt += "<reasoning>"
        
        return {
            "question": question,
            "answer": answer,
            "prompt": prompt
        }
    
    def collate_fn(self, batch):
        questions = [item["question"] for item in batch]
        answers = [item["answer"] for item in batch]
        prompts = [item["prompt"] for item in batch]
        
        encoded = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=1024,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "questions": questions,
            "answers": answers,
            "prompts": prompts
        }


class GPQADataset(Dataset):
    """GPQA研究生级别科学问答数据集 - 使用chat template和<reasoning>格式"""
    
    def __init__(self, tokenizer, subsample=-1, subset="main"):
        self.tokenizer = tokenizer
        self.subset = subset
        
        # 🔥 System prompt（与 BMC 脚本完全一致）
        self.system_prompt = """You are an expert in graduate-level science (physics, chemistry, biology). You will be given a multiple-choice question with 4 options (A, B, C, D).

These questions are designed to be challenging and require:
- Deep scientific knowledge
- Critical thinking and reasoning
- Careful analysis of each option

Read the question carefully, analyze each option, and select the correct answer.

Respond in the following format:
<reasoning>
Your detailed reasoning and analysis here
</reasoning>
<answer>
\\boxed{X}
</answer>

Where X is one of: A, B, C, or D"""
        
        try:
            if subset == "diamond":
                self.data = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
            elif subset == "extended":
                self.data = load_dataset("Idavidrein/gpqa", "gpqa_extended", split="train")
            else:
                self.data = load_dataset("Idavidrein/gpqa", "gpqa_main", split="train")
        except Exception as e:
            print(f"[GPQA] Load error: {e}, using diamond")
            self.data = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
        
        if subsample > 0:
            self.data = self.data.select(range(min(subsample, len(self.data))))
        
    def __len__(self):
        return len(self.data)
    
    def format_question_with_choices(self, item):
        """格式化多选题（与 BMC 脚本完全一致）"""
        question = item.get("Question", item.get("question", ""))
        
        all_choices = [
            ("Correct", item.get("Correct Answer", "")),
            ("Incorrect", item.get("Incorrect Answer 1", "")),
            ("Incorrect", item.get("Incorrect Answer 2", "")),
            ("Incorrect", item.get("Incorrect Answer 3", ""))
        ]
        
        import random
        rng = random.Random(hash(question))
        rng.shuffle(all_choices)
        
        correct_idx = next(i for i, (label, _) in enumerate(all_choices) if label == "Correct")
        correct_letter = chr(65 + correct_idx)
        
        formatted = f"{question}\n\n"
        for i, (_, choice_text) in enumerate(all_choices):
            formatted += f"{chr(65+i)}. {choice_text}\n"
        
        return formatted.strip(), correct_letter
    
    def __getitem__(self, idx):
        item = self.data[idx]
        formatted_question, answer = self.format_question_with_choices(item)
        
        # 🔥 使用 chat template（与 BMC 脚本完全一致）
        messages = [{
            "role": "user",
            "content": self.system_prompt + "\n\n" + formatted_question
        }]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        # 🔥 添加<reasoning>标签引导模型生成
        prompt += "<reasoning>"
        
        return {
            "question": formatted_question,
            "answer": answer,
            "prompt": prompt
        }
    
    def collate_fn(self, batch):
        questions = [item["question"] for item in batch]
        answers = [item["answer"] for item in batch]
        prompts = [item["prompt"] for item in batch]
        
        encoded = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=1024,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "questions": questions,
            "answers": answers,
            "prompts": prompts
        }


class ARCDataset(Dataset):
    """ARC-Challenge科学问答数据集 - 使用chat template和<reasoning>格式"""
    
    def __init__(self, tokenizer, subsample=-1):
        self.tokenizer = tokenizer
        
        # 🔥 System prompt（与 BMC 脚本完全一致）
        self.system_prompt = """You are an expert in science reasoning.
You will be given a multiple-choice question with options (A, B, C, D, or E).

Respond in the following format:
<reasoning>
Your detailed reasoning and analysis here
</reasoning>
<answer>
\\boxed{X}
</answer>

Where X is one of the available options (A, B, C, D, or E)"""
        
        try:
            self.data = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
        except Exception as e:
            raise RuntimeError(f"Could not load ARC-Challenge: {e}")
        
        if subsample > 0:
            self.data = self.data.select(range(min(subsample, len(self.data))))
        
    def __len__(self):
        return len(self.data)
    
    def format_question_with_choices(self, item):
        """格式化多选题（与 BMC 脚本完全一致）"""
        question = item.get("question", "")
        choices = item.get("choices", {})
        correct_answer = item.get("answerKey", "")
        
        choice_labels = choices.get("label", [])
        choice_texts = choices.get("text", [])
        
        if not choice_labels or not choice_texts:
            raise ValueError(f"Missing choice labels or texts")
        
        if len(choice_labels) != len(choice_texts):
            raise ValueError(f"Mismatch between labels and texts")
        
        if correct_answer in choice_labels:
            correct_letter = correct_answer
        elif correct_answer.isdigit():
            idx = int(correct_answer) - 1
            correct_letter = chr(65 + idx) if 0 <= idx < 5 else 'A'
        else:
            correct_letter = correct_answer.upper() if correct_answer.isalpha() else 'A'
        
        formatted = f"{question}\n\n"
        for label, text in zip(choice_labels, choice_texts):
            formatted += f"{label}) {text}\n"
        
        return formatted.strip(), correct_letter
    
    def __getitem__(self, idx):
        item = self.data[idx]
        formatted_question, answer = self.format_question_with_choices(item)
        
        # 🔥 使用 chat template（与 BMC 脚本完全一致）
        messages = [{
            "role": "user",
            "content": self.system_prompt + "\n\n" + formatted_question
        }]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        # 🔥 添加<reasoning>标签引导模型生成
        prompt += "<reasoning>"
        
        return {
            "question": formatted_question,
            "answer": answer,
            "prompt": prompt
        }
    
    def collate_fn(self, batch):
        questions = [item["question"] for item in batch]
        answers = [item["answer"] for item in batch]
        prompts = [item["prompt"] for item in batch]
        
        encoded = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=1024,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "questions": questions,
            "answers": answers,
            "prompts": prompts
        }


# ============================================================
# 答案提取和验证
# ============================================================

def extract_answer(text, task_name):
    """从生成的文本中提取答案"""
    try:
        if task_name == "gsm8k":
            return AnswerExtractor.extract_gsm8k(text)
        elif task_name == "math":
            return AnswerExtractor.extract_math500(text)
        elif task_name == "arc":
            return AnswerExtractor.extract_arc(text)
        elif task_name == "gpqa":
            return AnswerExtractor.extract_gpqa(text)
        else:
            return text
    except Exception as e:
        return None


def validate_answer(generated_answer, gt_answer, task_name, question=None):
    """验证答案是否正确"""
    try:
        if task_name == "gsm8k":
            is_correct, _ = AnswerValidator.validate_gsm8k(generated_answer, gt_answer)
        elif task_name == "math":
            is_correct, _ = AnswerValidator.validate_math500(generated_answer, gt_answer)
        elif task_name == "arc":
            is_correct, _ = AnswerValidator.validate_arc(generated_answer, gt_answer)
        elif task_name == "gpqa":
            is_correct, _ = AnswerValidator.validate_gpqa(generated_answer, gt_answer)
        else:
            is_correct = str(generated_answer).strip() == str(gt_answer).strip()
        
        return bool(is_correct)
    except Exception as e:
        if dist.get_rank() == 0:
            print(f"⚠️  Validation error: {e}")
        return False


# ============================================================
# 评估函数
# ============================================================

def evaluate(
    model,
    tokenizer,
    dataloader,
    task_name,
    gen_length=256,
    temperature=0.1,
    top_p=0.9,
    diffusion_steps=64,
    alg="entropy",
):
    """评估函数"""
    model.eval()
    
    all_results = []
    total_correct = 0
    total_samples = 0
    
    device = model.device
    
    for batch in tqdm(dataloader, disable=(dist.get_rank() != 0)):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        gt_answers = batch["answers"]
        questions = batch["questions"]
        prompts = batch["prompts"]
        
        # 使用Dream模型生成
        generation_output = model.diffusion_generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=gen_length,
            steps=diffusion_steps,
            temperature=temperature,
            top_p=top_p,
            alg=alg,
            return_dict_in_generate=True,
            output_history=False,
        )
        
        # 解码生成文本
        generated_texts = []
        for i, (prompt_ids, gen_ids) in enumerate(zip(input_ids, generation_output.sequences)):
            actual_prompt_len = (prompt_ids != tokenizer.pad_token_id).sum().item()
            gen_text = tokenizer.decode(
                gen_ids[actual_prompt_len:].tolist(), 
                skip_special_tokens=True
            )
            generated_texts.append(gen_text)
        
        # 提取答案并验证
        for j in range(len(gt_answers)):
            generated_answer = extract_answer(generated_texts[j], task_name)
            gt_answer = gt_answers[j]
            
            is_correct = validate_answer(generated_answer, gt_answer, task_name, questions[j])
            
            total_samples += 1
            if is_correct:
                total_correct += 1
            
            result = {
                "question": questions[j],
                "prompt": prompts[j],
                "generation": generated_texts[j],
                "generated_answer": str(generated_answer) if generated_answer is not None else "",
                "ground_truth": str(gt_answer),
                "is_correct": is_correct,
            }
            
            all_results.append(result)
        
        # 打印示例
        if dist.get_rank() == 0 and len(questions) > 0:
            idx = random.randint(0, len(questions) - 1)
            print(f"\n{'='*60}")
            print(f"Question: {questions[idx][:100]}...")
            print(f"Generation: {generated_texts[idx][:200]}...")
            print(f"Ground truth: {str(gt_answers[idx])[:50]}")
            print(f"Generated answer: {extract_answer(generated_texts[idx], task_name)}")
            print(f"Is correct: {all_results[-len(questions) + idx]['is_correct']}")
            print(f"{'='*60}\n")
    
    # 计算准确率
    accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    
    if dist.get_rank() == 0:
        print(f"\n{'='*60}")
        print(f"📊 Task: {task_name.upper()}")
        print(f"  Total: {total_samples}")
        print(f"  Correct: {total_correct}")
        print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"{'='*60}\n")
    
    return {
        "results": all_results,
        "total_samples": total_samples,
        "total_correct": total_correct,
        "accuracy": accuracy,
    }


# ============================================================
# 主函数
# ============================================================

if __name__ == "__main__":
    init_seed(42)
    local_rank = setup_ddp()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--dataset", type=str, 
                       choices=["gsm8k", "math", "arc", "gpqa"], 
                       required=True)
    parser.add_argument("--gen_length", type=int, default=256)
    parser.add_argument("--diffusion_steps", type=int, default=64,
                       help="Diffusion steps (recommend: same as gen_length)")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--alg", type=str, default="entropy")
    parser.add_argument("--output_dir", type=str, default="results_acc")
    parser.add_argument("--subsample", type=int, default=-1)
    parser.add_argument("--gpqa_subset", type=str, default="main")
    
    args = parser.parse_args()
    
    # 加载模型
    if dist.get_rank() == 0:
        print(f"\n{'='*60}")
        print("Loading Dream model...")
        print(f"Model: {args.model_path}")
        print(f"🔥 Using chat template + system prompt (BMC-aligned)")
        print(f"{'='*60}\n")
    
    model = AutoModel.from_pretrained(
        args.model_path, 
        trust_remote_code=True, 
        torch_dtype=torch.bfloat16
    ).to(local_rank)
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, 
        trust_remote_code=True
    )
    
    # 准备数据集
    if args.dataset == "gsm8k":
        dataset = GSM8KDataset(tokenizer, args.subsample)
    elif args.dataset == "math":
        dataset = MATH500Dataset(tokenizer, args.subsample)
    elif args.dataset == "gpqa":
        dataset = GPQADataset(tokenizer, args.subsample, args.gpqa_subset)
    elif args.dataset == "arc":
        dataset = ARCDataset(tokenizer, args.subsample)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=CustomDistributedSampler(dataset, shuffle=False),
        collate_fn=dataset.collate_fn,
    )
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 运行评估
    metrics = evaluate(
        model,
        tokenizer,
        dataloader,
        args.dataset,
        gen_length=args.gen_length,
        temperature=args.temperature,
        top_p=args.top_p,
        diffusion_steps=args.diffusion_steps,
        alg=args.alg,
    )
    
    # 保存结果
    filename = f"{args.output_dir}/{args.dataset}_len{args.gen_length}_{dist.get_rank()}_results.json"
    
    with open(filename, "w") as f:
        json.dump({
            "results": metrics["results"],
            "metrics": {
                "total_samples": metrics["total_samples"],
                "total_correct": metrics["total_correct"],
                "accuracy": metrics["accuracy"],
            },
            "config": {
                "model_path": args.model_path,
                "dataset": args.dataset,
                "gen_length": args.gen_length,
                "diffusion_steps": args.diffusion_steps,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "alg": args.alg,
                "prompt_format": "chat_template_with_reasoning_tag",  # 🔥 标记使用的格式
            }
        }, f, indent=2)
    
    if dist.get_rank() == 0:
        print(f"✓ Results saved to: {filename}")
    
    cleanup_ddp()