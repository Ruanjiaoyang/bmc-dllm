"""
统一的数据集基类 - 多任务适配
定义所有任务共享的接口和方法
"""
import torch
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional


class BaseReasoningDataset(torch.utils.data.Dataset, ABC):
    """
    推理任务数据集的统一基类
    
    所有继承的子类需要实现：
    1. load_test_dataset() - 加载测试数据
    2. extract_answer() - 从生成文本中提取答案
    3. check_correctness() - 验证答案正确性bmc
    4. load_few_shot_examples() - 加载 few-shot 示例（可选）
    """
    
    # 任务名称（子类需要设置）
    TASK_NAME = "base"
    
    def __init__(
        self,
        tokenizer,
        num_examples: int = 0,
        add_reasoning: bool = True,
        system_prompt: str = "",
        subsample: int = -1,
    ):
        """
        Args:
            tokenizer: HuggingFace tokenizer
            num_examples: few-shot 示例数量
            add_reasoning: 是否添加 <reasoning> 标签
            system_prompt: 任务的 system prompt
            subsample: 采样数量（-1 表示全部）
        """
        self.tokenizer = tokenizer
        self.num_examples = num_examples
        self.add_reasoning = add_reasoning
        self.system_prompt = system_prompt
        
        # 1. 加载数据集
        self.load_test_dataset()
        
        # 2. 创建 few-shot prompt
        self.create_few_shot_prompt()
        
        # 3. 采样 - 🔧 修复：先调整 subsample 大小
        import numpy as np
        
        # 🔧 关键修复：如果 subsample 大于数据集大小，自动调整
        if subsample != -1:
            if subsample > len(self.dataset):
                print(f"⚠️  Warning: subsample ({subsample}) > dataset size ({len(self.dataset)})")
                print(f"   Adjusting subsample to {len(self.dataset)}")
                subsample = len(self.dataset)
            
            self.subsample = np.random.choice(
                len(self.dataset), 
                subsample, 
                replace=False
            )
        else:
            self.subsample = np.arange(len(self.dataset))
        
        print(f"[{self.TASK_NAME}] Evaluating {len(self.subsample)} examples")
    
    def __len__(self):
        return len(self.subsample)
    
    @abstractmethod
    def load_test_dataset(self):
        """加载测试数据集 - 子类必须实现"""
        pass
    
    @abstractmethod
    def extract_answer(self, text: str) -> Optional[str]:
        """
        从生成的文本中提取答案
        
        Args:
            text: 模型生成的文本
            
        Returns:
            提取的答案（字符串形式），如果提取失败返回 None
        """
        pass
    
    @abstractmethod
    def check_correctness(
        self, 
        generated_answer: Any, 
        ground_truth: Any,
        question: Optional[str] = None
    ) -> bool:
        """
        检查生成的答案是否正确
        
        Args:
            generated_answer: 从生成文本中提取的答案
            ground_truth: 标准答案
            question: 原始问题（某些任务需要）
            
        Returns:
            是否正确
        """
        pass
    
    def create_prompt(self, input_text: str) -> str:
        """创建完整的输入 prompt"""
        if self.num_examples > 0:
            prompt = f"{self.few_shot_prompt}\n\nQuestion: {input_text}\nAnswer:\n"
        else:
            prompt = input_text
        
        messages = [{
            "role": "user", 
            "content": self.system_prompt + "\n\n" + prompt
        }]
        
        user_input = self.tokenizer.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            tokenize=False
        )
        
        if self.add_reasoning:
            return user_input + "<reasoning>"
        else:
            return user_input
    
    def load_few_shot_examples(self):
        """
        加载 few-shot 示例 - 子类可以重写
        
        Returns:
            List of examples with 'question' and 'answer' keys
        """
        return []
    
    def create_few_shot_prompt(self):
        """创建 few-shot prompt"""
        few_shot_examples = self.load_few_shot_examples()
        
        formatted_examples = []
        for example in few_shot_examples:
            input_text = example["question"]
            answer = example["answer"]
            formatted_examples.append(f"Question: {input_text}\nAnswer:\n{answer}")
        
        self.few_shot_prompt = "\n\n".join(formatted_examples)
    
    @abstractmethod
    def __getitem__(self, idx) -> Tuple[str, str, Any]:
        """
        获取数据样本
        
        Returns:
            (prompt, question, answer)
        """
        pass
    
    def collate_fn(self, batch):
        """批处理函数 - 通用实现"""
        prompts = [item[0] for item in batch]
        questions = [item[1] for item in batch]
        answers = [item[2] for item in batch]
        
        input_ids = self.tokenizer(
            prompts, 
            padding_side="left", 
            return_tensors="pt", 
            padding="longest"
        ).input_ids
        
        return {
            "input_ids": input_ids, 
            "questions": questions, 
            "answers": answers, 
            "prompts": prompts
        }
    
    def get_metrics_info(self) -> Dict[str, Any]:
        """
        返回任务特定的评估信息
        
        Returns:
            包含任务元信息的字典
        """
        return {
            "task_name": self.TASK_NAME,
            "num_examples": self.num_examples,
            "total_samples": len(self.subsample),
        }