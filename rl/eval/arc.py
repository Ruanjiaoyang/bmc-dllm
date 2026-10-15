# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
import random
import re
from datasets import load_dataset
from parsers import Parser

ARC_SYSTEM_PROMPT = """You are an expert in science reasoning.
You will be given a multiple-choice question with options (A, B, C, D, or E).

Respond in the following format:
<reasoning>
Your detailed reasoning and analysis here
</reasoning>
<answer>
\\boxed{X}
</answer>

Where X is one of the available options (A, B, C, D, or E)"""


class ARCDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        tokenizer,
        num_examples=0,
        add_reasoning=True,
        system_prompt=ARC_SYSTEM_PROMPT,
        subsample=-1,
    ):
        self.tokenizer = tokenizer
        self.num_examples = num_examples
        self.add_reasoning = add_reasoning
        self.system_prompt = system_prompt
        self.load_test_dataset()
        self.create_few_shot_prompt()

        self.subsample = (
            np.random.choice(len(self.dataset), subsample, replace=False)
            if subsample != -1
            else np.arange(len(self.dataset))
        )
        print(f"evaluating {len(self.subsample)} examples")
        assert subsample <= len(self.dataset), "Subsample size is greater than dataset size"

    def __len__(self):
        return len(self.subsample)

    def load_test_dataset(self):
        """Load ARC-Challenge test set"""
        try:
            self.dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
            print(f"[ARC] Loaded {len(self.dataset)} test examples from ARC-Challenge")
        except Exception as e:
            raise RuntimeError(f"Could not load ARC-Challenge dataset: {e}")

    def format_question_with_choices(self, item, shuffle_seed=None):
        """
        Format multiple-choice question (NO shuffling to match training framework)
        
        Returns:
            tuple: (formatted_question, correct_letter)
        """
        try:
            question = item.get("question", "")
            choices = item.get("choices", {})
            correct_answer = item.get("answerKey", "")
            
            # Extract choice labels and texts
            choice_labels = choices.get("label", [])
            choice_texts = choices.get("text", [])
            
            # Validate
            if not choice_labels or not choice_texts:
                raise ValueError(f"Missing choice labels or texts")
            
            if len(choice_labels) != len(choice_texts):
                raise ValueError(f"Mismatch between labels and texts")
            
            # Find correct answer (handle different formats)
            if correct_answer in choice_labels:
                correct_letter = correct_answer
            elif correct_answer.isdigit():
                # Convert numeric to letter (1->A, 2->B, etc.)
                idx = int(correct_answer) - 1
                correct_letter = chr(65 + idx) if 0 <= idx < 5 else 'A'
            elif correct_answer.isalpha():
                correct_letter = correct_answer.upper()
            else:
                correct_letter = 'A'  # Default fallback
                print(f"Warning: Unknown answerKey format: {correct_answer}")
            
            # Format the question with choices (using original order and A) format)
            formatted = f"{question}\n\n"
            for label, text in zip(choice_labels, choice_texts):
                formatted += f"{label}) {text}\n"
            
            return formatted.strip(), correct_letter
        
        except Exception as e:
            print(f"ERROR in format_question_with_choices: {e}")
            print(f"Item keys: {list(item.keys())}")
            raise

    def create_prompt(self, input_text):
        """Create prompt for the model"""
        if self.num_examples > 0:
            prompt = f"{self.few_shot_prompt}\n\nQuestion: {input_text}\nAnswer:\n"
        else:
            prompt = input_text
        messages = [{"role": "user", "content": self.system_prompt + "\n\n" + prompt}]
        user_input = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        if self.add_reasoning:
            return user_input + "<reasoning>"
        else:
            return user_input

    def load_few_shot_examples(self):
        """Load few-shot examples from training set"""
        if self.num_examples == 0:
            return []
        
        try:
            train_data = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train")
            samples = random.sample(range(min(len(train_data), 100)), min(self.num_examples, len(train_data)))
            
            few_shot_examples = []
            for idx in samples:
                item = train_data[idx]
                formatted_question, answer = self.format_question_with_choices(item)
                few_shot_examples.append({"question": formatted_question, "answer": answer})
            
            return few_shot_examples
        except Exception as e:
            print(f"[ARC] Failed to load few-shot examples: {e}")
            return []

    def create_few_shot_prompt(self):
        """Create few-shot prompt from dataset examples"""
        few_shot_examples = self.load_few_shot_examples()

        formatted_examples = []
        for example in few_shot_examples:
            input_text = example["question"]
            answer = example["answer"]
            formatted_examples.append(f"Question: {input_text}\nAnswer:\n\\boxed{{{answer}}}")
        self.few_shot_prompt = "\n\n".join(formatted_examples)

    def __getitem__(self, idx):
        """Get a sample"""
        item = self.dataset[self.subsample[idx].item()]
        question, answer = self.format_question_with_choices(item)
        prompt = self.create_prompt(question)
        return prompt, question, answer

    def collate_fn(self, batch):
        """Collate function for DataLoader"""
        prompts = [item[0] for item in batch]
        questions = [item[1] for item in batch]
        answers = [item[2] for item in batch]
        input_ids = self.tokenizer(
            prompts, padding_side="left", return_tensors="pt", padding="longest"
        ).input_ids
        return {"input_ids": input_ids, "questions": questions, "answers": answers, "prompts": prompts}
