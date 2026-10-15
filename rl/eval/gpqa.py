import torch
import numpy as np
import random
import re
from gsm8k import GSM8KDataset
from datasets import load_dataset
from parsers import Parser

GPQA_SYSTEM_PROMPT = """You are an expert in graduate-level science (physics, chemistry, biology). You will be given a multiple-choice question with 4 options (A, B, C, D).

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


class GPQADataset(GSM8KDataset):
    """
    Dataset class for GPQA (Graduate-Level Google-Proof Q&A Benchmark).
    
    GPQA is a challenging multiple-choice question dataset covering:
    - Physics
    - Chemistry
    - Biology
    
    Key characteristics:
    - Graduate-level difficulty
    - 4 options per question (A, B, C, D)
    - Requires deep domain knowledge
    - Designed to be "Google-proof" (hard to find answers online)
    
    Data sources (with fallback):
    1. Idavidrein/gpqa (primary, main split)
    2. GPQA diamond subset (highest quality)
    3. GPQA extended subset (larger set)
    """
    
    def __init__(
        self,
        tokenizer,
        num_examples=0,
        add_reasoning=True,
        system_prompt=GPQA_SYSTEM_PROMPT,
        subsample=-1,
        subset="main",  # Options: "main", "diamond", "extended"
    ):
        self.subset = subset
        super().__init__(tokenizer, num_examples, add_reasoning, system_prompt, subsample)

    def load_test_dataset(self):
        """Load GPQA test dataset with fallback mechanism."""
        try:
            # Primary: Load the main GPQA dataset
            if self.subset == "diamond":
                self.dataset = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
                print(f"Loaded GPQA Diamond dataset with {len(self.dataset)} problems")
            elif self.subset == "extended":
                self.dataset = load_dataset("Idavidrein/gpqa", "gpqa_extended", split="train")
                print(f"Loaded GPQA Extended dataset with {len(self.dataset)} problems")
            else:
                # Default to main
                self.dataset = load_dataset("Idavidrein/gpqa", "gpqa_main", split="train")
                print(f"Loaded GPQA Main dataset with {len(self.dataset)} problems")
        except Exception as e:
            print(f"Failed to load GPQA {self.subset}: {e}")
            try:
                # Fallback to diamond (highest quality subset)
                self.dataset = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
                print(f"Loaded GPQA Diamond dataset (fallback) with {len(self.dataset)} problems")
            except Exception as e:
                print(f"Failed to load GPQA diamond: {e}")
                try:
                    # Final fallback to extended
                    self.dataset = load_dataset("Idavidrein/gpqa", "gpqa_extended", split="train")
                    print(f"Loaded GPQA Extended dataset (fallback) with {len(self.dataset)} problems")
                except Exception as e:
                    print(f"Failed to load any GPQA dataset: {e}")
                    raise RuntimeError("Could not load GPQA dataset")

    def format_question_with_choices(self, item, shuffle_seed=None):
        """
        Format the question with multiple choice options.
        
        Args:
            item: Dataset item
            shuffle_seed: Seed for shuffling choices (for reproducibility)
        
        Returns:
            formatted_question: Formatted question string
            correct_letter: Correct answer letter (A/B/C/D)
        """
        question = item.get("Question", item.get("question", ""))
        
        # Collect all choices with labels
        all_choices = [
            ("Correct", item.get("Correct Answer", "")),
            ("Incorrect", item.get("Incorrect Answer 1", "")),
            ("Incorrect", item.get("Incorrect Answer 2", "")),
            ("Incorrect", item.get("Incorrect Answer 3", ""))
        ]
        
        # Shuffle choices using the question hash as seed
        # This ensures:
        # 1. Each question has a fixed but different order (reproducible)
        # 2. Different questions have different orders (avoid position bias)
        if shuffle_seed is None:
            shuffle_seed = hash(question)
        
        rng = random.Random(shuffle_seed)
        rng.shuffle(all_choices)
        
        # Find the correct answer's new position
        correct_idx = next(i for i, (label, _) in enumerate(all_choices) 
                        if label == "Correct")
        correct_letter = chr(65 + correct_idx)  # A, B, C, or D
        
        # Format the question
        formatted = f"{question}\n\n"
        for i, (_, choice_text) in enumerate(all_choices):
            formatted += f"{chr(65 + i)}. {choice_text}\n"
        
        return formatted.strip(), correct_letter


    def load_few_shot_examples(self):
        """Load few-shot examples for in-context learning."""
        if self.num_examples == 0:
            return []
        
        few_shot_examples = []
        
        try:
            # Try to load from the same subset
            if self.subset == "diamond":
                train_data = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
            elif self.subset == "extended":
                train_data = load_dataset("Idavidrein/gpqa", "gpqa_extended", split="train")
            else:
                train_data = load_dataset("Idavidrein/gpqa", "gpqa_main", split="train")
            
            # Sample examples
            samples = random.sample(range(min(len(train_data), 100)), min(self.num_examples, len(train_data)))
            for idx in samples:
                item = train_data[idx]
                # Use the shuffling formatter so few-shot examples match the eval format
                formatted_question, answer = self.format_question_with_choices(item)
                few_shot_examples.append({"question": formatted_question, "answer": answer})
            
            return few_shot_examples
        except Exception as e:
            print(f"Failed to load few-shot examples: {e}")
            return []


    def __getitem__(self, idx):
        """Get a single item from the dataset."""
        item = self.dataset[self.subsample[idx].item()]
        
        # The formatter returns (question, correct_letter)
        question, answer = self.format_question_with_choices(item)
        
        prompt = self.create_prompt(question)
        return prompt, question, answer
