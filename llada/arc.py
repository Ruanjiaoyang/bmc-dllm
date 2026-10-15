import torch
import numpy as np
import random
import re
from gsm8k import GSM8KDataset
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


class ARCDataset(GSM8KDataset):
    """
    Dataset class for ARC-Challenge (AI2 Reasoning Challenge).
    
    ARC-Challenge is a challenging multiple-choice question dataset covering:
    - Physics
    - Chemistry
    - Biology
    - Earth Science
    
    Key characteristics:
    - Grade-school level science questions
    - Multiple choice (typically 3-5 options)
    - Challenge set contains harder questions
    - Requires reasoning beyond simple retrieval
    
    Data source: allenai/ai2_arc with "ARC-Challenge" config
    """
    
    def __init__(
        self,
        tokenizer,
        num_examples=0,
        add_reasoning=True,
        system_prompt=ARC_SYSTEM_PROMPT,
        subsample=-1,
    ):
        super().__init__(tokenizer, num_examples, add_reasoning, system_prompt, subsample)

    def load_test_dataset(self):
        """Load ARC-Challenge test dataset."""
        try:
            # Load the ARC-Challenge dataset
            self.dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
            print(f"Loaded ARC-Challenge test set with {len(self.dataset)} problems")
        except Exception as e:
            print(f"Failed to load ARC-Challenge: {e}")
            raise RuntimeError("Could not load ARC-Challenge dataset")

    def format_question_with_choices(self, item, shuffle_seed=None):
        """
        Format the question with multiple choice options.
        
        Args:
            item: Dataset item
            shuffle_seed: Seed for shuffling choices (for reproducibility)
        
        Returns:
            formatted_question: Formatted question string
            correct_letter: Correct answer letter
        """
        question = item.get("question", "")
        choices = item.get("choices", {})
        correct_answer = item.get("answerKey", "")
        
        # Extract choice labels and texts
        choice_labels = choices.get("label", [])
        choice_texts = choices.get("text", [])
        
        # CRITICAL: ARC answerKey can be EITHER letter ("A", "B") OR number ("1", "2")
        # We need to normalize to find the correct choice
        
        # Create pairs of (original_label, text)
        all_choices = list(zip(choice_labels, choice_texts))
        
        # Find which choice matches the answerKey
        # The answerKey should match one of the labels in choice_labels
        try:
            correct_idx_original = choice_labels.index(correct_answer)
        except ValueError:
            # Fallback: if answerKey doesn't match labels, try to infer
            # Sometimes answerKey is "1" but labels are ["A", "B", "C", "D"]
            # or answerKey is "A" but labels are ["1", "2", "3", "4"]
            if correct_answer.isdigit():
                # answerKey is numeric like "1", convert to 0-indexed
                correct_idx_original = int(correct_answer) - 1
            elif correct_answer.isalpha():
                # answerKey is letter like "A", convert to 0-indexed
                correct_idx_original = ord(correct_answer.upper()) - ord('A')
            else:
                # Unknown format, default to 0
                correct_idx_original = 0
                print(f"Warning: Unknown answerKey format: {correct_answer}")
        
        # Shuffle choices using question hash as seed
        if shuffle_seed is None:
            shuffle_seed = hash(question)
        
        rng = random.Random(shuffle_seed)
        rng.shuffle(all_choices)
        
        # Find the correct answer's new position after shuffling
        # We need to track which original choice was correct
        correct_idx_shuffled = next(
            i for i, (label, _) in enumerate(all_choices) 
            if choice_labels.index(label) == correct_idx_original
        )
        correct_letter = chr(65 + correct_idx_shuffled)  # A, B, C, D, etc.
        
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
            # Load from train split
            train_data = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train")
            
            # Sample examples
            samples = random.sample(range(min(len(train_data), 100)), min(self.num_examples, len(train_data)))
            for idx in samples:
                item = train_data[idx]
                formatted_question, answer = self.format_question_with_choices(item)
                few_shot_examples.append({"question": formatted_question, "answer": answer})
            
            return few_shot_examples
        except Exception as e:
            print(f"Failed to load few-shot examples: {e}")
            return []

    def __getitem__(self, idx):
        """Get a single item from the dataset."""
        item = self.dataset[self.subsample[idx].item()]
        
        question, answer = self.format_question_with_choices(item)
        
        prompt = self.create_prompt(question)
        return prompt, question, answer