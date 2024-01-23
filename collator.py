from transformers import AutoTokenizer
import torch

class MultimodalCollator:
    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer

    def tokenize_text(self, texts):
        encoded_text = self.tokenizer(
            text=texts,
            max_length=60,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            return_attention_mask=True)

        # Avoid squeezing if batch size is 1
        return {
            "input_ids": encoded_text['input_ids'],
            "attention_mask": encoded_text['attention_mask']
        }

    def tokenize_answer(self, texts):
        encoded_text = self.tokenizer(
            text=texts,
            max_length=60,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            return_attention_mask=True)

        encoded_text['input_ids'][encoded_text['input_ids'] == 1] = -1

        # Avoid squeezing if batch size is 1
        return {
            "labels": encoded_text['input_ids']
        }

    def preprocess_images(self, images):
        # Ensure that images is a tensor
        images = torch.stack(images) if not isinstance(images, torch.Tensor) else images
        return {
            "pixel_values": images
        }

    def __call__(self, examples):
        # Check if input is a single example or a batch
        if isinstance(examples, dict):  # Single example
            questions = [examples['question']]
            answers = [examples['answer']]
            images = [examples['image']]
        else:  # Batch of examples
            questions = [ex['question'] for ex in examples]
            answers = [ex['answer'] for ex in examples]
            images = torch.stack([ex['image'] for ex in examples])

        return {
            **self.tokenize_text(questions),
            **self.tokenize_answer(answers),
            **self.preprocess_images(images)
        }