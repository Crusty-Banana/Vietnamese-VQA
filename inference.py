import json
import torch
import logging
import os
from tqdm import tqdm

def inference(model, dataset, tokenizer, inference_dir, inference_name, device):
    model.eval()  # Set the model to evaluation mode
    predictions = {}

    # Configure logging
    log_file_path = os.path.join(inference_dir, f'{inference_name}.log')
    logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(message)s')

    logger = logging.getLogger()

    # Using tqdm for progress bar
    for data in tqdm(dataset, desc="Processing", unit="sample"):
        id, image, question = data['id'], data['image'], data['question']

        # Tokenize the question
        question_tok = tokenizer(question, max_length=60, padding='max_length', return_tensors="pt", return_attention_mask=True)
        input_ids = question_tok["input_ids"].to(device)
        attention_mask = question_tok["attention_mask"].to(device)

        # Predict
        with torch.no_grad():
            output = model(input_ids, pixel_values=image.unsqueeze(0).to(device), attention_mask=attention_mask)
        logit = output['logits']
        pred = logit.argmax(dim=2)
        pred_word = tokenizer.batch_decode(pred, skip_special_tokens=True)[0]

        predictions[id] = pred_word

        # Log the question and its predicted answer
        logger.info(f"Question: {question}")
        logger.info(f"Predicted Answer: {pred_word}\n")

    # Write predictions to file
    file_path = os.path.join(inference_dir, f'predictions_{inference_name}.json')
    with open(file_path, 'w') as file:
        json.dump(predictions, file)

    print(f"Predictions exported to {file_path}")
    print(f"Log file created at {log_file_path}")
