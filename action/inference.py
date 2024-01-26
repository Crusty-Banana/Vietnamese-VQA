import json
import torch
import logging
import os
from tqdm import tqdm

def inference(model, dataset, tokenizer, inference_dir, inference_name, device, batch_size=128):
    model.eval()  # Set the model to evaluation mode
    predictions = {}

    # Configure logging
    log_file_path = os.path.join(inference_dir, f'{inference_name}.log')
    logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(message)s')
    logger = logging.getLogger()

    # Process dataset in batches
    for batch_start in tqdm(range(0, len(dataset), batch_size), desc="Processing", unit="batch"):
        # Initialize batch variables
        input_ids = []
        attention_masks = []
        img_ws = []
        ids = []
        questions = []

        # Accumulate batch data by iterating
        for i in range(batch_start, min(batch_start + batch_size, len(dataset))):
            data = dataset[i]
            ids.append(data['id'])
            img_ws.append(data['img_w'].unsqueeze(0).to(device))
            questions.append(data['question'])

            question_tok = tokenizer(data['question'], max_length=60, truncation=True, padding='max_length', return_tensors="pt", return_attention_mask=True)
            input_ids.append(question_tok["input_ids"].to(device))
            attention_masks.append(question_tok["attention_mask"].to(device))

        # Convert lists to tensors
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        img_ws = torch.cat(img_ws, dim=0)

        # Predict
        with torch.no_grad():
            outputs = model.generate(
                img_w=img_ws,
                input_ids=input_ids,
                attention_mask=attention_masks,
                max_length=60,
                return_dict_in_generate=True, 
                output_attentions=True,
                num_beams=8,
                no_repeat_ngram_size=2)

        with tokenizer.as_target_tokenizer():
            outputs = [tokenizer.decode(out, clean_up_tokenization_spaces=False, skip_special_tokens=True) for out in outputs.sequences]

        # Process and log each item in the batch
        for i, id in enumerate(ids):
            pred_word = outputs[i]
            predictions[id] = pred_word
            logger.info(f"Question: {questions[i]}")
            logger.info(f"Predicted Answer: {pred_word}\n")

    # Write predictions to file
    file_path = os.path.join(inference_dir, f'predictions_{inference_name}.json')
    with open(file_path, 'w') as file:
        json.dump(predictions, file)

    print(f"Predictions exported to {file_path}")
    print(f"Log file created at {log_file_path}")
