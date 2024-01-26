import json
import torch
import logging
import os
from tqdm import tqdm
from metrics import bleu_score, cider_score, rouge_score, meteor_score

def validation(model, dataset, tokenizer, validation_dir, validation_name, device, batch_size=64, num_beams=1):
    model.eval()  # Set the model to evaluation mode

    # Logging setup
    log_file_path = os.path.join(validation_dir, f'{validation_name}.log')
    logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(message)s')
    logger = logging.getLogger()

    # Accumulators for predictions and references
    all_predictions = []
    all_references = []
    text_predictions = []
    text_references = []
    non_text_predictions = []
    non_text_references = []

    for batch_start in tqdm(range(0, len(dataset), batch_size), desc="Processing", unit="batch"):
        input_ids, attention_masks, img_ws = [], [], []
        batch_ids, image_ids, questions, answers = [], [], [], []

        # Batch data accumulation
        for i in range(batch_start, min(batch_start + batch_size, len(dataset))):
            data = dataset[i]
            batch_ids.append(data['id'])
            img_ws.append(data['img_w'].unsqueeze(0).to(device))
            image_ids.append(data['image_id'])
            questions.append(data['question'])
            answers.append(data['answer'])
            input_ids.append(tokenizer(data['question'], max_length=60, truncation=True, padding='max_length', return_tensors="pt")["input_ids"].to(device))
            attention_masks.append(tokenizer(data['question'], max_length=60, truncation=True, padding='max_length', return_tensors="pt")["attention_mask"].to(device))

        # Convert lists to tensors
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        img_ws = torch.cat(img_ws, dim=0)

        # Model prediction
        with torch.no_grad():
            preds = model.generate(
                img_w=img_ws,
                input_ids=input_ids,
                attention_mask=attention_masks,
                num_beams=num_beams,  # Set num_beams to 1 for greedy decoding
                max_length=60,
                return_dict_in_generate=True,
                output_attentions=True,
                no_repeat_ngram_size=2
            ).sequences

        
        # Decode predictions
        pred_words = [tokenizer.decode(pred, skip_special_tokens=True) for pred in preds]

        # Accumulate predictions and references
        for i, pred_word in enumerate(pred_words):
            all_predictions.append(pred_word)
            all_references.append(answers[i])

            is_text = int(image_ids[i]) < 3034
            if is_text:
                text_predictions.append(pred_word)
                text_references.append(answers[i])
            else:
                non_text_predictions.append(pred_word)
                non_text_references.append(answers[i])

            # Log the question and its predicted answer
            logger.info(f"Question: {questions[i]}\n")
            logger.info(f"True Answer: {answers[i]}")
            logger.info(f"Predicted Answer: {pred_word}\n")
            logger.info(f"----------------------------------------------\n")
            

    # Calculate BLEU, CIDEr, ROUGE scores for all, text, and non-text
    scores = {
        "all": {
            "BLEU": bleu_score(all_predictions, all_references),
            "CIDEr": cider_score(all_predictions, all_references),
            "ROUGE-L": rouge_score(all_predictions, all_references),
            "METEOR": meteor_score(all_predictions, all_references)
        },
        "text": {
            "BLEU": bleu_score(text_predictions, text_references),
            "CIDEr": cider_score(text_predictions, text_references),
            "ROUGE-L": rouge_score(text_predictions, text_references),
            "METEOR": meteor_score(text_predictions, text_references)
        },
        "non_text": {
            "BLEU": bleu_score(non_text_predictions, non_text_references),
            "CIDEr": cider_score(non_text_predictions, non_text_references),
            "ROUGE-L": rouge_score(non_text_predictions, non_text_references),
            "METEOR": meteor_score(non_text_predictions, non_text_references)
        }
    }

    # Logging scores
    for category, category_scores in scores.items():
        logger.info(f"Scores for {category}:")
        for metric, value in category_scores.items():
            logger.info(f"{metric}: {value}")

    # Save predictions
    # file_path = os.path.join(validation_dir, f'predictions_{validation_name}.json')
    # with open(file_path, 'w') as file:
    #     json.dump({"all": all_predictions, "text": text_predictions, "non_text": non_text_predictions}, file)

    # print(f"Predictions and scores exported to {file_path}")
    print(f"Log file created at {log_file_path}")

# Call this function with appropriate parameters
