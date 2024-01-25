import json
import torch
import logging
import os
from tqdm import tqdm

from torchmetrics import F1Score
from evaluation import bleu_score, cider_score, rouge_score

def validation(model, dataset, tokenizer, validation_dir, validation_name, device):
    model.eval()  # Set the model to evaluation mode
    predictions = {}
    references = {}

    # Configure logging
    log_file_path = os.path.join(validation_dir, f'{validation_name}.log')
    logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(message)s')

    logger = logging.getLogger()

    # Evaluation Metrics
    f1 = F1Score(task="multiclass", num_classes=250027, top_k = 1, ignore_index = -100).to(device)
    sum_bleu_score = 0
    sum_f1_score = 0
    sum_cider_score = 0
    sum_non_text_bleu = 0
    sum_text_bleu = 0
    count_text = 0
    count_non_text = 0

    # Using tqdm for progress bar
    for data in tqdm(dataset, desc="Processing", unit="sample"):
        id, image, question, answer, image_id = data['id'], data['image'], data['question'], data['answer'], data['image_id']

        # Tokenize the question
        question_tok = tokenizer(question, max_length=60, padding='max_length', return_tensors="pt", return_attention_mask=True)
        input_ids = question_tok["input_ids"].to(device)
        attention_mask = question_tok["attention_mask"].to(device)
        
        # Tokenize the answer
        answer_tok = tokenizer(answer, max_length = 60, padding='max_length',return_tensors="pt")["input_ids"].to(device)
        answer_tok[answer_tok == 1] = -100

        # Predict
        with torch.no_grad():
            output = model(input_ids, pixel_values=image.unsqueeze(0).to(device), attention_mask=attention_mask)
        logit = output['logits']
        pred = logit.argmax(dim=2)
        pred_word = tokenizer.batch_decode(pred, skip_special_tokens=True)[0]

        # Save predictions and corressponding reference
        predictions[id] = pred_word
        references[id] = answer

        # Evaluation Score
        bleu = bleu_score(pred_word, answer)
        f1_score = f1(pred, answer_tok)

        # Cummulative Score
        sum_bleu_score += bleu
        sum_f1_score += f1_score
        if (int(image_id) < 3034):
            sum_text_bleu += bleu
            count_text += 1
            logger.info("This is a text image")
        else:
            sum_non_text_bleu += bleu
            count_non_text += 1
            logger.info("This is a non text image")

        # Log the question and its predicted answer
        logger.info(f"Question: {question}\n")
        logger.info(f"True Answer: {answer}")
        logger.info(f"Predicted Answer: {pred_word}\n")
        logger.info(f"F1: {f1_score}")
        logger.info(f"BLEU: {bleu}")
        logger.info(f"----------------------------------------------\n")
        

    print(sum_bleu_score, sum_f1_score, sum_text_bleu, sum_non_text_bleu)
    logger.info(f"Average F1: {sum_f1_score/len(dataset)}")
    logger.info(f"Average Cider: {cider_score(predictions, references)}")
    logger.info(f"Average Rouge-L: {rouge_score(predictions, references)}")
    logger.info(f"Average BLEU: {sum_bleu_score/len(dataset)}\n")
    logger.info(f"Average BLEU for text images: {sum_text_bleu/count_text}")
    logger.info(f"Average BLEU for non-text images: {sum_non_text_bleu/count_non_text}")

    # Write predictions to file
    file_path = os.path.join(validation_dir, f'predictions_{validation_name}.json')
    with open(file_path, 'w') as file:
        json.dump(predictions, file)

    print(f"Predictions exported to {file_path}")
    print(f"Log file created at {log_file_path}")