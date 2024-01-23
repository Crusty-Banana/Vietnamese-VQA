from torchmetrics import F1Score
from nltk.translate import bleu
from nltk.translate.bleu_score import SmoothingFunction
import torch
import string

def bleu_score(reference, candidate):
  # ignore punctuations
    reference = reference.translate(str.maketrans('', '', string.punctuation))
    candidate = candidate.translate(str.maketrans('', '', string.punctuation))

    ref_list = reference.split()
    can_list = candidate.split()

    smoothie = SmoothingFunction().method4

    score = bleu([ref_list], can_list,smoothing_function=smoothie, weights = [0.25,0.25,0.25,0.25])
    return score

def test_eval(test_data, model, device, tokenizer):
    f1_torchmetric, bleu_l, pred_token_l, pred_word_l = [],[],[],[]

    model.eval()
    with torch.no_grad():
        for i in range(len(test_data)):
            data = test_data.__getitem__(i)
            image, question, answer = data['image'], data['question'], data['answer']
            image = image.to(device)

            question_tok = tokenizer(question, max_length = 60, padding='max_length',return_tensors="pt", return_attention_mask = True)
            input_ids = question_tok["input_ids"].to(device)
            attention_mask = question_tok["attention_mask"].to(device)
            answer_tok = tokenizer(answer, max_length = 60, padding='max_length',return_tensors="pt")['input_ids']
            answer_tok[answer_tok == 1] = -1
            answer_tok = answer_tok.to(device)
            output = model(input_ids = input_ids, pixel_values = image.unsqueeze(0), attention_mask = attention_mask)

            logit = output['logits']
            pred = logit.argmax(dim =2)
            pred_token_l.append(pred.clone().cpu())
            pred_word = tokenizer.batch_decode(pred, skip_special_tokens=True)[0]
            pred_word_l.append(pred_word)

            # F1 score from torch.metric
            f1 = F1Score(task="multiclass", num_classes=250027, top_k = 1, ignore_index = -1).to(device)
            s3 = f1(pred, answer_tok)
            f1_torchmetric.append(s3)

            # Compute bleu score
            s2 = bleu_score(answer,pred_word)
            bleu_l.append(s2)
    return f1_torchmetric, bleu_l, pred_token_l, pred_word_l