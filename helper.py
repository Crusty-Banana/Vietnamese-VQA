import matplotlib.pyplot as plt
import torch
import numpy as np
from torchmetrics import F1Score
import os 

from evaluation import bleu_score

# Function to visualize a batch of data
def visualize_batch(batch, save_dir):
    plt.figure(figsize=(12, 4))
    for i in range(len(batch['image'])):
        fig = plt.figure(figsize=(12, 4))
        img = batch['image'][i].permute(1, 2, 0)
        plt.imshow(img)
        plt.title(f"Q: {batch['question'][i]}\nA: {batch['answer'][i]}")
        plt.axis('off')
        # Save each plot as an image file
        fig.savefig(os.path.join(save_dir, f"batch_image_{i}.png"))
        plt.close(fig)


def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int):
    """
    Shift input ids one token to the right, and wrap the last non pad token (the <LID> token). Note that MBart does not
    have a single `decoder_start_token_id` in contrast to other Bart-like models.
    """
    # Ensure input_ids is 2D (batch_size, sequence_length)
    prev_output_tokens = input_ids.clone()

    if pad_token_id is None:
        raise ValueError("pad_token_id has to be defined.")

    # Replace possible -100 values in labels by `pad_token_id`
    prev_output_tokens.masked_fill_(prev_output_tokens == -100, pad_token_id)

    # Calculate index_of_eos with ensuring 2D shape
    index_of_eos = (prev_output_tokens.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
    decoder_start_tokens = prev_output_tokens.gather(1, index_of_eos).squeeze()
    prev_output_tokens[:, 1:] = prev_output_tokens[:, :-1].clone()
    prev_output_tokens[:, 0] = decoder_start_tokens

    return prev_output_tokens

def plot_img_test(no_, dataset, model, device, tokenizer, save_dir):
    images, questions, answers, answer_tokens, predictions, pred_tokens, f1_tm_lst, bleu_lst = [],[],[],[],[],[],[],[]
    random_idx = np.random.choice(len(dataset),no_)
    for i in random_idx:
        data = dataset.__getitem__(i)
        image, question, answer = data['image'], data['question'], data['answer']
        images.append(image.permute(1, 2, 0))
        questions.append(question)
        answers.append(answer)

        question_tok = tokenizer(question, max_length = 60, padding='max_length',return_tensors="pt", return_attention_mask = True)
        input_ids = question_tok["input_ids"].to(device)
        attention_mask = question_tok["attention_mask"].to(device)
        answer_tok = tokenizer(answer, max_length = 60, padding='max_length',return_tensors="pt")["input_ids"].to(device)
        answer_tokens.append(answer_tok)
        answer_tok[answer_tok == 1] = -100

        output = model(input_ids, pixel_values = image.unsqueeze(0).to(device), attention_mask = attention_mask)
        logit = output['logits']
        pred = logit.argmax(dim =2)
        pred_tokens.append(pred)
        pred_word = tokenizer.batch_decode(pred, skip_special_tokens=True)[0]
        predictions.append(pred_word)

        pred[pred == 1] = -100
        f1 = F1Score(task="multiclass", num_classes=250027, top_k = 1, ignore_index = -100).to(device)
        s3 = f1(pred, answer_tok)
        f1_tm_lst.append(s3)
        s2 = bleu_score(answer,pred_word)
        bleu_lst.append(s2)

        print('\n Q: ' + question + '\n A: '+ answer + '\n Pred: '+ pred_word)

    print("\n F1 torchmetric",f1_tm_lst)
    print("\n Bleu_score",bleu_lst)

    for idx in range(no_):
        image = images[idx]
        fig = plt.figure(figsize=(10, 6))
        plt.imshow(image)
        plt.title('Q: ' + questions[idx] + '\n A: '+ answers[idx] + '\n Pred: '+ predictions[idx])
        plt.axis('off')
        # Save each plot as an image file
        fig.savefig(os.path.join(save_dir, f"test_image_{idx}.png"))
        plt.close(fig)

    return answers, answer_tokens, predictions, pred_tokens, f1_tm_lst, bleu_lst