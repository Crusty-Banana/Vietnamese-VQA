from torchmetrics import F1Score
import nltk
from nltk.translate import bleu
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.meteor_score import meteor_score as nltk_meteor_score
from pycocoevalcap.cider.cider import Cider
from rouge import Rouge
import torch
import string

def bleu_score(predictions, references):
    smoothie = SmoothingFunction().method4
    total_scores = {1: 0, 2: 0, 3: 0, 4: 0}
    num_samples = len(predictions)

    for pred, ref in zip(predictions, references):
        # Ignore punctuations
        ref = ref.translate(str.maketrans('', '', string.punctuation))
        pred = pred.translate(str.maketrans('', '', string.punctuation))

        ref_list = [ref.split()]
        can_list = pred.split()

        # Calculate BLEU for each n-gram
        for n in range(1, 5):
            weights = tuple((1.0 / n, ) * n)  # Equal weights for each n-gram
            score = bleu(ref_list, can_list, weights=weights, smoothing_function=smoothie)
            total_scores[n] += score

    # Calculate average scores
    avg_scores = {n: total_scores[n] / num_samples for n in total_scores}

    # Calculate the overall average of the four BLEU scores
    overall_avg_score = sum(avg_scores.values()) / 4

    return avg_scores[1], avg_scores[2], avg_scores[3], avg_scores[4], overall_avg_score


def cider_score(predictions, references):
    # Initialize Cider scorer
    cider_scorer = Cider()

    # Convert lists to dictionaries
    predictions_dict = {str(idx): [pred] for idx, pred in enumerate(predictions)}
    references_dict = {str(idx): [ref] for idx, ref in enumerate(references)}

    # Compute the score
    (score, _) = cider_scorer.compute_score(references_dict, predictions_dict)

    return score

def rouge_score(predictions, references):
    # Initialize Rouge scorer
    rouge_scorer = Rouge(metrics=['rouge-l'])

    # Compute the score
    scores = rouge_scorer.get_scores(references, predictions)
    total_f1 = sum(score['rouge-l']['f'] for score in scores)
    average_f1 = total_f1 / len(scores)

    return average_f1

def meteor_score(predictions, references):
    nltk.download('punkt')
    nltk.download('wordnet')
    scores = [nltk_meteor_score([word_tokenize(ref)], word_tokenize(pred)) for pred, ref in zip(predictions, references)]
    total_score = sum(scores)
    average_score = total_score / len(scores)
    return average_score