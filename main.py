import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import ViTModel, AutoTokenizer, MBartModel
from huggingface_hub import snapshot_download
import zipfile
import os
import datetime

from dataset import OPENVIVQA_Dataset
from model import VQAModel
from collator import MultimodalCollator
from train import train
from inference import inference
from evaluation import test_eval
from helper import visualize_batch, plot_img_test

def extract_zip_files(data_dir):
    zip_files = ['train-images.zip', 'dev-images.zip', 'test-images.zip']
    for file in zip_files:
        with zipfile.ZipFile(os.path.join(data_dir, file), "r") as zip_ref:
            zip_ref.extractall(data_dir)

def create_model(device):
    image_encoder = ViTModel.from_pretrained('google/vit-base-patch16-224').to(device)
    text_model = MBartModel.from_pretrained("facebook/mbart-large-cc25").to(device)
    model = VQAModel(image_encoder, text_model, device).to(device)
    return model

def load_model_from_checkpoint(model, checkpoint_path, optimizer=None):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint)

    if 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    del checkpoint
    torch.cuda.empty_cache()

    print(f"Model loaded from checkpoint: {checkpoint_path}")

def main(args):
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using ', device)

    # Tokenizer (created only once)
    tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-cc25")

    # Check if data needs to be downloaded and extracted
    required_files = ['train-images.zip', 'dev-images.zip', 'test-images.zip']
    need_download = not all(os.path.exists(os.path.join(args.data_dir, file)) for file in required_files)

    if need_download:
        # Download and extract data
        snapshot_download(repo_id=args.repo_id, local_dir=args.data_dir, repo_type="dataset", local_dir_use_symlinks="auto")
        extract_zip_files(args.data_dir)

    if args.action == 'train':
        run_dir = f"{args.save_dir}/{current_time}"
        os.makedirs(run_dir, exist_ok=True)

        # Data Loaders        
        train_dataset = OPENVIVQA_Dataset(os.path.join(args.data_dir, 'vlsp2023_train_data.json'), os.path.join(args.data_dir, 'training-images'))
        val_dataset = OPENVIVQA_Dataset(os.path.join(args.data_dir, 'vlsp2023_dev_data.json'), os.path.join(args.data_dir, 'dev-images'))

        collate_fn = MultimodalCollator(tokenizer)

        train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn)
        val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)

        # Model
        model = create_model(device)

        # Training
        criterion = nn.CrossEntropyLoss(ignore_index=-1)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

        if args.checkpoint_path:
            load_model_from_checkpoint(model, args.checkpoint_path, optimizer=optimizer)

        model, train_loss, val_loss, train_f1, val_f1 = train(model, train_loader, val_loader, optimizer, criterion, args.num_epochs, lr_scheduler, device, run_dir, current_time)

        # Evaluation
        f1_torchmetric, bleu_l, pred_token_l, pred_word_l = test_eval(val_loader.dataset, model, device, tokenizer)
        print("average f1 score: ", sum(f1_torchmetric) / len(f1_torchmetric))
        print("average bleu score: ", sum(bleu_l) / len(bleu_l))

        # Visualization model
        answers, answer_tokens, predictions, pred_tokens, f1_tm_lst, bleu_lst = plot_img_test(3, train_loader.dataset, model, device, tokenizer, run_dir)

        answers, answer_tokens, predictions, pred_tokens, f1_tm_lst, bleu_lst = plot_img_test(3, val_loader.dataset, model, device, tokenizer, run_dir)

    elif args.action == 'inference':

        if not args.checkpoint_path:
            raise ValueError("Checkpoint path must be provided for inference action.")            

        os.makedirs(args.inference_dir, exist_ok=True)

        test_dataset = OPENVIVQA_Dataset(os.path.join(args.data_dir, 'vlsp2023_test_data.json'), os.path.join(args.data_dir, 'test-images'))
        
        model = create_model(device)
        load_model_from_checkpoint(model, args.checkpoint_path)
        
        inference(model, test_dataset, tokenizer, args.inference_dir, current_time, device)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VQA Model Training and Evaluation")

    parser.add_argument('--action', type=str, default='train', choices=['train', 'inference'], help='Action to perform: train or inference')
    parser.add_argument('--repo_id', type=str, default='uitnlp/OpenViVQA-dataset', help='Huggingface repo ID for dataset')
    parser.add_argument('--data_dir', type=str, default='data', help='Local directory for dataset')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for data loaders')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loaders')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--step_size', type=int, default=5, help='Step size for learning rate scheduler')
    parser.add_argument('--gamma', type=float, default=0.5, help='Gamma for learning rate scheduler')
    parser.add_argument('--num_epochs', type=int, default=12, help='Number of epochs for training')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Path to the model checkpoint')
    parser.add_argument('--save_dir', type=str, default='run', help='Directory to save runs and images')
    parser.add_argument('--inference_dir', type=str, default='inference', help='Directory to save inference name')

    args = parser.parse_args()
    main(args)
