import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import ViTModel, AutoTokenizer, MBartModel
from huggingface_hub import snapshot_download
import zipfile
import os
import datetime
import yaml
import gdown

from dataset import OPENVIVQA_Dataset, MultimodalCollator
from model import T5ForConditionalGeneration, MBartForConditionalGeneration
from action import train, inference, validation

from helper import visualize_batch, plot_img_test

def save_config(args, run_dir):
    with open(f'{run_dir}/config.yaml', 'w') as file:
        yaml.dump(vars(args), file)


def download_from_drive(file_id, output_path):
    url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(url, output_path, quiet=False)

def extract_zip_files(data_dir):
    zip_files = ['train-images.zip', 'dev-images.zip', 'test-images.zip']
    for file in zip_files:
        with zipfile.ZipFile(os.path.join(data_dir, file), "r") as zip_ref:
            zip_ref.extractall(data_dir)

def create_model(model_type, model_name, device):
    if model_type == 'mbart':
        model = MBartForConditionalGeneration.from_pretrained(model_name).to(device)
    else:
        model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
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

    # Check if data needs to be downloaded and extracted
    required_files = ['train-images.zip', 'dev-images.zip', 'test-images.zip']
    need_download = not all(os.path.exists(os.path.join(args.data_dir, file)) for file in required_files)

    if need_download:
        # Download and extract data
        snapshot_download(repo_id=args.repo_id, local_dir=args.data_dir, repo_type="dataset", local_dir_use_symlinks="auto")
        extract_zip_files(args.data_dir)

    # Check and download additional files from Google Drive if not present
    google_drive_files = {
        '1b_oRS_tE9feJO_5voS4eiIZc48jzoJqU': 'vlsp2023_dev_data_new.json',
        '1xm3Syc_RoEiXW6Rx7Q_bXpERf5pg-g5j': 'vlsp2023_train_data_new.json'
    }

    for file_id, file_name in google_drive_files.items():
        file_path = os.path.join(args.data_dir, file_name)
        if not os.path.exists(file_path):
            print(f"Downloading {file_name} from Google Drive...")
            download_from_drive(file_id, file_path)

    # Choose dataset files based on the use_cleaned_dataset argument
    train_data_file = 'vlsp2023_train_data_new.json' if args.use_cleaned_dataset else 'vlsp2023_train_data.json'
    dev_data_file = 'vlsp2023_dev_data_new.json' if args.use_cleaned_dataset else 'vlsp2023_dev_data.json'
    test_data_file = 'vlsp2023_test_data.json'
    # Tokenizer
    # tokenizer = AutoTokenizer.from_pretrained("vinai/bartpho-word")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    if args.action == 'train':
        run_dir = f"{args.train_dir}/{current_time}"
        os.makedirs(run_dir, exist_ok=True)
        save_config(args, run_dir)

        # Data Loaders        
        train_dataset = OPENVIVQA_Dataset(os.path.join(args.data_dir, train_data_file), os.path.join(args.data_dir, 'training-images'), 'google/vit-base-patch16-224-in21k', args.use_ocr)
        val_dataset = OPENVIVQA_Dataset(os.path.join(args.data_dir, dev_data_file), os.path.join(args.data_dir, 'dev-images'), 'google/vit-base-patch16-224-in21k', args.use_ocr)

        collate_fn = MultimodalCollator(tokenizer)

        train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn)
        val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)

        # Model
        model = create_model(args.model_type, args.model_name, device)

        # Training
        criterion = nn.CrossEntropyLoss(ignore_index=-100)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

        if args.checkpoint_path:
            load_model_from_checkpoint(model, args.checkpoint_path, optimizer=optimizer)

        model, train_loss, val_loss, train_f1, val_f1 = train(model, train_loader, val_loader, optimizer, criterion, args.num_epochs, lr_scheduler, device, run_dir, current_time)

        # Visualization model
        # answers, answer_tokens, predictions, pred_tokens, f1_tm_lst, bleu_lst = plot_img_test(3, train_loader.dataset, model, device, tokenizer, run_dir)

        # answers, answer_tokens, predictions, pred_tokens, f1_tm_lst, bleu_lst = plot_img_test(3, val_loader.dataset, model, device, tokenizer, run_dir)

    elif args.action == 'inference':

        if not args.checkpoint_path:
            raise ValueError("Checkpoint path must be provided for inference action.")            

        os.makedirs(args.inference_dir, exist_ok=True)

        test_dataset = OPENVIVQA_Dataset(os.path.join(args.data_dir, test_data_file), os.path.join(args.data_dir, 'test-images'), 'google/vit-base-patch16-224-in21k', args.use_ocr)
        
        model = create_model(args.model_type, args.model_name, device)
        load_model_from_checkpoint(model, args.checkpoint_path)
        
        inference(model, test_dataset, tokenizer, args.inference_dir, current_time, device)

    elif args.action == 'validation':

        if not args.checkpoint_path:
            raise ValueError("Checkpoint path must be provided for validation action.")            

        os.makedirs(args.validation_dir, exist_ok=True)

        val_dataset = OPENVIVQA_Dataset(os.path.join(args.data_dir, dev_data_file), os.path.join(args.data_dir, 'dev-images'), 'google/vit-base-patch16-224-in21k', args.use_ocr)
        
        model = create_model(args.model_type, args.model_name, device)
        load_model_from_checkpoint(model, args.checkpoint_path)
        
        validation(model, val_dataset, tokenizer, args.validation_dir, current_time, device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VQA Model Training and Evaluation")

    parser.add_argument('--action', type=str, default='train', choices=['train', 'inference', 'validation'], help='Action to perform: train or inference or validation')
    parser.add_argument('--repo_id', type=str, default='uitnlp/OpenViVQA-dataset', help='Huggingface repo ID for dataset')
    parser.add_argument('--data_dir', type=str, default='data', help='Local directory for dataset')
    parser.add_argument('--train_dir', type=str, default='run/train', help='Directory to save training log and model')
    parser.add_argument('--inference_dir', type=str, default='run/inference', help='Directory to save inference name')
    parser.add_argument('--validation_dir', type=str, default='run/validation', help='Directory to save validation name')

    parser.add_argument('--checkpoint_path', type=str, default=None, help='Path to the model checkpoint')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for data loaders')
    parser.add_argument('--num_workers', type=int, default=12, help='Number of workers for data loaders')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--step_size', type=int, default=5, help='Step size for learning rate scheduler')
    parser.add_argument('--gamma', type=float, default=0.5, help='Gamma for learning rate scheduler')
    parser.add_argument('--num_epochs', type=int, default=12, help='Number of epochs for training')
    
    parser.add_argument('--model_type', type=str, default='mt5', choices=['mbart', 'mt5'], help='Type of model to use: mbart or mt5')
    parser.add_argument('--model_name', type=str, default='VietAI/vit5-base', help='Model name for the transformer model')

    parser.add_argument('--use_cleaned_dataset', action='store_true', help='Use the cleaned-up dataset')

    parser.add_argument('--use_ocr', type=bool, default=False, help='Concat ocr to the end of question')

    args = parser.parse_args()
    main(args)
