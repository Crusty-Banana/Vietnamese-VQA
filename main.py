import torch
import gc
from dataset import OPENVIVQA_Dataset
from helper import visualize_batch, plot_img_test
from transformers import pipeline, ViTImageProcessor, ViTForImageClassification, ViTModel, AutoTokenizer, MBartModel,MBartForConditionalGeneration
from transformers import T5Tokenizer, T5Model
from model import VQAModel
from collator import MultimodalCollator
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import torch.nn as nn
from train import train
from evaluation import test_eval
from pytorch_model_summary import summary
import zipfile
from huggingface_hub import snapshot_download

snapshot_download(repo_id='uitnlp/OpenViVQA-dataset', repo_type="dataset",
                    local_dir="data",
                    local_dir_use_symlinks="auto"
                )
with zipfile.ZipFile("data/train-images.zip","r") as zip_ref:
    zip_ref.extractall("data")
with zipfile.ZipFile("data/dev-images.zip","r") as zip_ref:
    zip_ref.extractall("data")
with zipfile.ZipFile("data/test-images.zip","r") as zip_ref:
    zip_ref.extractall("data")

dataset = OPENVIVQA_Dataset('data/vlsp2023_dev_data.json', 'data/dev-images')

# Create a DataLoader
data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

# Iterate and visualize the first few batches
num_batches_to_visualize = 3
for i, batch in enumerate(data_loader):
    if i >= num_batches_to_visualize:
        break
    visualize_batch(batch)

# Get GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using ', device)

# Get models
image_encoder = ViTModel.from_pretrained('google/vit-base-patch16-224').to(device)
tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-cc25")
text_model = MBartModel.from_pretrained("facebook/mbart-large-cc25").to(device)

model = VQAModel(image_encoder, text_model, device).to(device)

print(summary(model, torch.zeros((2,60)).long().to(device),
              torch.zeros((2,3,224,224)).to(device),
              torch.zeros((2,60)).long().to(device),
              torch.zeros((2,60)).long().to(device),
              show_input=True))

# Get Collator
collate_fn = MultimodalCollator(tokenizer)

# Get Dataset
train_dataset = OPENVIVQA_Dataset('data/vlsp2023_train_data.json', 'data/training-images')
val_dataset = OPENVIVQA_Dataset('data/vlsp2023_dev_data.json', 'data/dev-images')
test_dataset = OPENVIVQA_Dataset('data/vlsp2023_test_data.json', 'data/test-images')

# Get Data Loader
train_loader = DataLoader(dataset=train_dataset, batch_size=4, shuffle = True, num_workers = 4, collate_fn = collate_fn)
val_loader = DataLoader(dataset=val_dataset, batch_size=4, shuffle = False, num_workers = 4, collate_fn = collate_fn)
test_loader = DataLoader(dataset=test_dataset, batch_size=4, shuffle = False, num_workers = 4, collate_fn = collate_fn)

# Setting up training parameters
num_epochs = 12
learning_rate = 1e-5
scale = 1
# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss(ignore_index = -100)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=5,
                                               gamma=0.5)


# #load old model
# checkpoint_path = '/home/huynh/Vietnamese-VQA/run_2024-01-04_17-09-53/model_2024-01-04_17-09-53.pth'

# # Load the checkpoint
# checkpoint = torch.load(checkpoint_path)

# # Load the model state dictionary
# model.load_state_dict(checkpoint)

# # Load the optimizer state dictionary if available
# if 'optimizer_state_dict' in checkpoint:
#     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# del checkpoint
# torch.cuda.empty_cache()

# Train model 

model, train_loss, val_loss, train_f1, val_f1 = train(model, train_loader, val_loader, optimizer, criterion, num_epochs, lr_scheduler, device)

# Evaluate model
f1_torchmetric, bleu_l, pred_token_l, pred_word_l = test_eval(val_dataset,model, device, tokenizer)

print("average f1 score: ", sum(f1_torchmetric)/len(f1_torchmetric))
print("average bleu score: ", sum(bleu_l)/len(bleu_l))

# Visualized model
answers, answer_tokens, predictions, pred_tokens, f1_tm_lst, bleu_lst = plot_img_test(3,val_dataset, model, device, tokenizer)

answers, answer_tokens, predictions, pred_tokens, f1_tm_lst, bleu_lst = plot_img_test(3,train_dataset, model, device, tokenizer)