import torch
import json
import pandas as pd
from torchvision import transforms as transforms
import os
from PIL import Image

train_ocr = json.load(open('ocr/train_ocr.json'))
dev_ocr = json.load(open('ocr/dev_ocr.json'))
test_ocr = json.load(open('ocr/test_ocr.json'))
easy_ocr = json.load(open('ocr/easyocr.json'))
test_easy_ocr = json.load(open('ocr/test_easyocr.json'))

class OPENVIVQA_Dataset(torch.utils.data.Dataset):
    """
    Dataset class for the OPENVIVQA dataset.
    """
    def __init__(self, annotation_file, img_dir, ocr_type):
        with open(annotation_file, encoding='utf-8') as f:
            json_file = json.load(f)
        # Type of OCR
        self.ocr_type = ocr_type

        # Flatten the annotations into a list of dictionaries
        annotations = [{'id': annot_id, **data} for annot_id, data in json_file['annotations'].items()]
        self.annotations = pd.DataFrame(annotations)

        # Convert image_id to the correct type (if necessary)
        self.annotations['image_id'] = self.annotations['image_id'].astype(str)

        # Creating img_reference DataFrame
        img_reference = [{'image_id': img_id, 'filename': filename} for img_id, filename in json_file['images'].items()]
        self.img_reference = pd.DataFrame(img_reference)

        # Convert image_id in img_reference to the same type as in annotations
        self.img_reference['image_id'] = self.img_reference['image_id'].astype(str)

        self.img_dir = img_dir
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize the image to 224x224 pixels
            transforms.ToTensor()           # Convert the image to a tensor
        ])
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annot = self.annotations.iloc[idx]
        annot_id = annot['id']
        image_id = annot['image_id']
        question = annot['question']
        answer = annot['answer']

        # Fetching the filename
        img_file = self.img_reference[self.img_reference['image_id'] == image_id]['filename'].iloc[0]

        # Concat OCR
        ocr_text = None
        if self.ocr_type == "train":
            ocr_text = ' ' + train_ocr[image_id]
        elif self.ocr_type == "dev":
            ocr_text = ' ' + dev_ocr[image_id]
        elif self.ocr_type == "test":
            ocr_text = ' ' + test_ocr[image_id]
        question = question + ocr_text
            
        # Load the image
        img_path = os.path.join(self.img_dir, img_file)
        img = Image.open(img_path).convert('RGB')

        img = self.transform(img)

        return {
            'id': annot_id,
            'question': question,
            'answer': answer,
            'image': img,
            'image_id': image_id
        }