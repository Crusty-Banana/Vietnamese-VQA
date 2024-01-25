from PIL import Image
import os
import torch
from transformers import AutoModel, AutoImageProcessor
import tqdm

def encode_images(img_dir, image_encode_model_name, device=None, tensor_path=None):
    # Set device and tensor_path defaults if not provided
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if tensor_path is None:
        tensor_path = 'img_encoding.pt'

    tensor_full_path = os.path.join(img_dir, tensor_path)

    # Check if the tensor file already exists
    if os.path.exists(tensor_full_path):
        # Load the tensor file
        img_w = torch.load(tensor_full_path, map_location=device)

        # Check if the tensors are on GPU and move them to CPU
        for img_name, encoded_image in img_w.items():
            if encoded_image.is_cuda:
                encoded_image = encoded_image.to('cpu')
            
            img_w[img_name] = encoded_image.squeeze()

        print("Loaded encoded images from existing tensor file.")
        return img_w
    else:
        print(f"Processing images and creating a new tensor file in {img_dir}.")

    # Load the model and preprocessor
    image_encoder = AutoModel.from_pretrained(image_encode_model_name).to(device)
    preprocessor = AutoImageProcessor.from_pretrained(image_encode_model_name)

    img_w = {}

    # Process each image in the directory
    for img_name in tqdm.tqdm(os.listdir(img_dir)):
        img_path = os.path.join(img_dir, img_name)
        if os.path.isfile(img_path):
            try:
                # Load and preprocess the image
                processed_images = preprocessor(images=[Image.open(img_path).convert('RGB')], return_tensors="pt")['pixel_values'].to(device)
                
                # Encode the image
                with torch.no_grad():
                    encoded_image = image_encoder(pixel_values=processed_images, return_dict=True).last_hidden_state

                # Move the encoded image to CPU
                encoded_image_cpu = encoded_image.squeeze().to('cpu')

                img_w[img_name] = encoded_image_cpu

            except Exception as e:
                print(f"Error processing image {img_name}: {e}")

    # Save the encoded images to the specified path (on CPU)
    torch.save(img_w, tensor_full_path)

    return img_w
