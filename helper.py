import matplotlib.pyplot as plt
import torch

# Function to visualize a batch of data
def visualize_batch(batch):
    plt.figure(figsize=(12, 4))
    for i in range(len(batch['image'])):
        ax = plt.subplot(1, len(batch['image']), i + 1)
        img = batch['image'][i].permute(1, 2, 0)  # Convert from (C, H, W) to (H, W, C)
        plt.imshow(img)
        plt.title(f"Q: {batch['question'][i]}\nA: {batch['answer'][i]}")
        plt.axis('off')
    plt.show()

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

