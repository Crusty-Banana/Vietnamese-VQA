import torch.nn as nn
import torch
from typing import Optional, Tuple, Union
from helper import shift_tokens_right

# Build the mVQA model. Some of the codes are modified from the source code of mBART at hugging face.
class VQAModel(nn.Module):
    """
    The VQAModel should consist of a image encoder and a multilingual language transformer (encoder, decoder, encoder-decoder).

    The visual_text_block is a module that integrates the encodings from the images and text.
    """

    def __init__(self, image_encoder, text_model, device, vocab_size = 250027, img_hidden_dim = 768, txt_hidden_dim = 1024):
        super(VQAModel, self).__init__()
        self.image_encoder = image_encoder
        self.config = text_model.config
        self.encoder = text_model.encoder
        self.decoder = text_model.decoder
        self.vocab_size = vocab_size
        self.device = device
        # Add a linear layer to change dim of image encoding if img_hidden_dim and txt_hidden_dim do not match.
        if img_hidden_dim != txt_hidden_dim:
            self.needConvert = True
            self.dim_change = nn.Linear(img_hidden_dim, txt_hidden_dim)
        # lm_head convert text in hidden state after decoder to logits
        self.lm_head = nn.Linear(txt_hidden_dim, out_features= vocab_size, bias=False) #250027 is the vocab size for mBART
        # Cross entropy loss, ignore padding index.
        self.loss_fn = nn.CrossEntropyLoss(ignore_index = -1)
        # Logit bias of mBART
        self.register_buffer("final_logits_bias", torch.zeros((1, vocab_size)))


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
     ):
        """
          question_input_ids - tokenized input question
          pixel_values - preprocessed image
          labels - resulting tokens from input
        """

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states)

        if input_ids is not None and input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)

        if attention_mask is not None and attention_mask.ndim == 1:
            attention_mask = attention_mask.unsqueeze(0)

        # Different to other models, MBart automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = shift_tokens_right(input_ids, self.config.pad_token_id)


        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Freeze question encoder and vision encoder
        with torch.no_grad():
            # Question encoding
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            # Image encoding
            image_encoder_outputs = self.image_encoder(pixel_values)

        if self.needConvert:
            image_state = self.dim_change(image_encoder_outputs['last_hidden_state'])
        else:
            image_state = image_encoder_outputs['last_hidden_state']

        # Concatneante image encoding and text encoding, built relative attention mask.
        device = self.device
        img_mask = torch.ones((image_state.shape[0], image_state.shape[1])).to(device)
        hidden_states = torch.cat([image_state, encoder_outputs[0]], dim = 1).to(device)
        attention_mask = torch.cat([img_mask, attention_mask], dim = 1).to(device)

        # Decode
        decoder_outputs = self.decoder(
                input_ids=decoder_input_ids,
                attention_mask=decoder_attention_mask,
                encoder_hidden_states = hidden_states,
                encoder_attention_mask = attention_mask,
                head_mask=decoder_head_mask,
                cross_attn_head_mask=cross_attn_head_mask,
                past_key_values=past_key_values,
                inputs_embeds=decoder_inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        # The logits of generated answer, in the shape of BatchNumber * hidden_state_size * vocab_size
        lm_logits = self.lm_head(decoder_outputs[0]) + self.final_logits_bias
        out = {
                "logits": lm_logits,
                "hidden_states": decoder_outputs['last_hidden_state']
            }

        # Calculate cross entropy loss (need to ignore paddings).
        if labels is not None:
            labels = labels.to(lm_logits.device)
            loss = self.loss_fn(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            out["loss"] = loss

        return out