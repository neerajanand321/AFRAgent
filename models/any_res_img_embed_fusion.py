import torch
from transformers import InstructBlipForConditionalGeneration, InstructBlipVisionModel,  InstructBlipVisionConfig
from transformers.utils import ModelOutput
from transformers.modeling_outputs import BaseModelOutputWithPooling
from torch.nn import CrossEntropyLoss
import torch.nn as nn
from typing import Optional, Tuple, Any, Union
from dataclasses import dataclass

@dataclass
# Copied from transformers.models.blip_2.modeling_blip_2.Blip2ForConditionalGenerationModelOutput with Blip2->InstructBlip
class InstructBlipForConditionalGenerationModelOutput(ModelOutput):
    """
    Class defining the outputs of [`InstructBlipForConditionalGeneration`].

    Args:
        loss (`torch.FloatTensor`, *optional*, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`):
            Language modeling loss from the language model.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head of the language model.
        vision_outputs (`BaseModelOutputWithPooling`):
            Outputs of the vision encoder.
        qformer_outputs (`BaseModelOutputWithPoolingAndCrossAttentions`):
            Outputs of the Q-Former (Querying Transformer).
        language_model_outputs (`CausalLMOutputWithPast` or `Seq2SeqLMOutput`):
            Outputs of the language model.
    """

    loss: Optional[Tuple[torch.FloatTensor]] = None
    logits: Optional[Tuple[torch.FloatTensor]] = None
    vision_outputs: Optional[torch.FloatTensor] = None
    qformer_outputs: Optional[Tuple[torch.FloatTensor]] = None
    language_model_outputs: Optional[Tuple[torch.FloatTensor]] = None

    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            self[k]
            if k not in ["vision_outputs", "qformer_outputs", "language_model_outputs"]
            else getattr(self, k).to_tuple()
            for k in self.keys()
        )

class ImagesFusionInstructBlipForConditionalGeneration(InstructBlipForConditionalGeneration):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.high_res_query_tokens =  nn.Parameter(torch.zeros(1, self.config.num_query_tokens*4, self.config.qformer_config.hidden_size))
        self.class_embedding_top = self.vision_model.embeddings.class_embedding
        self.class_embedding_bottom = self.vision_model.embeddings.class_embedding

    def handle_low_res_image(
        self, 
        pixel_values,
        qformer_input_ids,
        qformer_attention_mask,
        output_attentions,
        output_hidden_states,
        return_dict,
        interpolate_pos_encoding,
        use_generate = False
    ):
        # step 1: get image embeddings for low resolution image
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )
        if use_generate:
            image_embeds = vision_outputs.last_hidden_state
        else:
            image_embeds = vision_outputs[0]
        
        # step 2: forward the query tokens through the QFormer, using the image embeddings for cross-attention
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)

        # difference with BLIP-2 here: we also feed the instruction prompt to the Q-Former
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_attention_mask = torch.ones(query_tokens.size()[:-1], dtype=torch.long, device=image_embeds.device)
        if qformer_attention_mask is None:
            qformer_attention_mask = torch.ones_like(qformer_input_ids)
        qformer_attention_mask = torch.cat([query_attention_mask, qformer_attention_mask], dim=1)
        query_outputs = self.qformer(
            input_ids=qformer_input_ids,
            attention_mask=qformer_attention_mask,
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if use_generate:
            query_output = query_outputs.last_hidden_state[:, : query_tokens.size(1), :]
        else:
            query_output = query_outputs[0][:, : query_tokens.size(1), :]

        # step 3: use the language model, conditioned on the query outputs and the prompt
        language_model_inputs = self.language_projection(query_output)
        language_model_attention_mask = torch.ones(
            language_model_inputs.size()[:-1], dtype=torch.long, device=language_model_inputs.device
        )

        return language_model_inputs, language_model_attention_mask

    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """
        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher
        resolution images.
    
        Source:
        https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174
        """
    
        class_pos_embed = embeddings[:, 0, :]
        patch_pos_embed = embeddings[:, 1:, :]
        dim = embeddings.shape[-1]
        patch_size = self.vision_model.embeddings.patch_size
        h0 = height // patch_size
        w0 = width // patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        h0, w0 = h0 + 0.1, w0 + 0.1
        patch_pos_embed = patch_pos_embed.reshape(1, 8, 16, dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed,
            scale_factor=(h0 / 8, w0 / 16),
            mode="bicubic",
            align_corners=False,
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1) 

    def get_high_res_crop_vision_output(
        self,
        pixel_values,
        output_attentions,
        output_hidden_states,
        return_dict,
        high_res_crop_location
    ):
        batch_size, _, height, width = pixel_values.shape
        target_dtype = self.vision_model.embeddings.patch_embedding.weight.dtype
        patch_embeds = self.vision_model.embeddings.patch_embedding(pixel_values.to(dtype=target_dtype))  # shape = [*, width, grid, grid]
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)
        if high_res_crop_location == "top":
            class_embeds = self.class_embedding_top.expand(batch_size, 1, -1).to(target_dtype)
        else:
            class_embeds = self.class_embedding_bottom.expand(batch_size, 1, -1).to(target_dtype)
        
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        if high_res_crop_location == "top":
            embedding_to_interpolate = self.vision_model.embeddings.position_embedding[:, :129, :]
        else:
            embedding_bottom = self.vision_model.embeddings.position_embedding[:, 129:, :]
            embedding_to_interpolate = torch.cat([self.class_embedding_bottom, embedding_bottom], dim=1)
        position_embedding = self.interpolate_pos_encoding(embedding_to_interpolate, height, width)
        embeddings = embeddings + position_embedding[:, : embeddings.size(1), :].to(target_dtype)

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        hidden_states = embeddings

        encoder_outputs = self.vision_model.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.vision_model.post_layernorm(last_hidden_state)

        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.vision_model.post_layernorm(pooled_output)

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def handle_high_res_image(
        self, 
        pixel_values_1,
        qformer_input_ids_1,
        qformer_attention_mask_1,
        pixel_values_2,
        qformer_input_ids_2,
        qformer_attention_mask_2,
        output_attentions,
        output_hidden_states,
        return_dict,
        use_generate=False
        
    ):
        # Step 1: get image embeddings for both crops of high resolution image
        vision_outputs_top = self.get_high_res_crop_vision_output(pixel_values_1,
                                                          output_attentions,
                                                          output_hidden_states,
                                                          return_dict,
                                                          "top"
                                                          )

        if use_generate:
            image_embeds_top = vision_outputs_top.last_hidden_state
        else:
            image_embeds_top = vision_outputs_top[0]

        vision_outputs_bottom = self.get_high_res_crop_vision_output(pixel_values_2,
                                                              output_attentions,
                                                              output_hidden_states,
                                                              return_dict,
                                                              "bottom"
                                                             )

        if use_generate:
            image_embeds_bottom = vision_outputs_bottom.last_hidden_state
        else:
            image_embeds_bottom = vision_outputs_bottom[0]

        image_embeds = torch.cat([image_embeds_top, image_embeds_bottom], dim=1)

        qformer_input_ids = qformer_input_ids_1
        qformer_attention_mask = qformer_attention_mask_1

        # step 2: forward the query tokens through the QFormer, using the image embeddings for cross-attention
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)

        # difference with BLIP-2 here: we also feed the instruction prompt to the Q-Former
        query_tokens = self.high_res_query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_attention_mask = torch.ones(query_tokens.size()[:-1], dtype=torch.long, device=image_embeds.device)
        if qformer_attention_mask is None:
            qformer_attention_mask = torch.ones_like(qformer_input_ids)
        qformer_attention_mask = torch.cat([query_attention_mask, qformer_attention_mask], dim=1)
        query_outputs = self.qformer(
            input_ids=qformer_input_ids,
            attention_mask=qformer_attention_mask,
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if use_generate:
            query_output = query_outputs.last_hidden_state[:, : query_tokens.size(1), :]
        else:
            query_output = query_outputs[0][:, : query_tokens.size(1), :]

        # step 3: use the language model, conditioned on the query outputs and the prompt
        language_model_inputs = self.language_projection(query_output)
        language_model_attention_mask = torch.ones(
            language_model_inputs.size()[:-1], dtype=torch.long, device=language_model_inputs.device
        )

        return language_model_inputs, language_model_attention_mask

    
    def forward(
        self,
        pixel_values_0: torch.FloatTensor,
        qformer_input_ids_0: torch.FloatTensor,
        qformer_attention_mask_0: Optional[torch.LongTensor] = None,
        pixel_values_1: Optional[torch.FloatTensor] = None,
        qformer_input_ids_1: Optional[torch.FloatTensor] = None,
        qformer_attention_mask_1: Optional[torch.LongTensor] = None,
        pixel_values_2: Optional[torch.FloatTensor] = None,
        qformer_input_ids_2: Optional[torch.FloatTensor] = None,
        qformer_attention_mask_2: Optional[torch.LongTensor] = None,
        input_ids: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        interpolate_pos_encoding: bool = False,
    ) -> Union[Tuple, InstructBlipForConditionalGenerationModelOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # language_model_inputs_low_res, language_model_attention_mask_low_res = self.handle_low_res_image(pixel_values_0,
        #                                                                                                  qformer_input_ids_0,
        #                                                                                                  qformer_attention_mask_0,
        #                                                                                                  output_attentions,
        #                                                                                                  output_hidden_states,
        #                                                                                                  return_dict,
        #                                                                                                  interpolate_pos_encoding
        #                                                                                                  )
        
        language_model_inputs_high_res, language_model_attention_mask_high_res = self.handle_high_res_image(pixel_values_1,
                                                                                                           qformer_input_ids_1,
                                                                                                           qformer_attention_mask_1,
                                                                                                           pixel_values_2,
                                                                                                           qformer_input_ids_2,
                                                                                                           qformer_attention_mask_2,
                                                                                                           output_attentions,
                                                                                                           output_hidden_states,
                                                                                                           return_dict,
                                                                                                           )

       
        # batch_size = pixel_values_0.shape[0]
        # eos_token_ids = torch.LongTensor([[self.config.text_config.eos_token_id]]).repeat(batch_size, 1).to(pixel_values_0.device)
        # eos_token_embedding = self.language_model.get_input_embeddings()(eos_token_ids)
        # eos_token_attention_mask = torch.LongTensor([[1]]).repeat(batch_size, 1).to(pixel_values_0.device)
        
        # language_model_inputs = torch.cat([language_model_inputs_low_res, 
        #                                    eos_token_embedding,
        #                                    language_model_inputs_high_res
        #                                   ], dim=1)

        # language_model_attention_mask = torch.cat([language_model_attention_mask_low_res, 
        #                                            eos_token_attention_mask,
        #                                            language_model_attention_mask_high_res
        #                                           ], dim=1)

        language_model_inputs = language_model_inputs_high_res
        language_model_attention_mask = language_model_attention_mask_high_res

        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        inputs_embeds = torch.cat([language_model_inputs, inputs_embeds.to(language_model_inputs.device)], dim=1)

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        attention_mask = torch.cat([language_model_attention_mask.to(attention_mask.device), attention_mask], dim=1)

        if self.config.use_decoder_only_language_model:
            outputs = self.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            logits = outputs.logits if return_dict else outputs[0]
            loss = None
            # we compute the loss here since we need to take into account the sequence length of the query embeds
            if labels is not None:
                labels = labels.to(logits.device)
                logits = logits[:, -labels.size(1) :, :]
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous().to(logits.device)

                # Flatten the tokens
                loss_fct = CrossEntropyLoss(reduction="mean")

                loss = loss_fct(shift_logits.view(-1, self.config.text_config.vocab_size), shift_labels.view(-1))
        else:
            outputs = self.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                labels=labels,
            )
            loss = outputs.loss if return_dict else outputs[0]
            logits = outputs.logits if return_dict else outputs[1]

        if not return_dict:
            output = (logits, vision_outputs, query_outputs, outputs)
            return ((loss,) + output) if loss is not None else output

        return InstructBlipForConditionalGenerationModelOutput(
            loss=loss,
            logits=logits,
            vision_outputs=None,
            qformer_outputs=None,
            language_model_outputs=outputs,
        )

    @torch.no_grad()
    def generate(
        self,
        pixel_values_0: torch.FloatTensor,
        qformer_input_ids_0: Optional[torch.LongTensor] = None,
        qformer_attention_mask_0: Optional[torch.LongTensor] = None,
        pixel_values_1: Optional[torch.FloatTensor] = None,
        qformer_input_ids_1: Optional[torch.LongTensor] = None,
        qformer_attention_mask_1: Optional[torch.LongTensor] = None,
        pixel_values_2: Optional[torch.FloatTensor] = None,
        qformer_input_ids_2: Optional[torch.LongTensor] = None,
        qformer_attention_mask_2: Optional[torch.LongTensor] = None,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        interpolate_pos_encoding: bool = False,
        **generate_kwargs,
    ) -> torch.LongTensor:

        if hasattr(self, "hf_device_map"):
            # preprocess for `accelerate`
            self._preprocess_accelerate()

        batch_size = pixel_values_0.shape[0]


        # language_model_inputs_low_res, language_attention_mask_low_res = self.handle_low_res_image(pixel_values = pixel_values_0,
        #                                                                                            qformer_input_ids = qformer_input_ids_0,
        #                                                                                            qformer_attention_mask = qformer_attention_mask_0,
        #                                                                                            output_attentions = None,
        #                                                                                            output_hidden_states = None,
        #                                                                                            return_dict = True,
        #                                                                                            interpolate_pos_encoding = interpolate_pos_encoding,
        #                                                                                            use_generate = True
        #                                                                                           )

        language_model_inputs_high_res, language_attention_mask_high_res = self.handle_high_res_image(pixel_values_1 = pixel_values_1,
                                                                                                     qformer_input_ids_1 = qformer_input_ids_1,
                                                                                                     qformer_attention_mask_1 = qformer_attention_mask_1,
                                                                                                     pixel_values_2 = pixel_values_2,
                                                                                                     qformer_input_ids_2 = qformer_input_ids_2,
                                                                                                     qformer_attention_mask_2 = qformer_attention_mask_2,
                                                                                                     output_attentions = None,
                                                                                                     output_hidden_states = None,
                                                                                                     return_dict = True,
                                                                                                     use_generate = True
                                                                                                    )

        # batch_size = pixel_values_0.shape[0]
        # eos_token_ids = torch.LongTensor([[self.config.text_config.eos_token_id]]).repeat(batch_size, 1).to(pixel_values_0.device)
        # eos_token_embedding = self.language_model.get_input_embeddings()(eos_token_ids)
        # eos_token_attention_mask = torch.LongTensor([[1]]).repeat(batch_size, 1).to(pixel_values_0.device)
        
        # language_model_inputs = torch.cat([language_model_inputs_low_res, 
        #                                    eos_token_embedding,
        #                                    language_model_inputs_high_res
        #                                   ], dim=1)

        # language_attention_mask = torch.cat([language_attention_mask_low_res, 
        #                                     eos_token_attention_mask,
        #                                     language_attention_mask_high_res
        #                                     ], dim=1)

        language_model_inputs = language_model_inputs_high_res
        language_attention_mask = language_attention_mask_high_res


        if input_ids is None:
            input_ids = (
                torch.LongTensor([[self.config.text_config.bos_token_id]])
                .repeat(batch_size, 1)
                .to(pixel_values_0.device) # original = image_embeds.device
            )
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        attention_mask = torch.cat([language_attention_mask, attention_mask.to(language_attention_mask.device)], dim=1)

        # concatenate query embeddings with prompt embeddings
        inputs_embeds = self.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([language_model_inputs, inputs_embeds.to(language_model_inputs.device)], dim=1)

        # add image_embeds length to max_length, so that the final max_length in counted only on token embeds
        # -1 is to account for the prepended BOS after `generate.`
        if not self.language_model.config.is_encoder_decoder:
            generate_kwargs["max_length"] = generate_kwargs.get("max_length", 20) + language_model_inputs.shape[1] - 1
            generate_kwargs["min_length"] = generate_kwargs.get("min_length", 0) + language_model_inputs.shape[1]

        outputs = self.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **generate_kwargs,
        )

        # this is a temporary workaround to be consistent with other generation models and
        # have BOS as the first token, even though under the hood we are calling LM with embeds
        if not self.language_model.config.is_encoder_decoder:
            # the InstructBLIP authors used inconsistent tokenizer/model files during training,
            # with the tokenizer's bos token being set to </s> which has ID=2,
            # whereas the model's text config has bos token id = 0
            bos_token_id = (
                2
                if self.config.text_config.architectures[0] == "LLaMAForCausalLM"
                else self.config.text_config.bos_token_id
            )
            bos_tokens = torch.LongTensor([[bos_token_id]]).repeat(batch_size, 1).to(image_embeds.device)
            if not isinstance(outputs, torch.Tensor):
                outputs.sequences = torch.cat([bos_tokens, outputs.sequences], dim=-1)
            else:
                outputs = torch.cat([bos_tokens, outputs], dim=-1)

        return outputs


        