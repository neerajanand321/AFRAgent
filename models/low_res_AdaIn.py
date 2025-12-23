import torch
from transformers import InstructBlipForConditionalGeneration, InstructBlipVisionModel,  InstructBlipVisionConfig
from transformers.utils import ModelOutput
from transformers.modeling_outputs import BaseModelOutputWithPooling
from torch.nn import CrossEntropyLoss
import torch.nn as nn
from typing import Optional, Tuple, Any, Union
from dataclasses import dataclass
from torch.distributions.normal import Normal
import numpy as np

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

class LowResAdaIn(InstructBlipForConditionalGeneration):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.new_query_tokens =  nn.Parameter(torch.randn(1, 257, self.config.qformer_config.hidden_size))
        modules_alpha = [nn.Linear(self.config.vision_config.hidden_size, self.config.vision_config.hidden_size)]
        modules_alpha.append(nn.GELU())
        modules_alpha.append(nn.Linear(self.config.vision_config.hidden_size, self.config.qformer_config.hidden_size))
        self.mlp_alpha = nn.Sequential(*modules_alpha)
        
        modules_beta = [nn.Linear(self.config.vision_config.hidden_size, self.config.vision_config.hidden_size)]
        modules_beta.append(nn.GELU())
        modules_beta.append(nn.Linear(self.config.vision_config.hidden_size, self.config.qformer_config.hidden_size))
        self.mlp_bias = nn.Sequential(*modules_beta)

    def AdaIn(
        self, 
        x,
        qformer_input_ids,
        qformer_attention_mask,
        output_attentions,
        output_hidden_states,
        return_dict,
        use_generate
    ):
        
        image_attention_mask = torch.ones(x.size()[:-1], dtype=torch.long, device=x.device)

        # difference with BLIP-2 here: we also feed the instruction prompt to the Q-Former
        query_tokens = self.new_query_tokens.expand(x.shape[0], -1, -1)
        query_attention_mask = torch.ones(query_tokens.size()[:-1], dtype=torch.long, device=x.device)
        if qformer_attention_mask is None:
            qformer_attention_mask = torch.ones_like(qformer_input_ids)
        qformer_attention_mask = torch.cat([query_attention_mask, qformer_attention_mask], dim=1)
        query_outputs = self.qformer(
            input_ids=qformer_input_ids,
            attention_mask=qformer_attention_mask,
            query_embeds=query_tokens,
            encoder_hidden_states=x,
            encoder_attention_mask=image_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if use_generate:
            query_output = query_outputs.last_hidden_state[:, : query_tokens.size(1), :]
        else:
            query_output = query_outputs[0][:, : query_tokens.size(1), :]

        # Add position embedding for improved performance of adain
        # alpha = self.mlp_alpha(x + self.vision_model.embeddings.position_embedding)
        # beta = self.mlp_bias(x + self.vision_model.embeddings.position_embedding)
        alpha = self.mlp_alpha(x)
        beta = self.mlp_bias(x)

        # Add residual connection for better performance of adain
        # output = (alpha*query_output + beta) + query_output
        output = alpha*query_output + beta
        return output

    def get_language_input(
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

        moe_output = self.AdaIn(image_embeds,
                              qformer_input_ids,
                              qformer_attention_mask,
                              output_attentions,
                              output_hidden_states,
                              return_dict,
                              use_generate
                             )
        
        # step 3: use the language model, conditioned on the query outputs and the prompt
        language_model_inputs = self.language_projection(moe_output)
        language_model_attention_mask = torch.ones(
            language_model_inputs.size()[:-1], dtype=torch.long, device=language_model_inputs.device
        )

        return language_model_inputs, language_model_attention_mask


    def forward(
        self,
        pixel_values: torch.FloatTensor,
        qformer_input_ids: torch.FloatTensor,
        qformer_attention_mask: Optional[torch.LongTensor] = None,
        input_ids: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        interpolate_pos_encoding: bool = True,
    ) -> Union[Tuple, InstructBlipForConditionalGenerationModelOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        language_model_inputs, language_model_attention_mask = self.get_language_input(pixel_values,
                                                                                       qformer_input_ids,
                                                                                       qformer_attention_mask,
                                                                                       output_attentions,
                                                                                       output_hidden_states,
                                                                                       return_dict,
                                                                                       interpolate_pos_encoding
                                                                                      )
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
        pixel_values: torch.FloatTensor,
        qformer_input_ids: Optional[torch.LongTensor] = None,
        qformer_attention_mask: Optional[torch.LongTensor] = None,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        interpolate_pos_encoding: bool = True,
        **generate_kwargs,
    ) -> torch.LongTensor:

        if hasattr(self, "hf_device_map"):
            # preprocess for `accelerate`
            self._preprocess_accelerate()

        batch_size = pixel_values.shape[0]

        language_model_inputs, language_attention_mask = self.get_language_input(pixel_values = pixel_values,
                                                                                 qformer_input_ids = qformer_input_ids,
                                                                                 qformer_attention_mask = qformer_attention_mask,
                                                                                 output_attentions = None,
                                                                                 output_hidden_states = None,
                                                                                 return_dict = True,
                                                                                 interpolate_pos_encoding = interpolate_pos_encoding,
                                                                                 use_generate=True
                                                                                )

        if input_ids is None:
            input_ids = (
                torch.LongTensor([[self.config.text_config.bos_token_id]])
                .repeat(batch_size, 1)
                .to(pixel_values.device) # original = image_embeds.device
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


