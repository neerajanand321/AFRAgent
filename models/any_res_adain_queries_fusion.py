import torch
from transformers import InstructBlipForConditionalGeneration, InstructBlipVisionModel,  InstructBlipVisionConfig
from transformers.utils import ModelOutput
from transformers.modeling_outputs import BaseModelOutputWithPooling
from torch.nn import CrossEntropyLoss
import torch.nn as nn
from typing import Optional, Tuple, Any, Union
from dataclasses import dataclass
from torch.distributions.normal import Normal
from .utils import get_2d_sincos_pos_embed_vertical_split
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

class AnyResAdaIn(InstructBlipForConditionalGeneration):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.new_query_tokens_low_res =  nn.Parameter(torch.randn(1, 257, self.config.qformer_config.hidden_size))
        self.new_query_tokens_high_res = nn.Parameter(torch.randn(1, 257, self.config.qformer_config.hidden_size))
        modules_adain_scale_low_res = [nn.Linear(self.config.vision_config.hidden_size, self.config.vision_config.hidden_size)]
        modules_adain_scale_low_res.append(nn.GELU())
        modules_adain_scale_low_res.append(nn.Linear(self.config.vision_config.hidden_size, self.config.qformer_config.hidden_size))
        self.mlp_adain_scale_low_res = nn.Sequential(*modules_adain_scale_low_res)
        
        modules_adain_shift_low_res = [nn.Linear(self.config.vision_config.hidden_size, self.config.vision_config.hidden_size)]
        modules_adain_shift_low_res.append(nn.GELU())
        modules_adain_shift_low_res.append(nn.Linear(self.config.vision_config.hidden_size, self.config.qformer_config.hidden_size))
        self.mlp_adain_shift_low_res = nn.Sequential(*modules_adain_shift_low_res)

        modules_adain_scale_high_res = [nn.Linear(self.config.qformer_config.hidden_size, self.config.qformer_config.hidden_size)]
        modules_adain_scale_high_res.append(nn.GELU())
        modules_adain_scale_high_res.append(nn.Linear(self.config.qformer_config.hidden_size, self.config.qformer_config.hidden_size))
        self.mlp_adain_scale_high_res = nn.Sequential(*modules_adain_scale_high_res)

        modules_adain_shift_high_res = [nn.Linear(self.config.qformer_config.hidden_size, self.config.qformer_config.hidden_size)]
        modules_adain_shift_high_res.append(nn.GELU())
        modules_adain_shift_high_res.append(nn.Linear(self.config.qformer_config.hidden_size, self.config.qformer_config.hidden_size))
        self.mlp_adain_shift_high_res = nn.Sequential(*modules_adain_shift_high_res)

        # absolute position embedding
        self.low_res_abs_pos_embedding = torch.from_numpy(get_2d_sincos_pos_embed_vertical_split(embed_dim=1408, percent_start=0, percent_end=1))
        self.high_res_crop_first_abs_pos_embedding = torch.from_numpy(get_2d_sincos_pos_embed_vertical_split(embed_dim=1408, percent_start=0, percent_end=0.31))
        self.high_res_crop_second_abs_pos_embedding = torch.from_numpy(get_2d_sincos_pos_embed_vertical_split(embed_dim=1408, percent_start=0.23, percent_end=0.54))
        self.high_res_crop_third_abs_pos_embedding = torch.from_numpy(get_2d_sincos_pos_embed_vertical_split(embed_dim=1408, percent_start=0.46, percent_end=0.77))
        self.high_res_crop_fourth_abs_pos_embedding = torch.from_numpy(get_2d_sincos_pos_embed_vertical_split(embed_dim=1408, percent_start=0.69, percent_end=1))

        # Learnable position embedding
        self.any_res_pos_embedding = nn.Parameter(torch.randn(5, 257, self.config.vision_config.hidden_size))

    def AdaIn(
        self, 
        image_embeds,
        qformer_input_ids,
        qformer_attention_mask,
        output_attentions,
        output_hidden_states,
        return_dict,
        use_generate
    ):
        image_embeds_low_res, image_embeds_high_res_crop_first, image_embeds_high_res_crop_second, image_embeds_high_res_crop_third, image_embeds_high_res_crop_fourth = image_embeds

        # Adding position embedding to improve performance of adain 
        #image_embeds_low_res = image_embeds_low_res + self.any_res_pos_embedding[0].to(image_embeds_low_res.device, dtype=torch.bfloat16)
        #image_embeds_high_res_crop_first = image_embeds_high_res_crop_first + self.any_res_pos_embedding[1].to(image_embeds_high_res_crop_first.device, dtype=torch.bfloat16)
        #image_embeds_high_res_crop_second = image_embeds_high_res_crop_second + self.any_res_pos_embedding[2].to(image_embeds_high_res_crop_second.device, dtype=torch.bfloat16)
        #image_embeds_high_res_crop_third = image_embeds_high_res_crop_third + self.any_res_pos_embedding[3].to(image_embeds_high_res_crop_third.device, dtype=torch.bfloat16)
        #image_embeds_high_res_crop_fourth = image_embeds_high_res_crop_fourth + self.any_res_pos_embedding[4].to(image_embeds_high_res_crop_fourth.device, dtype=torch.bfloat16)
        
        image_attention_mask = torch.ones(image_embeds_low_res.size()[:-1], dtype=torch.long, device=image_embeds_low_res.device)

        # difference with BLIP-2 here: we also feed the instruction prompt to the Q-Former
        query_tokens = self.new_query_tokens_low_res.expand(image_embeds_low_res.shape[0], -1, -1)
        query_attention_mask = torch.ones(query_tokens.size()[:-1], dtype=torch.long, device=image_embeds_low_res.device)
        if qformer_attention_mask is None:
            qformer_attention_mask = torch.ones_like(qformer_input_ids)
        qformer_attention_mask = torch.cat([query_attention_mask, qformer_attention_mask], dim=1)
        query_outputs = self.qformer(
            input_ids=qformer_input_ids,
            attention_mask=qformer_attention_mask,
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds_low_res,
            encoder_attention_mask=image_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if use_generate:
            query_output = query_outputs.last_hidden_state[:, : query_tokens.size(1), :]
        else:
            query_output = query_outputs[0][:, : query_tokens.size(1), :]

        alpha_low_res = self.mlp_adain_scale_low_res(image_embeds_low_res)
        beta_low_res = self.mlp_adain_shift_low_res(image_embeds_low_res)

        # Adain between queries tokens of low resolution image and their encoder reprsentation
        output_low_res = alpha_low_res*query_output + beta_low_res

        image_embeds_high_res = torch.cat([image_embeds_high_res_crop_first, image_embeds_high_res_crop_second, image_embeds_high_res_crop_third, image_embeds_high_res_crop_fourth], dim=1)
        image_attention_mask_high_res = torch.ones(image_embeds_high_res.size()[:-1], dtype=torch.long, device=image_embeds_high_res.device)

        query_tokens_high_res = self.new_query_tokens_high_res.expand(image_embeds_high_res.shape[0], -1, -1)
        query_outputs_high_res = self.qformer(
            input_ids=qformer_input_ids,
            attention_mask=qformer_attention_mask,
            query_embeds=query_tokens_high_res,
            encoder_hidden_states=image_embeds_high_res,
            encoder_attention_mask=image_attention_mask_high_res,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if use_generate:
            query_output_high_res = query_outputs_high_res.last_hidden_state[:, : query_tokens_high_res.size(1), :]
        else:
            query_output_high_res = query_outputs_high_res[0][:, : query_tokens_high_res.size(1), :]

        alpha_high_res = self.mlp_adain_scale_high_res(query_output_high_res)
        beta_high_res = self.mlp_adain_shift_high_res(query_output_high_res)

        # output = (alpha_high_res*output_low_res + beta_high_res) + output_low_res
        output = alpha_high_res*output_low_res + beta_high_res
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
        high_res = False,
        use_generate = False
    ):
        pixel_values_0, pixel_values_1, pixel_values_2, pixel_values_3, pixel_values_4 = pixel_values
        qformer_input_ids_0, qformer_input_ids_1, qformer_input_ids_2, qformer_input_ids_3, qformer_input_ids_4 = qformer_input_ids
        qformer_attention_mask_0, qformer_attention_mask_1, qformer_attention_mask_2, qformer_attention_mask_3, qformer_attention_mask_4 = qformer_attention_mask
        
        # step 1: get image embeddings for low resolution image
        vision_outputs_low_res = self.vision_model(
            pixel_values=pixel_values_0,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )
        if use_generate:
            image_embeds_low_res = vision_outputs_low_res.last_hidden_state
        else:
            image_embeds_low_res = vision_outputs_low_res[0]

        # step 2: get image embeddings for both the crop of high resolution image
        vision_outputs_high_res_crop_first = self.vision_model(
            pixel_values=pixel_values_1,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )
            

        if use_generate:
            image_embeds_high_res_crop_first = vision_outputs_high_res_crop_first.last_hidden_state
        else:
            image_embeds_high_res_crop_first = vision_outputs_high_res_crop_first[0]

        vision_outputs_high_res_crop_second = self.vision_model(
            pixel_values=pixel_values_2,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )

        if use_generate:
            image_embeds_high_res_crop_second = vision_outputs_high_res_crop_second.last_hidden_state
        else:
            image_embeds_high_res_crop_second = vision_outputs_high_res_crop_second[0]

        vision_outputs_high_res_crop_third = self.vision_model(
            pixel_values=pixel_values_3,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )

        if use_generate:
            image_embeds_high_res_crop_third = vision_outputs_high_res_crop_third.last_hidden_state
        else:
            image_embeds_high_res_crop_third = vision_outputs_high_res_crop_third[0]

        vision_outputs_high_res_crop_fourth = self.vision_model(
            pixel_values=pixel_values_4,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )

        if use_generate:
            image_embeds_high_res_crop_fourth = vision_outputs_high_res_crop_fourth.last_hidden_state
        else:
            image_embeds_high_res_crop_fourth = vision_outputs_high_res_crop_fourth[0]

        image_embeds = (image_embeds_low_res, image_embeds_high_res_crop_first, image_embeds_high_res_crop_second, image_embeds_high_res_crop_third, image_embeds_high_res_crop_fourth)

        # step 3: apply adain operation for both low resolution and high resolution image
        moe_output = self.AdaIn(image_embeds,
                                qformer_input_ids_0,
                                qformer_attention_mask_0,
                                output_attentions,
                                output_hidden_states,
                                return_dict,
                                use_generate
                                )
        
        # step 4: use the language model, conditioned on the query outputs and the prompt
        language_model_inputs = self.language_projection(moe_output)
        language_model_attention_mask = torch.ones(
            language_model_inputs.size()[:-1], dtype=torch.long, device=language_model_inputs.device
        )

        return language_model_inputs, language_model_attention_mask


    def forward(
        self,
        pixel_values_0: torch.FloatTensor,
        qformer_input_ids_0: torch.FloatTensor = None,
        qformer_attention_mask_0: Optional[torch.LongTensor] = None,
        pixel_values_1: torch.FloatTensor = None,
        qformer_input_ids_1: torch.FloatTensor = None,
        qformer_attention_mask_1: Optional[torch.LongTensor] = None,
        pixel_values_2: torch.FloatTensor = None,
        qformer_input_ids_2: torch.FloatTensor = None,
        qformer_attention_mask_2: Optional[torch.LongTensor] = None,
        pixel_values_3: torch.FloatTensor = None,
        qformer_input_ids_3: torch.FloatTensor = None,
        qformer_attention_mask_3: Optional[torch.LongTensor] = None,
        pixel_values_4: torch.FloatTensor = None,
        qformer_input_ids_4: torch.FloatTensor = None,
        qformer_attention_mask_4: Optional[torch.LongTensor] = None,
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

        pixel_values = (pixel_values_0, pixel_values_1, pixel_values_2, pixel_values_3, pixel_values_4)
        qformer_input_ids = (qformer_input_ids_0, qformer_input_ids_1, qformer_input_ids_2, qformer_input_ids_3, qformer_input_ids_4)
        qformer_attention_mask = (qformer_attention_mask_0, qformer_attention_mask_1, qformer_attention_mask_2, qformer_attention_mask_3, qformer_attention_mask_4)

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
        pixel_values_0: torch.FloatTensor,
        qformer_input_ids_0: Optional[torch.LongTensor] = None,
        qformer_attention_mask_0: Optional[torch.LongTensor] = None,
        pixel_values_1: torch.FloatTensor = None,
        qformer_input_ids_1: Optional[torch.LongTensor] = None,
        qformer_attention_mask_1: Optional[torch.LongTensor] = None,
        pixel_values_2: torch.FloatTensor = None,
        qformer_input_ids_2: Optional[torch.LongTensor] = None,
        qformer_attention_mask_2: Optional[torch.LongTensor] = None,
        pixel_values_3: torch.FloatTensor = None,
        qformer_input_ids_3: Optional[torch.LongTensor] = None,
        qformer_attention_mask_3: Optional[torch.LongTensor] = None,
        pixel_values_4: torch.FloatTensor = None,
        qformer_input_ids_4: Optional[torch.LongTensor] = None,
        qformer_attention_mask_4: Optional[torch.LongTensor] = None,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        interpolate_pos_encoding: bool = False,
        **generate_kwargs,
    ) -> torch.LongTensor:

        if hasattr(self, "hf_device_map"):
            # preprocess for `accelerate`
            self._preprocess_accelerate()

        batch_size = pixel_values_0.shape[0]

        pixel_values = (pixel_values_0, pixel_values_1, pixel_values_2, pixel_values_3, pixel_values_4)
        qformer_input_ids = (qformer_input_ids_0, qformer_input_ids_1, qformer_input_ids_2, qformer_input_ids_3, qformer_input_ids_4)
        qformer_attention_mask = (qformer_attention_mask_0, qformer_attention_mask_1, qformer_attention_mask_2, qformer_attention_mask_3, qformer_attention_mask_4)

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


