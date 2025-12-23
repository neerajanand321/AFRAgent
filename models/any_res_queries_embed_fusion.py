import torch
from transformers import InstructBlipForConditionalGeneration 
from transformers.utils import ModelOutput
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

class QueriesFusionInstructBlipForConditionalGeneration(InstructBlipForConditionalGeneration):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.resolution_type_embedding = nn.Embedding(2, 1408)
        self.high_res_crop_position_embedding = self.sinusoidal_positional_embedding(2, 1408)

    def sinusoidal_positional_embedding(self, token_sequence_size, token_embedding_dim, n=10000.0):

        if token_embedding_dim % 2 != 0:
            raise ValueError("Sinusoidal positional embedding cannot apply to odd token embedding dim (got dim={:d})".format(token_embedding_dim))
    
        T = token_sequence_size
        d = token_embedding_dim #d_model=head_num*d_k, not d_q, d_k, d_v
    
        positions = torch.arange(0, T).unsqueeze_(1)
        embeddings = torch.zeros(T, d)
    
        denominators = torch.pow(n, 2*torch.arange(0, d//2)/d) # 10000^(2i/d_model), i is the index of embedding
        embeddings[:, 0::2] = torch.sin(positions/denominators) # sin(pos/10000^(2i/d_model))
        embeddings[:, 1::2] = torch.cos(positions/denominators) # cos(pos/10000^(2i/d_model))
    
        return embeddings

    def _forward(
        self,
        pixel_values,
        qformer_input_ids,
        qformer_attention_mask,
        output_attentions,
        output_hidden_states,
        return_dict,
        interpolate_pos_encoding,
        res_type_embeddings,
        pos_type_embeddings = None,
        image_type = "low_res"
    ):
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )
        
        image_embeds = vision_outputs[0]
        if image_type == "low_res":
            image_embeds += res_type_embeddings[:, :257, :]
        elif image_type == "high_res_top":
            image_embeds += res_type_embeddings[:, 257:514, :]
            image_embeds += pos_type_embeddings[:, :257, :]
        else:
            image_embeds += res_type_embeddings[:, 514:, :]
            image_embeds += pos_type_embeddings[:, 257:, :]

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

        res_type_mask = torch.zeros(pixel_values_0.shape[0], 771, dtype = torch.long).to(pixel_values_0.device)
        res_type_mask[257:] = 1

        res_type_embeddings = self.resolution_type_embedding(res_type_mask)

        high_res_mask = torch.zeros(pixel_values_0.shape[0], 514, 1).to(pixel_values_0.device)
        high_res_mask[:, 257:, :] = 1

        high_res_crop_position_embedding = self.high_res_crop_position_embedding.to(pixel_values_0.device)

        high_res_pos_embeds = (1 - high_res_mask) * high_res_crop_position_embedding[0].unsqueeze(0) + high_res_mask * high_res_crop_position_embedding[1].unsqueeze(0)

        
        language_model_inputs_0, language_model_attention_mask_0 = self._forward(pixel_values_0,
                                                                                 qformer_input_ids_0,
                                                                                 qformer_attention_mask_0,
                                                                                 output_attentions,
                                                                                 output_hidden_states,
                                                                                 return_dict,
                                                                                 interpolate_pos_encoding,
                                                                                 res_type_embeddings,
                                                                                 high_res_pos_embeds,
                                                                                 "low_res"
                                                                                )
        if pixel_values_1 is not None and qformer_input_ids_1 is not None:
            language_model_inputs_1, language_model_attention_mask_1 = self._forward(pixel_values_1,
                                                                                     qformer_input_ids_1,
                                                                                     qformer_attention_mask_1,
                                                                                     output_attentions,
                                                                                     output_hidden_states,
                                                                                     return_dict,
                                                                                     interpolate_pos_encoding,
                                                                                     res_type_embeddings,
                                                                                     high_res_pos_embeds,
                                                                                     "high_res_top"
                                                                                    )
        if pixel_values_2 is not None and qformer_input_ids_2 is not None:
            language_model_inputs_2, language_model_attention_mask_2 = self._forward(pixel_values_2,
                                                                                     qformer_input_ids_2,
                                                                                     qformer_attention_mask_2,
                                                                                     output_attentions,
                                                                                     output_hidden_states,
                                                                                     return_dict,
                                                                                     interpolate_pos_encoding,
                                                                                     res_type_embeddings,
                                                                                     high_res_pos_embeds,
                                                                                     "high_res_bottom"
                                                                                    )

        if pixel_values_1 is not None and pixel_values_2 is not None:
            language_model_inputs = torch.cat([language_model_inputs_0,
                                               language_model_inputs_1.to(language_model_inputs_0.device),
                                               language_model_inputs_2.to(language_model_inputs_0.device)
                                              ], dim=1)
                                                                          
            language_model_attention_mask = torch.cat([language_model_attention_mask_0, 
                                                       language_model_attention_mask_1.to(language_model_attention_mask_0.device),
                                                       language_model_attention_mask_2.to(language_model_attention_mask_0.device)
                                                      ], dim=1)

        else:
            language_model_inputs = language_model_inputs_0
            language_model_attention_mask = language_model_attention_mask_0
        

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


    def _generate(
        self,
        pixel_values,
        interpolate_pos_encoding,
        qformer_input_ids,
        qformer_attention_mask,
        res_type_embeddings,
        pos_type_embeddings = None,
        image_type = "low_res"
    ):
        image_embeds = self.vision_model(
            pixel_values,
            return_dict=True,
            interpolate_pos_encoding=interpolate_pos_encoding,
        ).last_hidden_state

        if image_type == "low_res":
            image_embeds += res_type_embeddings[:, :257, :]
        elif image_type == "high_res_top":
            image_embeds += res_type_embeddings[:, 257:514, :]
            image_embeds += pos_type_embeddings[:, :257, :]
        else:
            image_embeds += res_type_embeddings[:, 514:, :]
            image_embeds += pos_type_embeddings[:, 257:, :]
        
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)

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
            return_dict=True,
        )
        query_output = query_outputs.last_hidden_state[:, : query_tokens.size(1), :]

        language_model_inputs = self.language_projection(query_output)
        language_attention_mask = torch.ones(
            language_model_inputs.size()[:-1], dtype=torch.long, device=language_model_inputs.device
        )
        return language_model_inputs, language_attention_mask
    
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

        res_type_mask = torch.zeros(pixel_values_0.shape[0], 771, dtype = torch.long).to(pixel_values_0.device)
        res_type_mask[257:] = 1

        res_type_embeddings = self.resolution_type_embedding(res_type_mask)

        high_res_mask = torch.zeros(pixel_values_0.shape[0], 514, 1).to(pixel_values_0.device)
        high_res_mask[:, 257:, :] = 1

        high_res_crop_position_embedding = self.high_res_crop_position_embedding.to(pixel_values_0.device)

        high_res_pos_embeds = (1 - high_res_mask) * high_res_crop_position_embedding[0].unsqueeze(0) + high_res_mask * high_res_crop_position_embedding[1].unsqueeze(0)
        
        language_model_inputs_0, language_attention_mask_0 = self._generate(pixel_values_0,
                                                                            interpolate_pos_encoding,
                                                                            qformer_input_ids_0,
                                                                            qformer_attention_mask_0,
                                                                            res_type_embeddings,
                                                                            high_res_pos_embeds,
                                                                            "low_res"
                                                                           )
        
        if pixel_values_1 is not None:
            language_model_inputs_1, language_attention_mask_1 = self._generate(pixel_values_1,
                                                                                interpolate_pos_encoding,
                                                                                qformer_input_ids_1,
                                                                                qformer_attention_mask_1,
                                                                                res_type_embeddings,
                                                                                high_res_pos_embeds,
                                                                                "high_res_top"
                                                                               )

        if pixel_values_2 is not None:
            language_model_inputs_2, language_attention_mask_2 = self._generate(pixel_values_2,
                                                                                interpolate_pos_encoding,
                                                                                qformer_input_ids_2,
                                                                                qformer_attention_mask_2,
                                                                                res_type_embeddings,
                                                                                high_res_pos_embeds,
                                                                                "high_res_bottom"
                                                                               )

        if pixel_values_1 is not None and pixel_values_2 is not None:
            language_model_inputs = torch.cat([language_model_inputs_0,
                                               language_model_inputs_1.to(language_model_inputs_0.device),
                                               language_model_inputs_2.to(language_model_inputs_0.device)
                                              ], dim=1)
            
            language_attention_mask = torch.cat([language_attention_mask_0,
                                                 language_attention_mask_1.to(language_attention_mask_0.device),
                                                 language_attention_mask_2.to(language_attention_mask_0.device)
                                                ], dim=1)

        else:
            language_model_inputs = language_model_inputs_0
            language_attention_mask = language_attention_mask_0

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





        