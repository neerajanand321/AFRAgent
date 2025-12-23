import os
import numpy as np 
import torch
import torch.distributed as dist
import os
import json
import re
import sys
import argparse
import random
import wandb
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration, T5ForConditionalGeneration
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, TrainerCallback
from utils.utils_data import AITWDatasetImg, load_data
from models.any_res_queries_embed_fusion import QueriesFusionInstructBlipForConditionalGeneration
from models.any_res_img_embed_fusion import ImagesFusionInstructBlipForConditionalGeneration
from models.low_res_qformer_MLP import LowResQformerMLPMoE
from models.low_res_AdaIn import LowResAdaIn
from models.any_res_adain_queries_fusion import AnyResAdaIn
from rich.table import Column, Table
from datasets import Dataset
from rich import box
from rich.console import Console
console = Console(record=True)
from utils import action_matching, action_type
import evaluate
import warnings
warnings.filterwarnings('ignore')

wandb.login(key = "9dbff17a9a0921d1a6ffb9f04ebfbfb56bbad8d9")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='dataset/aitw/general/general')
    parser.add_argument('--output_dir', type=str, default='experiments')
    parser.add_argument('--model', type=str, default='Salesforce/instructblip-flan-t5-xl')
    parser.add_argument('--model_name', type=str, default="InstructBlip")
    parser.add_argument('--data_ratio', type=float, default=None)
    parser.add_argument('--eval_name', type=str, default=None, help='the saved subset name used for evaluation')
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--epoch', type=int, default=12)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--bs', type=int, default=32)
    parser.add_argument('--debug_num', type=int, default=None)
    parser.add_argument('--input_len', type=int, default=512)
    parser.add_argument('--output_len', type=int, default=256)
    parser.add_argument('--img_dim', type=int, default=1408)
    parser.add_argument('--eval_bs', type=int, default=64)
    parser.add_argument('--eval_acc', type=int, default=None, help='evaluate accumulation step')
    parser.add_argument('--all_data', type=float, default=None, help='whether using all the data for training. Set the ratio for google apps to save computation')
    parser.add_argument('--eval_subset', type=str, default=None, help='use which subset for evaluation/test when training with all data')
    parser.add_argument('--eval_subset_1', type=str, default=None, help='use which subset for evaluation/test when training with all data')
    parser.add_argument('--eval_subset_2', type=str, default=None, help='use which subset for evaluation/test when training with all data')
    parser.add_argument('--all_eval', type=bool, default=False, help="Evaluate on all subset during all data training")
    parser.add_argument('--use_history', type=int, default=8, help='use textual action history')
    parser.add_argument('--use_qformer_history', type=int, default=8, help='use textual action history in qformer')
    parser.add_argument('--use_img_history', action='store_true', help='use screen history')
    parser.add_argument('--use_future', type=int, default=0, help='planning the future actions before giving the current action')
    parser.add_argument('--use_layout', action='store_true', help='use annotated layout information')
    parser.add_argument('--use_coco_agent_layout', default=False, action='store_true', help='use annotated layout info like coco agent')
    parser.add_argument('--transform_axis', default=True, action='store_true', help='use coordinate normalization')
    parser.add_argument('--use_generate', default=True, action='store_true', help='only for baseline to improve inference speed')
    parser.add_argument('--final_eval', action='store_true', help='only evaluate the model at the final epoch')
    parser.add_argument('--user_msg', type=str, default="all_data_low_res_adain_pretrained_finetuning", help='experiment type in the save_dir')
    parser.add_argument('--img_type', type=str, default="blip", help='type of image features')
    parser.add_argument('--img_size', type=int, default=224, help="size of image")
    parser.add_argument('--evaluate_dir', type=str, default=None, help='the directory of model for evaluation')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--gradient_checkpointing', type=bool, default=True)
    parser.add_argument('--num_workers', type=int, default=32)
    parser.add_argument('--debug', type=bool, default=False, help="When set true, print images, questions and answers")
    parser.add_argument('--drop_last', type=bool, default=False, help="To drop last batch")
    parser.add_argument('--topk_example', type=bool, default=False, help="To Pick k hardest examples from each batch")
    parser.add_argument('--use_high_res', type=bool, default=False, help="To use high resolution image in training and inference")
    parser.add_argument('--hard_example_path', type=str, default=None, help="Path to the index of hard examples")
    parser.add_argument('--use_image_fusion', type=bool, default=True, help="Wheather to fuse image embeddings of high res crop or not")
    parser.add_argument('--save_total_limit', type=int, default=2, help="numbers of checkpoint to save")
    parser.add_argument('--verbalized_output', type=bool, default=False, help="weather to verbalize the output")
    parser.add_argument('--bin_range', type=int, default=0, help="Range of binning")
    parser.add_argument('--check', type=bool, default=True, help="wheather to remove null text screens or not")
    parser.add_argument('--train_moe', type=bool, default=False, help="wheather to train moe or not")
    parser.add_argument('--train_low_res_adain', type=bool, default=False, help="wheather to train adain or not")
    parser.add_argument('--train_any_res_adain', type=bool, default=False, help="Wheather to train any res adain or not")
    parser.add_argument('--resume', type=str, default=None, help="Help to resume from checkpoint")
    parser.add_argument('--compute_rouge', type=bool, default=False, help="wheather to compute rouge or not")
    parser.add_argument('--compute_action_matching', type=bool, default=True, help="wheather to compute action matching or not")
    parser.add_argument('--oversample_single', type=bool, default=False, help="Oversample the single category")
    args = parser.parse_args()
    return args

# Define a global variable for epoch number
global_epoch = 0
global_eval_step = 0

class CustomTrainerCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        global global_epoch
        global_epoch = state.epoch

    def on_evaluate(self, args, state, control, **kwargs):
        global global_eval_step
        global_eval_step = global_eval_step + 1
            
if __name__ == '__main__':

    # training logger to log training progress
    training_logger = Table(
        Column("Epoch", justify="center"),
        Column("Steps", justify="center"),
        Column("Loss", justify="center"),
        title="Training Status",
        pad_edge=False,
        box=box.ASCII,
    )
    args = parse_args()
    print("args",args)
    print('====Input Arguments====')
    print(json.dumps(vars(args), indent=2, sort_keys=False))

    random.seed(args.seed)
    torch.manual_seed(args.seed)  # pytorch random seed
    np.random.seed(args.seed)  # numpy random seed
    torch.backends.cudnn.deterministic = True 
    
    if not os.path.exists(args.output_dir):
            os.mkdir(args.output_dir)
    if args.evaluate_dir is not None:
        args.model = args.evaluate_dir

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    print("Tokenizer Loaded")

    console.log(f"""[Model]: Loading {args.model}...\n""")
    console.log(f"[Data]: Reading data...\n")

    if args.debug_num:
        args.user_msg = "check"
        wandb.init(mode="disabled")
    
    if args.evaluate_dir is not None:
        save_dir = args.evaluate_dir
    else:
        model_name = args.model.replace("/","-")
        gpu_count = torch.cuda.device_count()
        save_dir = f"{args.output_dir}/{args.user_msg}_{args.img_type}_lr{args.lr}_bs{args.bs * gpu_count}_ip{args.input_len}_op{args.output_len}_ep{args.epoch}"
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
    print(save_dir)

    if args.use_high_res and args.use_image_fusion and not args.train_any_res_adain:
        model = ImagesFusionInstructBlipForConditionalGeneration.from_pretrained(args.model)
        print("High res image fusion model loaded")
    elif args.use_high_res and not args.train_any_res_adain:
        model = QueriesFusionInstructBlipForConditionalGeneration.from_pretrained(args.model)
        print("High res queries fusion model loaded")
    elif args.train_moe:
        model = LowResQformerMLPMoE.from_pretrained(args.model)
        print("MoE Model Loaded")
    elif args.train_low_res_adain:
        model = LowResAdaIn.from_pretrained(args.model)
        print("Low Res AdaIn Model Loaded")
    elif args.train_any_res_adain:
        model = AnyResAdaIn.from_pretrained(args.model)
        print("Any Res AdaIn Model Loaded")
    else:
        model = InstructBlipForConditionalGeneration.from_pretrained(args.model)
        print("Model loaded")
   
    for name, param in model.named_parameters():
        if('vision_model' in name):
            param.requires_grad = False

    if args.model_name == "InstructBlip":
        processor = InstructBlipProcessor.from_pretrained(args.model)
        processor.image_processor.size = {'height': args.img_size, 'width': args.img_size}
    else:
        processor = None
    print("Processor Loaded")
    train_test_val_map = {}
    train_test_val_map['train'] = 'train'
    train_test_val_map['val'] = 'val'
    train_test_val_map['test'] = 'test'
    if args.debug_num:
        train_test_val_map['train'] = 'test'
        train_test_val_map['val'] = 'test'
        train_test_val_map['test'] = 'test'
        args.epoch = 1
    
    if args.evaluate_dir is not None:
        train_set = None
    else:
        training_data = load_data(args, train_test_val_map['train'])
        train_set = AITWDatasetImg(
            data = training_data,
            split = "train",
            processor = processor,
            tokenizer = tokenizer,
            source_len = args.input_len,
            target_len = args.output_len,
            debug = args.debug,
            require_source_image_id = args.topk_example,
            use_high_res = args.use_high_res
            )
    eval_data = load_data(args, train_test_val_map['val'])
    if args.all_eval:
        eval_set_list = [
            AITWDatasetImg(
                data = eval_data[val_data],
                split="test",
                processor = processor,
                tokenizer = tokenizer,
                source_len = args.input_len,
                target_len = args.output_len,
                debug = args.debug,
                use_high_res = args.use_high_res,
            )
            for val_data in ["general", "single", "install", "google_apps", "web_shopping"]
        ]
    elif args.eval_subset_1 and args.eval_subset_2:
        eval_set_list = [
            AITWDatasetImg(
                data = eval_data[val_data],
                split="test",
                processor = processor,
                tokenizer = tokenizer,
                source_len = args.input_len,
                target_len = args.output_len,
                debug = args.debug,
                use_high_res = args.use_high_res,
            )
            for val_data in [args.eval_subset_1, args.eval_subset_2]
        ]
    else:
        eval_set = AITWDatasetImg(
            data = eval_data,
            split = "test",
            processor = processor,
            tokenizer = tokenizer,
            source_len = args.input_len,
            target_len = args.output_len,
            debug = args.debug,
            use_high_res = args.use_high_res
        )
    print("Prepared evaluation data")
    test_data = load_data(args, train_test_val_map['test'])
    if args.all_eval:
        test_set_list = [
            AITWDatasetImg(
                data = test_data[val_data],
                split="test",
                processor = processor,
                tokenizer = tokenizer,
                source_len = args.input_len,
                target_len = args.output_len,
                debug = args.debug,
                use_high_res = args.use_high_res,
            )
            for val_data in ["general", "single", "install", "google_apps", "web_shopping"]
        ]
    elif args.eval_subset_1 and args.eval_subset_2:
        test_set_list = [
            AITWDatasetImg(
                data = test_data[val_data],
                split="test",
                processor = processor, 
                tokenizer = tokenizer,
                source_len = args.input_len,
                target_len = args.output_len,
                debug = args.debug,
                use_high_res = args.use_high_res,
            )
            for val_data in [args.eval_subset_1, args.eval_subset_2]
        ]
    else:
        test_set = AITWDatasetImg(
            data = test_data,
            split = "test",
            processor = processor,
            tokenizer = tokenizer,
            source_len = args.input_len,
            target_len = args.output_len,
            debug = args.debug,
            use_high_res = args.use_high_res
        ) 
    print("Prepared test data")

    datacollator = DataCollatorForSeq2Seq(tokenizer)
    print("model parameters: ", model.num_parameters())

    eval_pointer = []
    test_pointer = []
    if args.all_eval:
        for i in range(5):
            eval_pointer.append([])
            test_pointer.append([])
        for i in range(args.epoch):
            for j in range(5):
                eval_pointer[j].append(i*10+j*2)
                test_pointer[j].append(i*10+j*2+1)

    elif args.eval_subset_1 and args.eval_subset_2:
        for i in range(2):
            eval_pointer.append([])
            test_pointer.append([]) 
        for i in range(args.epoch):
            eval_pointer[0].append(i*4)
            test_pointer[0].append(i*4+1)
            eval_pointer[1].append(i*4+2)
            test_pointer[1].append(i*4+3)
    else:
        for i in range(args.epoch):
            eval_pointer.append(i*2)
            test_pointer.append(i*2+1)

    # rougel for rationale generation
    metric = evaluate.load("rouge")
    overall_val_accuracy = 0
    overall_test_accuracy = 0
    def compute_rouge_and_action_matching_score(eval_preds):
        global global_epoch
        global global_eval_step
        global overall_test_accuracy
        global overall_val_accuracy
        
        preds, targets = eval_preds
        
        if isinstance(preds, tuple):
            preds = preds[0]
        preds= np.where(preds != -100, preds, tokenizer.pad_token_id)
        preds = tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        targets = tokenizer.batch_decode(targets, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        if args.compute_rouge:
            result_metric = metric.compute(predictions=preds, references=targets)
            result_metric = {k: round(v * 100, 4) for k, v in result_metric.items()}
            prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
            result_metric["gen_len"] = np.mean(prediction_lens)
        else:
            result_metric = {}

        
        if args.compute_action_matching:
            action_correct = 0
            text_correct = 0
            type_correct = 0

            output_data = []

            if args.all_eval:
                if global_eval_step%2 == 0:
                    if global_eval_step in eval_pointer[0]:
                        curr_dataset = eval_set_list[0]
                    elif global_eval_step in eval_pointer[1]:
                        curr_dataset = eval_set_list[1]
                    elif global_eval_step in eval_pointer[2]:
                        curr_dataset = eval_set_list[2]
                    elif global_eval_step in eval_pointer[3]:
                        curr_dataset = eval_set_list[3]
                    else:
                        curr_dataset = eval_set_list[4]
                else:
                    if global_eval_step in test_pointer[0]:
                        curr_dataset = test_set_list[0]
                    elif global_eval_step in test_pointer[1]:
                        curr_dataset = test_set_list[1]
                    elif global_eval_step in test_pointer[2]:
                        curr_dataset = test_set_list[2]
                    elif global_eval_step in test_pointer[3]:
                        curr_dataset = test_set_list[3]
                    else:
                        curr_dataset = test_set_list[4]
            elif args.eval_subset_1 and args.eval_subset_2:
                if global_eval_step in eval_pointer[0]:
                    curr_dataset = eval_set_list[0]
                elif global_eval_step in test_pointer[0]:
                    curr_dataset = test_set_list[0]
                elif global_eval_step in eval_pointer[1]:
                    curr_dataset = eval_set_list[1]
                else:
                    curr_dataset = test_set_list[1]
            else:
                if global_eval_step in eval_pointer:
                    curr_dataset = eval_set
                else:
                    curr_dataset = test_set 

            reference_test_positions = curr_dataset.anno_positions
            len_eval_data = len(reference_test_positions)
            targets = targets[:len_eval_data]
            preds = preds[:len_eval_data]
            
            pattern = r'(?<=Action Decision:\s).*'
            
            assert len(preds) == len(targets)  == len(reference_test_positions)
            for idx, pred in enumerate(preds):
                try:
                    if args.use_future:
                        result = re.search(pattern, targets[idx])
                        target_text = result.group(0)
                        target_text = target_text.strip()
                    else:
                        target_text = targets[idx]
                    if args.verbalized_output:
                        if 'tap at' in target_text:
                            loc = '['+target_text.split('[')[1]
                            target_text = f'"action_type": "DUAL_POINT", "touch_point": "{loc}", "lift_point": "{loc}", "typed_text": ""'
                        elif 'swipe' in target_text:
                            if 'up' in target_text:
                                loc1 = '[0.8, 0.5]'
                                loc2 = '[0.2, 0.5]'
                            elif 'down' in target_text:
                                loc1 = '[0.2, 0.5]'
                                loc2 = '[0.8, 0.5]'
                            elif 'left' in target_text:
                                loc1 = '[0.5, 0.8]'
                                loc2 = '[0.5, 0.2]'
                            elif 'right' in target_text:
                                loc1 = '[0.5, 0.2]'
                                loc2 = '[0.5, 0.8]'
                            else:
                                loc1 = '['+target_text.split('[')[1].split(']')[0]+']'
                                loc2 = '['+target_text.split('[')[2].split(']')[0]+']'
                            target_text = f'"action_type": "DUAL_POINT", "touch_point": "{loc1}", "lift_point": "{loc2}", "typed_text": ""'
                        elif 'press home' in target_text:
                            loc = '[-1.0, -1.0]'
                            target_text = f'"action_type": "PRESS_HOME", "touch_point": "{loc}", "lift_point": "{loc}", "typed_text": ""'
                        elif 'press back' in target_text:
                            loc = '[-1.0, -1.0]'
                            target_text = f'"action_type": "PRESS_BACK", "touch_point": "{loc}", "lift_point": "{loc}", "typed_text": ""'
                        elif 'press enter' in target_text:
                            loc = '[-1.0, -1.0]'
                            target_text = f'"action_type": "PRESS_ENTER", "touch_point": "{loc}", "lift_point": "{loc}", "typed_text": ""'
                        elif 'Input text' in target_text:
                            loc = '[-1.0, -1.0]'
                            inp_txt = " ".join(target_text.split(' ')[2:])
                            target_text = f'"action_type": "TYPE", "touch_point": "{loc}", "lift_point": "{loc}", "typed_text": {inp_txt}'
                        elif 'completed' in target_text:
                            loc = '[-1.0, -1.0]'
                            target_text = f'"action_type": "STATUS_TASK_COMPLETE", "touch_point": "{loc}", "lift_point": "{loc}", "typed_text": ""'
                        else:
                            loc = '[-1.0, -1.0]'
                            target_text = f'"action_type": "STATUS_TASK_IMPOSSIBLE", "touch_point": "{loc}", "lift_point": "{loc}", "typed_text": ""'

                    reference = eval("{" + target_text + "}")
                except:
                    print("reference error")
                    continue

                try:
                    if args.use_future:
                        result = re.search(pattern, preds[idx])
                        pred_text = result.group(0)
                        pred_text = pred_text.strip()
                    else:
                        pred_text = preds[idx]
                    if args.verbalized_output:
                        if 'tap at' in pred_text:
                            loc = '['+pred_text.split('[')[1]
                            pred_text = f'"action_type": "DUAL_POINT", "touch_point": "{loc}", "lift_point": "{loc}", "typed_text": ""'
                        elif 'swipe' in pred_text:
                            if 'up' in pred_text:
                                loc1 = '[0.8, 0.5]'
                                loc2 = '[0.2, 0.5]'
                            elif 'down' in pred_text:
                                loc1 = '[0.2, 0.5]'
                                loc2 = '[0.8, 0.5]'
                            elif 'left' in pred_text:
                                loc1 = '[0.5, 0.8]'
                                loc2 = '[0.5, 0.2]'
                            elif 'right' in pred_text:
                                loc1 = '[0.5, 0.2]'
                                loc2 = '[0.5, 0.8]'
                            else:
                                loc1 = '['+pred_text.split('[')[1].split(']')[0]+']'
                                loc2 = '['+pred_text.split('[')[2].split(']')[0]+']'
                            pred_text = f'"action_type": "DUAL_POINT", "touch_point": "{loc1}", "lift_point": "{loc2}", "typed_text": ""'
                        elif 'press home' in pred_text:
                            loc = '[-1.0, -1.0]'
                            pred_text = f'"action_type": "PRESS_HOME", "touch_point": "{loc}", "lift_point": "{loc}", "typed_text": ""'
                        elif 'press back' in pred_text:
                            loc = '[-1.0, -1.0]'
                            pred_text = f'"action_type": "PRESS_BACK", "touch_point": "{loc}", "lift_point": "{loc}", "typed_text": ""'
                        elif 'press enter' in pred_text:
                            loc = '[-1.0, -1.0]'
                            pred_text = f'"action_type": "PRESS_ENTER", "touch_point": "{loc}", "lift_point": "{loc}", "typed_text": ""'
                        elif 'Input text' in pred_text:
                            loc = '[-1.0, -1.0]'
                            inp_txt = " ".join(pred_text.split(' ')[2:])
                            pred_text = f'"action_type": "TYPE", "touch_point": "{loc}", "lift_point": "{loc}", "typed_text": {inp_txt}'
                        elif 'completed' in pred_text:
                            loc = '[-1.0, -1.0]'
                            pred_text = f'"action_type": "STATUS_TASK_COMPLETE", "touch_point": "{loc}", "lift_point": "{loc}", "typed_text": ""'
                        else:
                            loc = '[-1.0, -1.0]'
                            pred_text = f'"action_type": "STATUS_TASK_IMPOSSIBLE", "touch_point": "{loc}", "lift_point": "{loc}", "typed_text": ""'

                    pred = eval("{" + pred_text + "}")
                    action_1_touch_yx = eval(pred["touch_point"])
                    action_1_lift_yx = eval(pred["lift_point"])
                    action_1_action_type = action_type.ActionType[pred["action_type"]].value
                    action_1_typed_text = pred["typed_text"].lower()
                    action_1_typed_text = action_1_typed_text.strip()

                    action_1_wrap = f'"action_type": "{action_1_action_type}", "touch_point": "{action_1_touch_yx}", "lift_point": "{action_1_lift_yx}", "typed_text": "{action_1_typed_text}"'
                    action_1_wrap = action_1_wrap.replace('"', "'")
                except:
                    pred = '{ "action_type": "TYPE", "touch_point": "[-1.0, -1.0]", "lift_point": "[-1.0, -1.0]", "typed_text": "Invalid"}'
                
                action_2_touch_yx = eval(reference["touch_point"])
                action_2_lift_yx = eval(reference["lift_point"])
                action_2_action_type = action_type.ActionType[reference["action_type"]].value
                action_2_typed_text = reference["typed_text"].lower()
                
                action_2_wrap = f'"action_type": "{action_2_action_type}", "touch_point": "{action_2_touch_yx}", "lift_point": "{action_2_lift_yx}", "typed_text": "{action_2_typed_text}"'
                action_2_wrap = action_2_wrap.replace('"', "'")

                annotation_positions = reference_test_positions[idx]

                try:
                    check_match = action_matching.check_actions_match(
                        action_1_touch_yx,
                        action_1_lift_yx,
                        action_1_action_type,
                        action_2_touch_yx,
                        action_2_lift_yx,
                        action_2_action_type,
                        annotation_positions
                    )

                except Exception as exc:
                    print(idx, action_1_touch_yx, action_1_lift_yx)
                    check_match = False
                    match_label = "invalid"

                if check_match:
                    action_correct += 1
                    match_label = 1
                else:
                    match_label = 0
                if check_match and (action_1_typed_text in action_2_typed_text or action_2_typed_text in action_1_typed_text):
                    text_correct += 1
                if action_1_action_type == action_2_action_type:
                    type_correct += 1

                action_data = {"pred": action_1_wrap, "target": action_2_wrap, "match_label": match_label}
                output_data.append(action_data)
                    
            result_metric["accuracy"] = (action_correct/len(targets))*100
            result_metric["text_acc"] = (text_correct/len(targets))*100
            result_metric["type_acc"] = (type_correct/len(targets))*100
            result_metric["action_correct"] = action_correct
            result_metric["text_correct"] = text_correct
            result_metric["type_correct"] = type_correct

            if args.all_eval:
                if global_eval_step%2 == 0:
                    overall_val_accuracy += (action_correct/len(targets))*100
                    if global_eval_step in eval_pointer[4]:
                        result_metric['overall_val_accuracy'] = overall_val_accuracy/5
                        overall_val_accuracy = 0
                else:
                    overall_test_accuracy += (action_correct/len(targets))*100
                    if global_eval_step in test_pointer[4]:
                        result_metric['overall_test_accuracy'] = overall_test_accuracy/5
                        overall_test_accuracy = 0
            elif args.eval_subset_1 and args.eval_subset_2:
                if global_eval_step%2 == 0:
                    overall_val_accuracy += (action_correct/len(targets))*100
                    if global_eval_step in eval_pointer[1]:
                        result_metric['overall_val_accuracy'] = overall_val_accuracy/5
                        overall_val_accuracy = 0
                else:
                    overall_test_accuracy += (action_correct/len(targets))*100
                    if global_eval_step in test_pointer[1]:
                        result_metric['overall_test_accuracy'] = overall_test_accuracy/5
                        overall_test_accuracy = 0
            else:
                print("Single subset used in evaluation")
                
            output_data = {
                "metrics": result_metric,
                "data": output_data
            }

            output_prediction_file = os.path.join(save_dir, f"epoch_{global_epoch}_predictions_ans_test.json")
            with open(output_prediction_file, "w") as writer:
                writer.write(json.dumps(output_data, indent=4))

        return result_metric

    remove_unused_columns = True
    if args.topk_example:
        remove_unused_columns = False

    # only use the last model for evaluation to save time
    if args.final_eval:
        training_args = Seq2SeqTrainingArguments(
            save_dir,
            do_train=True if args.evaluate_dir is None else False,
            do_eval=False,
            warmup_ratio=args.warmup_ratio,
            evaluation_strategy="no",
            logging_strategy="steps",
            save_strategy="epoch",
            save_total_limit = args.save_total_limit,
            learning_rate= args.lr,
            eval_accumulation_steps=args.eval_acc,
            per_device_train_batch_size=args.bs,
            per_device_eval_batch_size=args.eval_bs,
            weight_decay=0.01,
            num_train_epochs=args.epoch,
            predict_with_generate=args.use_generate,
            generation_max_length=args.output_len,
            report_to="wandb",
            local_rank=args.local_rank,
            bf16_full_eval=True,
            gradient_checkpointing = args.gradient_checkpointing,
            dataloader_num_workers = args.num_workers,
            dataloader_drop_last = args.drop_last,
            remove_unused_columns = remove_unused_columns,
            save_only_model = True
        )
    # evaluate at each epoch
    else:
        training_args = Seq2SeqTrainingArguments(
            save_dir,
            do_train=True if args.evaluate_dir is None else False,
            do_eval=True,
            warmup_ratio=args.warmup_ratio,
            evaluation_strategy="epoch",
            logging_strategy="steps",
            save_strategy="epoch",
            save_total_limit = args.save_total_limit,
            learning_rate= args.lr,
            eval_accumulation_steps=args.eval_acc,
            per_device_train_batch_size=args.bs,
            per_device_eval_batch_size=args.eval_bs,
            weight_decay=0.01,
            num_train_epochs=args.epoch,
            metric_for_best_model="eval_test_general_accuracy",
            predict_with_generate=args.use_generate,
            generation_max_length=args.output_len,
            report_to="wandb",
            local_rank=args.local_rank,
            bf16_full_eval=True,
            gradient_checkpointing = args.gradient_checkpointing,
            dataloader_num_workers = args.num_workers,
            dataloader_drop_last = args.drop_last,
            remove_unused_columns = remove_unused_columns,
            save_only_model = False,
            load_best_model_at_end = True
        )

    eval_dataset = {}
    if args.all_eval:
        for i, subset in enumerate(["general", "single", "install", "google_apps", "web_shopping"]):
            eval_dataset["val_"+subset] = eval_set_list[i]
            eval_dataset["test_"+subset] = test_set_list[i]
    elif args.eval_subset_1 and args.eval_subset_2:
        eval_dataset["val_"+args.eval_subset_1] = eval_set_list[0]
        eval_dataset["test_"+args.eval_subset_1] = test_set_list[0]
        eval_dataset["val_"+args.eval_subset_2] = eval_set_list[1]
        eval_dataset["test_"+args.eval_subset_2] = test_set_list[1]
    else:
        eval_dataset["val"] = eval_set
        eval_dataset["test"] = test_set 
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=eval_dataset,
        data_collator=datacollator,
        tokenizer=tokenizer,
        compute_metrics=compute_rouge_and_action_matching_score
    )
    # Add the epoch callback
    trainer.add_callback(CustomTrainerCallback)

    if args.evaluate_dir is None:
        if args.topk_example:
            trainer.train(save_dir)
        else:
            trainer.train(resume_from_checkpoint=args.resume)
        trainer.save_model(save_dir)
        processor.save_pretrained(save_dir)
    else:
        predict_results = trainer.predict(test_dataset=test_set, max_length=args.output_len)
        preds, targets = predict_results.predcitions, predict_results.label_ids
        eval_preds = (preds, targets)

