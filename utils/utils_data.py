from torch.utils.data import Dataset
import torch
import math 
import pickle
from tqdm import tqdm
from PIL import Image
from . import action_type
import numpy as np
import jax.numpy as jnp
import random
import re
import cv2
img_shape = {
    "resnet": (512, 2048),
    "clip": (49, 2048),
    "detr": (100, 256),
    "vit": (577, 768),
    "vit-large": (145, 1024),
    "vit-global": (1, 768),
    "vit-merge": (578, 768),
}

def bin_coordinate(coord, bin_range):
    return math.floor(coord * bin_range)

def _load_data(data, args):
  target_text = []
  source_text = []
  source_text_qformer = []
  source_image_path = []
  anno_positions = []
  episodes_id = []
  image_id = [] # used for top k hard samples selection for each batch 
  count = 0

  for qid, episode in enumerate(tqdm(data)):
      episode_id = episode["episode_id"]
      episode_data = episode["data"]
      if args.use_history:
          history_action = []
          if args.use_img_history:
              history_image = [torch.zeros(args.img_dim)] * args.use_history
      if args.use_qformer_history:
          history_action_qformer = []

      for step_idx, step_data in enumerate(episode_data):
          question = step_data["goal"]
          question = f"Goal: {question}"
          question_qformer = question

          image_path = step_data["image_path"]

          ui_positions = step_data["ui_positions"]
          ui_text = step_data["ui_text"]
          ui_type = step_data["ui_type"]

          if args.use_layout:
              icon_string = ""
              for ui_idx, ui_type_i in enumerate(ui_type):
                  ui_axis = ui_positions[ui_idx]
                  top, left, height, width = ui_axis
                  if args.use_coco_agent_layout:
                      golden_location = [top + height/2, left + width/2]
                      golden_location = ["{:.4f}".format(g) for g in golden_location]
                      golden_location = f"[{golden_location[0]}, {golden_location[1]}]"
                      if ui_type[ui_idx] == "TEXT":
                          icon_string += f"{ui_text[ui_idx]} location: {golden_location}\n"
                      elif "ICON" in ui_type[ui_idx]:
                          icon_string += f"{ui_type[ui_idx]} location: {golden_location}\n"
                      else:
                          assert "parsing ui failed!!!"
                  else:
                      # The y-axis is inverted for AndroidEnv, so bottom = top + height.
                      bottom, right = top + height, left + width
                      ui_axis = [top, left, bottom, right]
                      ui_axis = ["{:.4f}".format(axis) for axis in ui_axis]
                      ui_axis = f"({ui_axis[0]}, {ui_axis[1]}, {ui_axis[2]}, {ui_axis[3]})"
                      if ui_type_i == "TEXT":
                          icon_string += f'<p id={ui_idx} class="text" alt="{ui_axis}">{ui_text[ui_idx]}</p>\n'
                          # icon_string += f'{ui_text[ui_idx]} location: "{ui_axis}"\n'
                      elif "ICON" in ui_type_i:
                          icon_string += f'<img id={ui_idx} class={ui_type_i} alt="{ui_axis}">{ui_text[ui_idx]}</p>\n'
                          # icon_string += f'{ui_type_i} location: "{ui_axis}"\n'
                      else:
                          print(icon_string)
                          assert "parsing ui failed!!!"
              
              question = f"{question}\nScreen: {icon_string}"
              # print(question)
          result_touch_yx = step_data["result_touch_yx"]
          result_lift_yx = step_data["result_lift_yx"]
          result_action = step_data["result_action"][0]
          result_text = step_data["result_action"][1]

          result_text = result_text.replace("\\", "").replace('"','').replace("'","")

          #Ignore example with null text input
          if not args.check:
              if result_action == "TYPE" and len(result_text) == 0:
                  continue

          if args.transform_axis:
              scroll_map = {
                  "up": [[0.8000, 0.5000], [0.2000, 0.5000]],
                  "down": [[0.2000, 0.5000], [0.8000, 0.5000]],
                  "left": [[0.5000, 0.8000], [0.5000, 0.2000]],
                  "right": [[0.5000, 0.2000], [0.5000, 0.8000]]
              }
              action_touch_yx = jnp.asarray(result_touch_yx)
              action_lift_yx = jnp.asarray(result_lift_yx)
              if result_action == "DUAL_POINT":
                  if is_tap_action(action_touch_yx, action_lift_yx):
                      result_touch_yx = [round(axis, 4) for axis in result_touch_yx]
                      if args.bin_range:
                          result_touch_yx = [bin_coordinate(axis, args.bin_range) for axis in result_touch_yx]
                      # if touching, the lift can be the same as touch
                      result_lift_yx = result_touch_yx
                  else:
                      drags_match = _check_drag_actions_match(
                          action_touch_yx, action_lift_yx
                      )
                      result_touch_yx, result_lift_yx = scroll_map[drags_match]
                      if args.bin_range:
                          result_touch_yx = [bin_coordinate(axis, args.bin_range) for axis in result_touch_yx]
                          result_lift_yx = [bin_coordinate(axis, args.bin_range) for axis in result_lift_yx]

          if args.verbalized_output:
              if result_action == "DUAL_POINT":
                  if is_tap_action(action_touch_yx, action_lift_yx):
                      target_action = f'tap at {result_touch_yx}'
                  else:
                      # target_action = f'swipe from {result_touch_yx} to {result_lift_yx}'
                      target_action = f'swipe {drags_match}'
              elif result_action == "TYPE":
                  target_action = f'Input text "{result_text}"'
                  
              elif result_action == "PRESS_HOME":
                  target_action = 'press home'
  
              elif result_action == "PRESS_BACK":
                  target_action = 'press back'
  
              elif result_action == "PRESS_ENTER":
                  target_action = 'press enter'
  
              elif result_action == "STATUS_TASK_COMPLETE":
                  target_action = 'complete'
  
              else:
                  target_action = 'impossible'

          else:
              target_action = f'"action_type": "{result_action}", "touch_point": "{result_touch_yx}", "lift_point": "{result_lift_yx}", "typed_text": "{result_text}"'
                  
          
          if args.use_history:
              # reversed_history_action = history_action[::-1]
              prev_actions = "\n".join(history_action)
              question = f"Previous Actions: {prev_actions}\n{question}"
              # question = f"{question}\nPrevious Actions: {prev_actions}" # Goal appears first in source text
              if args.use_img_history:
                  image = history_image + [image]
                  image = torch.stack(image)

          if args.use_qformer_history:
              prev_actions_qformer = "\n".join(history_action_qformer)
              question_qformer = f"Previous Actions: {prev_actions_qformer}\n{question_qformer}"

          verbalized_map = {
              'DUAL_POINT_0': 'tap',
              'DUAL_POINT_1': 'swipe',
              'TYPE': 'Input text',
              'PRESS_HOME': 'press home',
              'PRESS_BACK': 'press back',
              'PRESS_ENTER': 'press enter',
              'STATUS_TASK_COMPLETE': 'complete',
              'STATUS_TASK_IMPOSSIBLE': 'impossible'
          }
          if args.use_future:
              future_actions = episode_data[step_idx:]
              if len(future_actions) > args.use_future:
                  future_actions = future_actions[:args.use_future]
              if args.verbalized_output:
                  future_actions_temp = []
                  for action_t in future_actions:
                      if action_t['result_action'][0] == "DUAL_POINT":
                          if is_tap_action(action_t['result_touch_yx'], action_t['result_lift_yx']):
                              future_actions_temp.append('DUAL_POINT_0')
                          else:
                              future_actions_temp.append('DUAL_POINT_1')
                      else:
                          future_actions_temp.append(action_t['result_action'][0])

                  future_actions = "[" + ", ".join([verbalized_map[action_t] for action_t in future_actions_temp]) + "]\n"
                              
              else:
                  future_actions = "[" + ",".join([action_t["result_action"][0] for action_t in future_actions]) + "]\n"
              target_action_label = "Action Plan: " + future_actions + "; Action Decision: " + target_action
          else:
              target_action_label = target_action

          source_text.append(question)
          source_text_qformer.append(question_qformer)
          source_image_path.append(image_path)
          image_id.append(count)
          target_text.append(target_action_label)
          anno_positions.append(ui_positions)
          episodes_id.append(episode_id)

          count += 1

          if args.use_history:
              history_action.append(target_action)
              if args.use_img_history:
                  history_image.append(image[-1])
                  history_image.pop(0)
              if len(history_action) > args.use_history:
                  history_action.pop(0)

          if args.use_qformer_history:
              history_action_qformer.append(target_action)
              if len(history_action_qformer) > args.use_qformer_history:
                  history_action_qformer.pop(0)
    
      if args.debug_num:
          if int(qid) > args.debug_num:
              break
          
  return source_text, source_text_qformer, source_image_path, target_text, anno_positions, episodes_id, image_id 


def load_data(args, split):  
  if args.all_data:
    if split == "train":
      data = []
      for subdir in ["general", "single", "install", "google_apps", "web_shopping"]:
          print(f"loading {subdir}", len(data))
          with open(f"{args.data_root}/aitw/{subdir}/{subdir}_{split}.obj", "rb") as rp:
              sub_data = pickle.load(rp)
          if subdir == "google_apps":
              sub_data = random.sample(sub_data, int(len(sub_data) * args.all_data))
          data.extend(sub_data)
      if args.oversample_single:
        with open(f"{args.data_root}/aitw/single/single_train.obj", "rb") as rp:
            sub_data = pickle.load(rp)
        data.extend(sub_data)
      return _load_data(data, args)
    else:
      if args.all_eval:
        data = {}
        for subdir in ["general", "single", "install", "google_apps", "web_shopping"]:
          with open(f"{args.data_root}/aitw/{subdir}/{subdir}_{split}.obj", "rb") as rp:
            sub_data = pickle.load(rp)
          data[subdir] = _load_data(sub_data, args) 
        return data 
      elif args.eval_subset_1 and args.eval_subset_2:
        data = {}
        for subdir in [args.eval_subset_1, args.eval_subset_2]:
          with open(f"{args.data_root}/aitw/{subdir}/{subdir}_{split}.obj", "rb") as rp:
            sub_data = pickle.load(rp)
          data[subdir] = _load_data(sub_data, args)
        return data
      else:
        with open(f"{args.data_root}/aitw/{args.eval_subset}/{subdir}_{split}.obj", "rb") as rp:
            data = pickle.load(rp)
        return _load_data(data , args)
  else:
    with open(f"{args.data_root}_{split}.obj", "rb") as rp:
        data = pickle.load(rp)
        if args.data_ratio:
            data = random.sample(data, int(len(data) * args.data_ratio))
    return _load_data(data, args)

_SWIPE_DISTANCE_THRESHOLD = 0.04
def is_tap_action(normalized_start_yx, normalized_end_yx):
    distance = jnp.linalg.norm(
        jnp.array(normalized_start_yx) - jnp.array(normalized_end_yx))
    return distance <= _SWIPE_DISTANCE_THRESHOLD

def _check_drag_actions_match(
    drag_touch_yx,
    drag_lift_yx,
):
    """Determines if two drag actions are the same."""
    # Store drag deltas (the change in the y and x coordinates from touch to
    # lift), magnitudes, and the index of the main axis, which is the axis with
    # the greatest change in coordinate value (e.g. a drag starting at (0, 0) and
    # ending at (0.3, 0.5) has a main axis index of 1).
    drag_1_deltas = drag_lift_yx - drag_touch_yx
    drag_1_magnitudes = jnp.abs(drag_1_deltas)
    drag_1_main_axis = np.argmax(drag_1_magnitudes)

    # y axis
    if drag_1_main_axis == 0:
        if drag_1_deltas[0] < 0:
            scroll = "up"
        else:
            scroll = "down"
    elif drag_1_main_axis == 1:
        if drag_1_deltas[1] < 0:
            scroll = "left"
        else:
            scroll = "right"
            
    return scroll

def calculate_fixed_crop_ranges(image_height, num_crops, step_ratio=0.31, overlap_ratio=0.08):
    step = int(image_height * step_ratio)
    overlap = int(image_height * overlap_ratio)

    ranges = []
    start = 0
    for i in range(num_crops):
        end = start + step
        ranges.append((start, end))
        start = end - overlap

    # Ensure the last crop reaches the bottom of the image
    ranges[-1] = (ranges[-1][0], image_height)
    
    return ranges

def split_image_fixed_crops(source_image, num_crops=4, step_ratio=0.31, overlap_ratio=0.08):
    h, w, channels = source_image.shape
    crop_ranges = calculate_fixed_crop_ranges(h, num_crops, step_ratio, overlap_ratio)
    crops = [source_image[start:end, :] for start, end in crop_ranges]
    return crops

class AITWDatasetImg(Dataset):
    """
    Creating a custom dataset for reading the dataset and
    loading it into the dataloader to pass it to the
    neural network for finetuning the model

    """

    def __init__(
        self, 
        data, 
        split,
        processor, 
        tokenizer, 
        source_len, 
        target_len, 
        debug, 
        require_source_image_id=False, 
        use_high_res=False,
        use_overlap=False,
        num_crops=4
    ):
        """
        Initializes a Dataset class

        Args:
            dataframe (pandas.DataFrame): Input dataframe
            tokenizer (transformers.tokenizer): Transformers tokenizer
            source_len (int): Max length of source text
            target_len (int): Max length of target text
            source_text (str): column name of source text
            target_text (str): column name of target text
        """
        self.tokenizer = tokenizer
        self.processor = processor
        self.source_len = source_len
        self.summ_len = target_len
        self.debug = debug
        self.split = split
        self.require_source_image_id = require_source_image_id
        self.use_high_res = use_high_res
        self.use_overlap = use_overlap
        self.num_crops = num_crops
        self.source_text = data[0]
        self.source_text_qformer = data[1]
        self.source_image = data[2]
        self.target_text = data[3]
        self.anno_positions = data[4]
        self.episodes_id = data[5]
        self.image_id = data[6]
        
            
    def __len__(self):
        """returns the length of dataframe"""
        return len(self.target_text)

    def __getitem__(self, index):
        """return the input ids, attention masks and target ids"""
        #set hardness to True only if len(self.hard_examplesPop) >= batch_size
        #overwrite index with hard example if the global variable is hard
        source_text = str(self.source_text[index])
        source_text_qformer = str(self.source_text_qformer[index])
        source_image_path = self.source_image[index]
        source_image = cv2.imread(source_image_path)
        source_image_id = self.image_id[index]
        target_text_org = str(self.target_text[index])

        # cleaning data so as to ensure data is in string type
        source_text = " ".join(source_text.split())
        source_text_qformer = " ".join(source_text_qformer.split())
        target_text_org = " ".join(target_text_org.split())

        if self.debug:
            print("Source text =", source_text)
            print("Source text qformer=", source_text_qformer)
            print("Source image type =", type(source_image))
            print("Target text =", target_text_org)
        
        if self.processor is not None:
            encoding_0 = self.processor(source_image, source_text_qformer, padding="max_length", return_tensors="pt", 
                                      max_length=self.source_len, pad_to_max_length=True, truncation=True)
            pixel_values_0 = encoding_0['pixel_values'].squeeze()
            qformer_input_ids_0 = encoding_0['qformer_input_ids'].squeeze()
            qformer_attention_mask_0 = encoding_0['qformer_attention_mask'].squeeze()

            if self.use_high_res:
                if self.use_overlap:
                    crops = split_image_fixed_crops(source_image)
                    source_image_patches = crops
                else:
                    h, w, channels = source_image.shape
                    quarter = h//self.num_crops
                    source_image_patches = [source_image[i*quarter:(i+1)*quarter, :] for i in range(self.num_crops-1)]
                    source_image_patches.append(source_image[(self.num_crops-1)*quarter:, :])

                encodings = [self.processor(source_img, source_text_qformer, padding="max_length", 
                                            return_tensors="pt", max_length=self.source_len,
                                            pad_to_max_length=True, truncation=True) for source_img in source_image_patches]
                pixel_values_crops = [encoding['pixel_values'].squeeze() for encoding in encodings]

                pixel_values_1, pixel_values_2, pixel_values_3, pixel_values_4 = pixel_values_crops
                qformer_input_ids_1 = qformer_input_ids_2 = qformer_input_ids_3 = qformer_input_ids_4 = qformer_input_ids_0
                qformer_attention_mask_1 = qformer_attention_mask_2 = qformer_attention_mask_3 = qformer_attention_mask_4 = qformer_attention_mask_0
    
        source = self.tokenizer.batch_encode_plus(
            [source_text],
            max_length=1024, #only for use_layout training else self.source_len
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        target = self.tokenizer.batch_encode_plus(
            [target_text_org],
            max_length=self.summ_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = source["input_ids"].squeeze()
        attention_mask = source["attention_mask"].squeeze()
        target_ids = target["input_ids"].squeeze()
        
        source_image_id = torch.tensor(source_image_id).squeeze()
        
        output = {}
        if self.processor is None:
            output["input_ids"] = input_ids
            output["attention_mask"] = attention_mask
            output["image_ids"] = source_image
            output["source_image_ids"] = source_image_id
            output["labels"] = target_ids
        else:
            output["input_ids"] = input_ids
            output["attention_mask"] = attention_mask
            output["labels"] = target_ids
            if not self.use_high_res:
                output["pixel_values"] = pixel_values_0
                output["qformer_input_ids"] = qformer_input_ids_0
                output["qformer_attention_mask"] = qformer_attention_mask_0
            else:
                output["pixel_values_0"] = pixel_values_0
                output["qformer_input_ids_0"] = qformer_input_ids_0
                output["qformer_attention_mask_0"] = qformer_attention_mask_0
                output["pixel_values_1"] = pixel_values_1
                output["qformer_input_ids_1"] = qformer_input_ids_1
                output["qformer_attention_mask_1"] = qformer_attention_mask_1
                output["pixel_values_2"] = pixel_values_2
                output["qformer_input_ids_2"] = qformer_input_ids_2
                output["qformer_attention_mask_2"] = qformer_attention_mask_2
                output["pixel_values_3"] = pixel_values_3
                output["qformer_input_ids_3"] = qformer_input_ids_3
                output["qformer_attention_mask_3"] = qformer_attention_mask_3
                output["pixel_values_4"] = pixel_values_4
                output["qformer_input_ids_4"] = qformer_input_ids_4
                output["qformer_attention_mask_4"] = qformer_attention_mask_4
            if self.require_source_image_id:
                output["source_image_ids"] = source_image_id

        return output
