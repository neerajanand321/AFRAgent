import numpy as np 
from tqdm import tqdm 
import cv2 
from PIL import Image
from torch.utils.data import Dataset
import torch
import math
import random
import os
import json
import re
import glob 

def parse_swipe_direction(touch, lift, device_dim):
    x1, y1 = touch
    x2, y2 = lift
    width, height = device_dim

    if abs(x2 - x1) / width > abs(y2 - y1) / height:
        if x2 > x1:
            return "RIGHT"
        else:
            return "LEFT"
    else:
        if y2 > y1:
            return "DOWN"
        else:
            return "UP"

def parse_action(action):
	if action['action'] == 'SWIPE':
		direction = parse_swipe_direction(
			action["touch_coord"], action["lift_coord"], action["device_dim"]
			)
		return f"SWIPE[{direction}]"
	elif action["action"] == "TAP":
		touch_coord = action['touch_coord']
		dimension = action['device_dim']
		x = touch_coord[0]/dimension[0]
		y = touch_coord[1]/dimension[1]
		return f'CLICK[{x:.4f}, {y:.4f}]'
	elif action['action'] == 'TYPE':
		return f'TYPE[{action['type_text']}]'
	elif action['action'] == 'PRESS_BACK':
		return 'PRESS_BACK'
	elif action['action'] == 'PRESS_HOME':
		return 'PRESS_HOME'
	elif action['action'] == 'PRESS_ENTER':
		return 'PRESS_ENTER'
	elif action['action'] == 'TASK_COMPLETE':
		return 'TASK_COMPLETE'
	else:
		return 'TASK_IMPOSSIBLE'


def load_data(args, split):
	# Screen Description
	target_text = []
	source_text = []
	source_text_qformer = []
	source_image_path = []
	element_anno_path = []
	instruction_anno_path = []
	for name in glob.glob(f'{args.data_root}/element_anno/*'):
		element_anno_path.append(name)
	for name in glob.glob(f'{args.data_root}/instruction_anno/*'):
		instruction_anno_path.append(name) 

	if split == 'train':
		if args.use_screen_captioning:
			print("Loading Screen Captioning Data")
			for path in tqdm(element_anno_path):
				with open(path, "r") as f:
					screen_captions = json.load(f)
				if 'page_caption' in screen_captions.keys():
					target_text.extend([screen_captions['page_caption']])
					source_text.extend(['Provide a one-sentence caption for the provided screenshot.'])
					source_text_qformer.extend(['Provide a one-sentence caption for the provided screenshot.'])
					source_image_path.extend([f'{args.data_root}/screenshot/{screen_captions['image_path']}'])

		if args.use_element_grounding:
			print("Loading Element Grounding Data")
			for path in tqdm(element_anno_path):
				with open(path, 'r') as f:
					elements = json.load(f)
				source_text.extend(['Identify all clickable elements on the screen and provide their 2D bounding boxes in the format of [x1,y1,x2,y2].'])
				source_text_qformer.extend(['Identify all clickable elements on the screen and provide their 2D bounding boxes in the format of [x1,y1,x2,y2].'])
				source_image_path.extend([f'{args.data_root}/screenshot/{elements['image_path']}'])
				target_click = ''
				for clickable_element in elements['clickable_elements']:
					if 'type' in clickable_element.keys():
						# match = re.search(r"Element\('(.+?)'\)", clickable_element['type'])
						# if match:
						# 	element_name = f'Element: {match.group(1)}'
						# else:
						# 	element_name = f'Element: {clickable_element['type']}'
						bounding_box = clickable_element['bbox']
						# img = cv2.imread(f'{args.data_root}/screenshot/{elements['image_path']}')
						# img_height, img_width, _ = img.shape
						img = Image.open(f'{args.data_root}/screenshot/{elements['image_path']}')
						img_width, img_height = img.size
						bounding_box = [bounding_box[0]/img_width, bounding_box[1]/img_height, bounding_box[2]/img_width, bounding_box[3]/img_height]
						bounding_box = [round(num, 4) for num in bounding_box]
						target_click = target_click + f'bbox: {bounding_box}\n' 

				if target_click != '':
					target_text.extend([target_click])
				else:
					source_text.pop()
					source_text_qformer.pop()
					source_image_path.pop()
				source_text.extend(['Identify all scrollable areas on the screen and provide their 2D bounding boxes in the format of [x1,y1,x2,y2]'])
				source_text_qformer.extend(['Identify all scrollable areas on the screen and provide their 2D bounding boxes in the format of [x1,y1,x2,y2]'])
				source_image_path.extend([f'{args.data_root}/screenshot/{elements['image_path']}'])
				target_scroll = ''
				for scrollable_element in elements['scrollable_elements']:
					if 'type' in scrollable_element.keys():
						bounding_box = scrollable_element['bbox']
						# img = cv2.imread(f'{args.data_root}/screenshot/{elements['image_path']}')
						# img_height, img_width, _ = img.shape
						img = Image.open(f'{args.data_root}/screenshot/{elements['image_path']}')
						img_width, img_height = img.size
						bounding_box = [bounding_box[0]/img_width, bounding_box[1]/img_height, bounding_box[2]/img_width, bounding_box[3]/img_height]
						bounding_box = [round(num, 4) for num in bounding_box]
						target_scroll = target_scroll + f'bbox: {bounding_box}\n'

				if target_scroll != '':
					target_text.extend([target_scroll])
				else:
					source_text.pop()
					source_text_qformer.pop()
					source_image_path.pop()

		if args.use_element_functionality_desc:
			print("Loading Widget Captioning Data")
			for path in tqdm(element_anno_path):
				with open(path, 'r') as f:
					widget_captions = json.load(f)
				for clickable_element in widget_captions['clickable_elements']:
					if ('functionality' in clickable_element.keys() and len(clickable_element['functionality'])) > 0:
						bounding_box = clickable_element['bbox']
						# img = cv2.imread(f'{args.data_root}/screenshot/{widget_captions['image_path']}')
						# img_height, img_width, _ = img.shape
						img = Image.open(f'{args.data_root}/screenshot/{widget_captions['image_path']}')
						img_width, img_height = img.size
						bounding_box = [bounding_box[0]/img_width, bounding_box[1]/img_height, bounding_box[2]/img_width, bounding_box[3]/img_height]
						bounding_box = [round(num, 4) for num in bounding_box]
						source_text.extend([f'What is the functionality of the element at: {bounding_box}.']) 
						source_text_qformer.extend([f'What is the functionality of the element at: {bounding_box}']) 
						source_image_path.extend([f'{args.data_root}/screenshot/{widget_captions['image_path']}'])
						target_text.extend([f'{clickable_element['functionality']}'])

	if args.use_amex:
		base_prompt = "Given the task instruction, please specify the action to take on the current screen."
		if split == "train":
			print("Loading AMEX Train Set")
			for path in tqdm(instruction_anno_path):
				with open(path, 'r') as f:
					amex_nav = json.load(f)
				instruction = amex_nav['instruction']
				if instruction.startswith('Open'):
					app_name = re.findall(r"Open (.+?)\.", instruction)[0].lower()
					if app_name not in ['citymapper', 'gmail', 'booking', 'microsoft to do', 'yelp', 'signal', 'youtube music', 'shein', 'nbc news']:
						history_action = []
						for step in amex_nav['steps']:
							target_text.extend([parse_action(step)])
							source_image_path.extend([f'{args.data_root}/screenshot/{step['image_path']}'])
							source_text.extend([base_prompt+f'\nTask: {instruction}\nHistory Actions: {','.join(history_action)}'])
							source_text_qformer.extend([base_prompt+f'\nTask: {instruction}\nHistory Actions: {','.join(history_action)}'])

							history_action.append(parse_action(step)) 
				else:
					history_action = []
					for step in amex_nav['steps']:
						target_text.extend([parse_action(step)])
						source_image_path.extend([f'{args.data_root}/screenshot/{step['image_path']}'])
						source_text.extend([base_prompt+f'\nTask: {instruction}\nHistory Actions: {','.join(history_action)}'])
						source_text_qformer.extend([base_prompt+f'\nTask: {instruction}\nHistory Actions: {','.join(history_action)}'])

						history_action.append(parse_action(step)) 

		else:
			print("Loading AMEX Test Set")
			source_text = []
			source_text_qformer = []
			source_image_path = []
			target_text = []
			for path in tqdm(instruction_anno_path):
				with open(path, 'r') as f:
					amex_nav = json.load(f)
				instruction = amex_nav['instruction']
				if instruction.startswith('Open'):
					app_name = re.findall(r"Open (.+?)\.", instruction)[0].lower()
					if app_name in ['citymapper', 'gmail', 'booking', 'microsoft to do', 'yelp', 'signal', 'youtube music', 'shein', 'nbc news']:
						history_action = []
						for step in amex_nav['steps']:
							target_text.extend([parse_action(step)])
							source_image_path.extend([f'{args.data_root}/screenshot/{step['image_path']}'])
							source_text.extend([base_prompt+f'\nTask: {instruction}\nHistory Actions: {','.join(history_action)}'])
							source_text_qformer.extend([base_prompt+f'\nTask: {instruction}\nHistory Actions: {','.join(history_action)}'])
							history_action.append(parse_action(step)) 

	return source_text, source_text_qformer, source_image_path, target_text 

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

class AMEXDatasetImg(Dataset):

	def __init__(
		self, 
		data, 
		split, 
		processor, 
		tokenizer, 
		source_len, 
		target_len, 
		debug, 
		use_high_res=False,
		use_overlap=False,
		num_crops=4
	):
		self.tokenizer = tokenizer 
		self.processor = processor 
		self.source_len = source_len
		self.target_len = target_len
		self.debug = debug 
		self.split = split
		self.use_high_res = use_high_res
		self.use_overlap = use_overlap
		self.num_crops = num_crops
		self.source_text = data[0]
		self.source_text_qformer = data[1]
		self.source_image_path = data[2]
		self.target_text = data[3]

	def __len__(self):
		return len(self.target_text)

	def __getitem__(self, index):
		source_text = str(self.source_text[index])
		source_text_qformer = str(self.source_text_qformer[index])
		source_image_path = str(self.source_image_path[index])
		source_image = cv2.imread(source_image_path)
		target_text_org = str(self.target_text[index])

		# cleaning data so as to ensure data is in string type
		source_text = " ".join(source_text.split())
		source_text_qformer = " ".join(source_text_qformer.split())
		target_text_org = " ".join(target_text_org.split())

		if self.processor is not None:
			encoding_0 = self.processor(
				source_image,
				source_text_qformer,
				padding="max_length",
				return_tensors="pt",
				max_length=self.source_len,
				pad_to_max_length=True,
				truncation=True
				)
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

				encodings = [self.processor(
					source_img,
					source_text_qformer,
					padding="max_length",
					return_tensors="pt",
					max_length=self.source_len,
					pad_to_max_length=True,
					truncation=True) for source_img in source_image_patches]
				pixel_values_crops = [encoding['pixel_values'].squeeze() for encoding in encodings]
				pixel_values_1, pixel_values_2, pixel_values_3, pixel_values_4 = pixel_values_crops
				qformer_input_ids_1 = qformer_input_ids_2 = qformer_input_ids_3 = qformer_input_ids_4 = qformer_input_ids_0
				qformer_attention_mask_1 = qformer_attention_mask_2 = qformer_attention_mask_3 = qformer_attention_mask_4 = qformer_attention_mask_0


		source = self.tokenizer.batch_encode_plus(
			[source_text],
			max_length=self.source_len,
			pad_to_max_length=True,
			truncation=True,
			padding="max_length",
			return_tensors="pt",
			)
		target = self.tokenizer.batch_encode_plus(
			[target_text_org],
			max_length=self.target_len,
			pad_to_max_length=True,
			truncation=True,
			padding="max_length",
			return_tensors="pt",
			)

		input_ids = source["input_ids"].squeeze()
		attention_mask = source["attention_mask"].squeeze()
		target_ids = target["input_ids"].squeeze()

		output = {}
		if self.processor is None:
			output["input_ids"] = input_ids
			output["attention_mask"] = attention_mask
			output["image_ids"] = source_image
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

		return output







