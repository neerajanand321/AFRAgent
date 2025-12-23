from torch.utils.data import Dataset
import torch
import json
import math
import pickle
from tqdm import tqdm
from PIL import Image
import numpy as np
import jax.numpy as jnp
import random
import re
import os
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

screen_layout_prompt = [
    "Predict the elements and their bounding boxes in this screenshot.",
    "List all elements with their bounding boxes.",
    "Identify the element and its bounding box.",
    "What are the positions (bounding boxes) of the elements?",
    "Analyze the screenshot for element bounding boxes.",
    "List the element and its bounding box.",
    "Identify elements and their bounding boxes.",
    "Determine the bounding boxes of all elements.",
    "Predict the bounding boxes for each element.",
    "What are the bounding boxes of all elements?"
]

# Screen Captioning
screen_caption_prompt = [
    "Can you provide a detailed description of the interface screenshot shown?",
    "Illustrate the details visible in the provided screenshot.",
    "What does the presented screen image depict?",
    "How would you narrate the contents of this screen capture to someone who can't see it?",
    "Please detail the elements shown in the interface screenshot.",
    "Describe the features and information displayed in this screenshot.",
    "Elaborate on what is visible in the screenshot of the interface.",
    "Give a comprehensive description of the screenshot's interface.",
    "What information is conveyed in the screenshot displayed?",
    "Could you depict the content and layout of the screen image provided?",
    "Explain the visual aspects of the screenshot taken from this interface.",
    "How would you verbally depict the interface shown in the screenshot?",
    "What key elements are shown in this interface screenshot?",
    "Provide a verbal representation of the screenshot's content.",
    "Narrate the components and information visible in this interface capture.",
    "What are the main features displayed in the screenshot of this screen?",
    "Outline the specific details shown in the interface image.",
    "How would you describe this screen image to someone who cannot see it?",
    "Enumerate the elements and information present in the provided interface screenshot.",
    "Detail the visual composition of the screen capture you see."
]

# widget captioning
widget_caption_prompt = [
    "Please generate a description for the element at {}.",
    "Describe the function of the element at {} on the screen.",
    "What is the function of the element at {} on the UI?",
    "What happens when you tap position {} on the screen?",
    "What happens when you click point {} on the screen?",
    "Can you explain what the user interface element at {} does?",
    "What action is triggered by interacting with the area at {}?",
    "Explain the purpose of the interactive element found at {}.",
    "What feature is accessed by selecting the location at {}?",
    "Identify and describe the component located at {}.",
    "What is the outcome of selecting the element at {}?",
    "Detail the functionality of the UI element positioned at {}.",
    "What is the significance of the element located at {} in the application?",
    "How does the element at {} contribute to the overall user experience?",
    "What kind of input or interaction is expected at the point marked {}?"
]

def bin_coordinate(coord, bin_range):
    return math.floor(coord * bin_range)

def load_pretraining_data(args, split):
    target_text_aitw = []
    source_text_aitw = []
    source_text_qformer_aitw = []
    source_image_path_aitw = []

    #Llava-Instruct
    target_text_llava = []
    source_text_llava = []
    source_text_qformer_llava = []
    source_image_path_llava = []

    # Screen captioning RICO
    target_text_screen_captions = []
    source_text_screen_captions = []
    source_text_qformer_screen_captions = []
    source_image_path_screen_captions = []

    # Widget captioning RICO
    target_text_widget_captions = []
    source_text_widget_captions = []
    source_text_qformer_widget_captions = []
    source_image_path_widget_captions = []

    # Screen Layout RICO
    target_text_screen_layout = []
    source_text_screen_layout = []
    source_text_qformer_screen_layout = []
    source_image_path_screen_layout = []
    
    if args.use_aitw_layout:
        if args.all_data:
            if split == "train":
                data = []
                for subdir in ["general", "google_apps", "install", "single", "web_shopping"]:
                    print(f"loading {subdir}", len(data))
                    with open(f"{args.data_root}/aitw/{subdir}/{subdir}_{split}.obj", "rb") as rp:
                        sub_data = pickle.load(rp)
                    if subdir == "google_apps":
                        sub_data = random.sample(sub_data, int(len(sub_data) * args.all_data))
                    data.extend(sub_data)
            else:
                data = []
                for subdir in ["general", "google_apps", "install", "single", "web_shopping"]:
                    with open(f"{args.data_root}/aitw/{subdir}/{subdir}_{split}.obj", "rb") as rp:
                        sub_data = pickle.load(rp)
                    data.extend(sub_data)
        else:
            with open(f"{args.data_root}_{split}.obj", "rb") as rp:
                data = pickle.load(rp)
                if args.data_ratio:
                    data = random.sample(data, int(len(data) * args.data_ratio))

        for qid, episode in enumerate(tqdm(data)): 
            episode_id = episode["episode_id"]
            episode_data = episode["data"]
    
            for step_idx, step_data in enumerate(episode_data):
                question = random.choice(screen_layout_prompt)
                question_qformer = question
                image_path = step_data["image_path"]
    
                if not os.path.exists(image_path):
                    continue
    
                ui_positions = step_data["ui_positions"]
                ui_text = step_data["ui_text"]
                ui_type = step_data["ui_type"]
                
                icon_string = ""
                for ui_idx, ui_type_i in enumerate(ui_type):
                    ui_axis = ui_positions[ui_idx]
                    top, left, height, width = ui_axis
                    bottom, right = top + height, left + width
                    golden_location = [round(top + height/2, 4), round(left + width/2, 4)]
                    if args.bin_range:
                        golden_location = [bin_coordinate(axis, args.bin_range) for axis in golden_location]
                    else:
                        golden_location = ["{:.4f}".format(g) for g in golden_location]
                    golden_location = f"[{golden_location[0]}, {golden_location[1]}]"
                    ui_axis = [top, left, bottom, right]
                    ui_axis = [round(axis, 4) for axis in ui_axis]
                    if args.bin_range:
                        ui_axis = [bin_coordinate(axis, args.bin_range) for axis in ui_axis]
                    else:
                        ui_axis = ["{:.4f}".format(axis) for axis in ui_axis]
                    ui_axis = f"({ui_axis[0]}, {ui_axis[1]}, {ui_axis[2]}, {ui_axis[3]})"
                    if ui_type[ui_idx] == "TEXT":
                        icon_string += f"element: '{ui_text[ui_idx]}' bbox: {ui_axis} centroid: {golden_location}\n"
                    elif "ICON" in ui_type[ui_idx]:
                        icon_string += f"element: '{ui_type[ui_idx]}' bbox: {ui_axis} centroid: {golden_location}\n"
                    else:
                        print(icon_string)
                        assert "parsing ui failed!!!"
    
                source_text_aitw.append(question)
                source_text_qformer_aitw.append(question_qformer)
                source_image_path_aitw.append(image_path)
                target_text_aitw.append(icon_string)
         
            if args.debug_num:
                if int(qid) > args.debug_num:
                    break

    if args.use_screen_captioning:
        print("Loading Screen Captioning data")
        if split == "train":
            with open(f'{args.data_root}/rico/screen_captioning_train_set.json', "r") as f:
                screen_captions = json.load(f)
        else:
            with open(f'{args.data_root}/rico/screen_captioning_test_set.json', "r") as f:
                screen_captions = json.load(f)

        for i in tqdm(range(len(screen_captions))):
            if os.path.exists('dataset/rico/combined/'+screen_captions[i]['img_filename']):
                source_image_path_screen_captions.append('dataset/rico/combined/'+screen_captions[i]['img_filename'])
                target_text_screen_captions.append(random.choice(screen_captions[i]['captions']))
                source_text_screen_captions.append(random.choice(screen_caption_prompt))
            
        source_text_qformer_screen_captions = source_text_screen_captions

        source_text_aitw = source_text_aitw + source_text_screen_captions
        source_text_qformer_aitw = source_text_qformer_aitw + source_text_qformer_screen_captions
        source_image_path_aitw = source_image_path_aitw + source_image_path_screen_captions
        target_text_aitw = target_text_aitw + target_text_screen_captions

    if args.use_widget_captioning:
        print("Loading Widget Captioning data")
        if split == "train":
            with open(f'{args.data_root}/rico/widget_captioning_train_set.json', "r") as f:
                widget_captions = json.load(f) 
        else:
            with open(f'{args.data_root}/rico/widget_captioning_test_set.json', "r") as f:
                widget_captions = json.load(f)

        for i in tqdm(range(len(widget_captions))):
            if os.path.exists('dataset/rico/combined/'+widget_captions[i]['img_filename']):
                source_image_path_widget_captions.append('dataset/rico/combined/'+widget_captions[i]['img_filename'])
                target_text_widget_captions.append(widget_captions[i]['instruction'])
                prompt = random.choice(widget_caption_prompt)
                source_text_widget_captions.append(random.choice(prompt.format(list(np.around(np.array(widget_captions[i]['bbox']), 4)))))
            
        source_text_qformer_widget_captions = source_text_widget_captions

        source_text_aitw = source_text_aitw + source_text_widget_captions
        source_text_qformer_aitw = source_text_qformer_aitw + source_text_qformer_widget_captions
        source_image_path_aitw = source_image_path_aitw + source_image_path_widget_captions
        target_text_aitw = target_text_aitw + target_text_widget_captions

    if args.use_screen_layout:
        print("Loading Layout Prediction data")
        if split == "train":
            with open(f'{args.data_root}/rico/screen_layout_train_set.json', "r") as f:
                screen_layout = json.load(f)
        else:
            with open(f'{args.data_root}/rico/screen_layout_test_set.json', "r") as f:
                screen_layout = json.load(f)
        

        for i in tqdm(range(len(screen_layout))):
            if os.path.exists('dataset/rico/combined/'+screen_layout[i]['screen_id']+'.jpg'):
                source_image_path_screen_layout.append('dataset/rico/combined/'+screen_layout[i]['screen_id']+'.jpg')
                screen_element = ''
                for icon in screen_layout[i]['screen_elements']:
                    screen_element += f"label: {icon['label']} bbox: [{icon['xmin']:.4f}, {icon['ymin']:.4f}, {icon['xmax']:.4f}, {icon['ymax']:.4f}]\n"
                target_text_screen_layout.append(screen_element[:-1])
                source_text_screen_layout.append(random.choice(screen_layout_prompt))
            
        source_text_qformer_screen_layout = source_text_screen_layout

        source_text_aitw = source_text_aitw + source_text_screen_layout
        source_text_qformer_aitw = source_text_qformer_aitw + source_text_qformer_screen_layout
        source_image_path_aitw = source_image_path_aitw + source_image_path_screen_layout
        target_text_aitw = target_text_aitw + target_text_screen_layout

    if args.use_llava_instruct:
        print("Loading Llava Instruct data")
        llava_data = []
        if split == "train":
            with open(f'{args.data_root}/Llava-Instruct/llava_instruct_150k_train_set.json', "r") as f:
                temp = json.load(f)
                for data in temp:
                    llava_data.append(data)
        else:
            with open(f'{args.data_root}/Llava-Instruct/llava_instruct_150k_test_set.json', "r") as f:
                temp = json.load(f)
                for data in temp:
                    llava_data.append(data)

        for i in tqdm(range(len(llava_data))):
            img_path = 'dataset/coco/images/train2017/'+llava_data[i]['image']
            for j in range(len(llava_data[i]['conversations'])):
                if llava_data[i]['conversations'][j]['from'] == 'human':
                    source_image_path_llava.append(img_path)
                    if j==0:
                        source_text_llava.append(llava_data[i]['conversations'][j]['value'].replace("<image>\n", "").replace("\n<image>", ""))
                        source_text_qformer_llava.append(llava_data[i]['conversations'][j]['value'].replace("<image>\n", "").replace("\n<image>", ""))
                    else:
                        source_text_llava.append(llava_data[i]['conversations'][j]['value'])
                        source_text_qformer_llava.append(llava_data[i]['conversations'][j]['value'])
                if llava_data[i]['conversations'][j]['from'] == 'gpt':
                    target_text_llava.append(llava_data[i]['conversations'][j]['value'])

        source_text_aitw = source_text_aitw + source_text_llava
        source_text_qformer_aitw = source_text_qformer_aitw + source_text_qformer_llava
        source_image_path_aitw = source_image_path_aitw + source_image_path_llava
        target_text_aitw = target_text_aitw + target_text_llava
                
    return source_text_aitw, source_text_qformer_aitw, source_image_path_aitw, target_text_aitw

class AITWPreTrainingDatasetImg(Dataset):
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
        use_high_res=False
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
        self.source_text = data[0]
        self.source_text_qformer = data[1]
        self.source_image = data[2]
        self.target_text = data[3]
            
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
        target_text_org = str(self.target_text[index])

        # abc = self.tokenizer.tokenize(target_text)
        # print(len(abc))

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
                h, w, channels = source_image.shape
                
                half = h//2

                source_image_top = source_image[:half, :]
                source_image_bottom = source_image[half:, :]
                encoding_1 = self.processor(source_image_top, source_text_qformer, padding="max_length", return_tensors="pt", 
                                            max_length=self.source_len, pad_to_max_length=True, truncation=True)
                pixel_values_1 = encoding_1['pixel_values'].squeeze()
                qformer_input_ids_1 = encoding_1['qformer_input_ids'].squeeze()
                qformer_attention_mask_1 = encoding_1['qformer_attention_mask'].squeeze()
                encoding_2 = self.processor(source_image_bottom, source_text_qformer, padding="max_length", return_tensors="pt", 
                                            max_length=self.source_len, pad_to_max_length=True, truncation=True)
                pixel_values_2 = encoding_2['pixel_values'].squeeze()
                qformer_input_ids_2 = encoding_2['qformer_input_ids'].squeeze()
                qformer_attention_mask_2 = encoding_2['qformer_attention_mask'].squeeze()
    
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
            max_length=self.summ_len,
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
                
        return output