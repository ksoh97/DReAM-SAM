import numpy as np
import torch
import os
import cv2
from scipy.ndimage import label
from scipy import ndimage
from torch.utils.data import Dataset
from PIL import Image

from utils.transforms import ResizeLongestSide, ResizeSqure


class Processor:
    def __init__(self, model_input_size):
        super().__init__
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = ResizeSqure(model_input_size)
        self.reset_image()
    
    def __call__(self, image: np.array, label: np.array, prompt: list) -> dict:
        """
        Return: 
            inputs = {
                "image": image_torch,
                "gt_mask": label_torch,
                "original_size": self.original_size,
                "point_coords": prompt_torch["Point"],
                "point_labels": prompt_torch["Point_label"],
                "boxes": prompt_torch["Box"],
                "mask_inputs": prompt_torch["Mask"],
                "origin_prompt" : prompt,
            }
        """
        image_torch, label_torch = self.process_image(image, label)
        prompt_torch = self.process_prompt(prompt)

        inputs = {"image": image_torch,
                  "gt_mask": label_torch,
                  "original_size": self.original_size,
                  "origin_prompt" : prompt}
        
        if prompt_torch.get("Point") is not None:
            inputs["point_coords"] = prompt_torch["Point"]
            inputs["point_labels"] = prompt_torch["Point_label"]
        
        if prompt_torch.get("Box") is not None:
            inputs["boxes"] = prompt_torch["Box"]
        
        if prompt_torch.get("Mask") is not None:
            inputs["mask_inputs"] = prompt_torch["Mask"]
        
        return inputs
    
    def process_image(self, image: np.array, label: np.array) -> torch.Tensor:
        input_image = np.transpose(self.transform.apply_image(np.transpose(image, axes=(1,2,0))), axes=(2,0,1))
        input_image_torch = torch.as_tensor(input_image, dtype=torch.float, device=self.device)
        
        pil_img = Image.fromarray(label.squeeze().astype('uint8'), mode='L')
        pil_img = pil_img.resize((1024, 1024))
        input_label = np.expand_dims(np.array(pil_img), axis=0)
        input_label_torch = torch.as_tensor(input_label, dtype=torch.float, device=self.device)

        self.original_size = image.shape[-2:]
        self.input_size = input_image.shape[-2:]

        return input_image_torch, input_label_torch

    def process_prompt(self, prompt) -> torch.tensor:
        point, box, mask = prompt['Point'], prompt['Box'], prompt['Mask']
        coords_torch, labels_torch, box_torch, mask_input_torch = None, None, None, None

        if point is not None:
            point_coords = np.array(point)
            point_coords = self.transform.apply_coords(point_coords, self.original_size)
            coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=self.device)
            labels_torch = torch.as_tensor(prompt['Point_label'], dtype=torch.int, device=self.device)
            coords_torch, labels_torch = coords_torch[:, None, :], labels_torch[:, None]
        
        if box is not None:
            box = np.array(box)
            box = self.transform.apply_boxes(box, self.original_size)
            box_torch = torch.as_tensor(box, dtype=torch.float, device=self.device)
            box_torch = box_torch[:, None, :]
        
        if mask is not None:
            mask = ndimage.zoom(mask, (1, 256 / mask.shape[1], 256 / mask.shape[2]), order=0)
            mask_input_torch = torch.as_tensor(mask.astype(np.float32), dtype=torch.float, device=self.device)
            mask_input_torch = mask_input_torch.unsqueeze(1)

        prompt_torch = {
            'Point': coords_torch,
            'Point_label': labels_torch,
            'Box': box_torch,
            'Mask': mask_input_torch,
        }
        return prompt_torch
    
    def reset_image(self) -> None:
        self.is_image_set = False
        self.features = None
        self.orig_h = None
        self.orig_w = None
        self.input_h = None
        self.input_w = None
    

class DatasetSAM(Dataset):
    def __init__(self, data_list, data_path, processor, configs):
        self.data_list = data_list
        self.data_path = data_path
        
        self.processor = processor
        self.configs = configs

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data_id = self.data_list[idx]

        img_path = os.path.join(self.data_path, data_id+".png")
        lbl_path = os.path.join(self.data_path.replace('img', 'lbl'), data_id+".png")
        img_slice = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img_slice = np.repeat(img_slice[None, :, :], 3, axis=0)
        
        lbl_slice = cv2.imread(lbl_path, cv2.IMREAD_GRAYSCALE)
        _, lbl_slice = cv2.threshold(lbl_slice, 128, 1, cv2.THRESH_BINARY)
        lbl_slice = np.expand_dims(lbl_slice, axis=0)

        slice_input = self.processor(
            image=img_slice,
            label=lbl_slice,
            prompt=self.get_prompt(lbl_slice, self.configs)
        )
        slice_input['orig_img'] = img_slice
        slice_input['orig_lbl'] = lbl_slice
        slice_input['file_name'] = data_id
        return slice_input

    def get_prompt(self, label, configs):
        point_coords, point_labels, box, mask = None, None, None, None
        if (label.sum() > 0) or (label.sum() == 0):
            if configs["Point"]:
                point_coords, point_labels = self.get_point(np.squeeze(label))

            if configs["Box"]:
                box = self.get_bounding_box(np.squeeze(label))
        
            if configs["Mask"]:
                mask = label

        prompts = {
            'Point': point_coords,
            'Point_label': point_labels,
            'Box': box,
            'Mask': mask,
        }
        return prompts
    
    def get_point(self, mask: np.array) -> tuple:
        instance_points, instance_labels  = [], []
        unique_classes = [c for c in np.unique(mask) if c != 0]
        
        if len(unique_classes) > 0:
            for c in unique_classes:
                class_mask = (mask == c).astype(np.uint8)
                labeled_mask, num_features = label(class_mask)
                
                for i in range(1, num_features+1):
                    instance_mask = (labeled_mask == i)
                    coords = np.argwhere(instance_mask)
                    if coords.size > 0:
                        random_idx = np.random.choice(coords.shape[0])
                        point = tuple(coords[random_idx][::-1])
                        instance_points.append(point)
                        instance_labels.append(1)
            
        else:   # No foreground — sample random background point
            random_point = (np.random.randint(0, mask.shape[1]), np.random.randint(0, mask.shape[0]))
            instance_points.append(random_point)
            instance_labels.append(0)   

        return instance_points, instance_labels
    
    def get_bounding_box(self, mask: np.array) -> list:
        labeled_mask, num_features = label(mask)
        instance_bboxes = []

        for i in range(1, num_features+1):
            instance_mask = (labeled_mask == i)

            rows, cols = np.any(instance_mask, axis=0), np.any(instance_mask, axis=1)
            min_row, max_row = np.argmax(rows), len(rows) - 1 - np.argmax(rows[::-1])
            min_col, max_col = np.argmax(cols), len(cols) - 1 - np.argmax(cols[::-1])

            instance_bboxes.append(np.array([min_col, min_row, max_col, max_row]))
        return instance_bboxes


def collate_fn(batch):
    return batch
