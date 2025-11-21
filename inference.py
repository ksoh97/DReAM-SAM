import os
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader

from segment_anything import sam_model_registry
from utils.LoRA import LoRA_Sam
from utils.dataloader import Processor, collate_fn, DatasetSAM
from utils.utils import seed_everything

""" Configuration """
gpu_id = '-1'

file_name = 'your_run_file_name'
best_model_num = 'best_epoch'

model_type = 'DReAM-SAM'    # DReAM-SAM, vit_b
model_name = 'LoRA'         # SAM, LoRA
data_type = 'your_dataset'
cls = int('num_classes')    # Change depending on the 'data_type'
num_regions = int('num_classes'+1)

prompt_config = {
    'Point': False,
    'Box': False,
    'Mask': False,
}

log_dir = "/path/to/the/saved/weights"
save_dir = "/path/to/save/outputs"
data_dir = "/path/to/the/datasets"
dataset_split_txt_dir = "/path/to/the/split/text/file"
""" Configuration """


seed_everything(seed=2024)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

""" Model load """
if model_type == 'vit_b':
    sam = sam_model_registry[model_type](checkpoint=None)
elif model_type == 'DReAM-SAM':    
    sam = sam_model_registry[model_type](cls=cls, num_regions=num_regions, checkpoint=None)
    sam.prompt_generator.point_embeddings = sam.prompt_encoder.point_embeddings
    sam.prompt_generator.not_a_point_embed = sam.prompt_encoder.not_a_point_embed

if model_name == 'SAM':
    model = sam
elif model_name == 'LoRA':
    model = LoRA_Sam(sam, r=4)

model_dir = os.path.join(log_dir, data_type, file_name)
checkpoint_path = os.path.join(model_dir, 'Models', f"best_model_{best_model_num}.pth")
pretrained_state_dict = torch.load(checkpoint_path, weights_only=False)['model_state_dict']
model.load_state_dict(pretrained_state_dict)

model.to(device=device)
""" End of model load """


""" Data list load and preprocessing """
data_path = data_dir
with open(os.path.join(dataset_split_txt_dir, 'train.txt'), 'r') as f:
    trn_list = [line.strip() for line in f if line.strip()]
with open(os.path.join(dataset_split_txt_dir, 'valid.txt'), 'r') as f:
    val_list = [line.strip() for line in f if line.strip()]

if model_name == 'SAM': processor = Processor(model.image_encoder.img_size)
else: processor = Processor(model.sam.image_encoder.img_size)

pred_save_path = os.path.join(save_dir, data_type, file_name, 'tst_pred')
if not os.path.exists(pred_save_path): os.makedirs(pred_save_path, exist_ok=True)
""" Data list load and preprocessing """


model.eval()
with torch.no_grad():
    tst_dataset = DatasetSAM(val_list, data_path, processor, prompt_config)
    tst_dataloader = DataLoader(tst_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    
    Tot_slice_dice = []
    for batch in tqdm(tst_dataloader):
        if model_type == 'vit_b':
            outputs = model(batched_input=batch, multimask_output=False)
        
        elif model_type == 'DReAM-SAM':
            outputs, pred_cls, _ = model(batched_input=batch, multimask_output=False)    

        out = outputs[0]['masks']
        out = torch.sigmoid(out)
        out = (out > 0.5).float()        

        np.save(os.path.join(pred_save_path, batch[0]['file_name']+".nii.gz"), out.detach().cpu().squeeze().numpy())
