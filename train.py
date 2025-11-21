import os
import torch
from tqdm import tqdm
import yaml
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torch.nn as nn

from segment_anything import sam_model_registry
from utils.LoRA import LoRA_Sam
from utils.utils import seed_everything, get_class_vector, soft_dice
from utils.dataloader import Processor, collate_fn, DatasetSAM
from utils.losses import DiceCELoss, RegionDecouplingLoss


with open('./config.yaml') as f:
    configs = yaml.load(f, Loader=yaml.FullLoader)

setup_conf = configs.pop("setup")
storage_conf = configs.pop("storage")
optim_conf = configs.pop("optimizer")
net_conf = configs.pop("network")

mode = setup_conf['mode']
seed_everything(seed=setup_conf["seed"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


""" Model load """
sam_checkpoint = os.path.join(storage_conf['preweights_dir'], setup_conf['weight_type'])
if setup_conf['model_type'] == 'DReAM-SAM':
    sam = sam_model_registry[setup_conf['model_type']](cls=setup_conf["cls"], num_regions=setup_conf["num_regions"], 
                                                       checkpoint=sam_checkpoint)
else:
    sam = sam_model_registry[setup_conf['model_type']](checkpoint=sam_checkpoint)
    
for module_name in ['image_encoder', 'prompt_encoder', 'mask_decoder']:
    update_flag = net_conf["Update"].get(module_name, False)    
    module = getattr(sam, module_name)
    for param in module.parameters():
        param.requires_grad = update_flag

if setup_conf['model_type'] == 'DReAM-SAM':
    sam.prompt_generator.point_embeddings = sam.prompt_encoder.point_embeddings
    sam.prompt_generator.not_a_point_embed = sam.prompt_encoder.not_a_point_embed

if setup_conf["model_name"] == 'SAM':
    model = sam
elif setup_conf["model_name"] == 'LoRA':
    model = LoRA_Sam(sam, r=4)

model.to(device=device)
""" End of model load """


""" Hyper-paramter setting """
batch_size, lr, lr_decay = int(optim_conf["batch_size"]), float(optim_conf["learning_rate"]), float(optim_conf["weight_decay"])
if optim_conf['optim_set'] == "AdamW":
    param_dicts = [{"params": [p for n, p in model.named_parameters() if p.requires_grad], "lr_scale": 1}]
    opt_params = {'lr': lr, 'weight_decay': lr_decay}
    for k in ['momentum', 'weight_decay']:
        if k in optim_conf:
            opt_params[k] = float(optim_conf[k])

    optimizer = eval(optim_conf['AdamW']['target'])(param_dicts, **opt_params)
    if optim_conf['AdamW']["lr_scheduler"] == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 - 50) / float(optim_conf["max_epoch"] - 50 + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)

    elif optim_conf['AdamW']["lr_scheduler"] == 'steplr':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=int(optim_conf["max_epoch"]/2), gamma=0.1)
else: raise ValueError("Invalid optimizer setting!!")
""" Hyper-paramter setting """


""" Loss function definition """
criterion = DiceCELoss()
decouple_loss = RegionDecouplingLoss(num_regions=setup_conf['num_regions'], 
                                     gamma=10.0, eps=1e-3, r_d=2, blur_sigma=1.5, from_logits=False)
classification_loss = nn.BCELoss()
""" Loss function definition """


""" Data list load and preprocessing """
data_path = storage_conf["data_dir"]
txt_save_path = storage_conf["dataset_split_txt_dir"]
with open(os.path.join(txt_save_path, 'train.txt'), 'r') as f:
    trn_list = [line.strip() for line in f if line.strip()]
with open(os.path.join(txt_save_path, 'valid.txt'), 'r') as f:
    val_list = [line.strip() for line in f if line.strip()]

if setup_conf['model_name'] == 'SAM': processor = Processor(model.image_encoder.img_size)
else: processor = Processor(model.sam.image_encoder.img_size)
""" Data list load and preprocessing """


""" Result Saving Preparation """
file_name = setup_conf['file_name']
save_path = os.path.join(storage_conf["log_dir"], setup_conf["data_type"], file_name)
model_save_path = os.path.join(save_path, "Models")
if not os.path.exists(model_save_path): os.makedirs(model_save_path)
""" End of Saving Setup """


""" Training """
best_val_dice = 0.0
for cur_epoch in tqdm(range(optim_conf["max_epoch"])):
    one_epoch_loss, one_epoch_trn_dice, local_count = 0.0, 0.0, 0.0

    train_dataset = DatasetSAM(trn_list, data_path, processor, net_conf["Prompt"])
    train_dataloader = DataLoader(train_dataset, batch_size=optim_conf["batch_size"], shuffle=True, collate_fn=collate_fn)

    for batch in tqdm(train_dataloader, desc=f"Current Epoch: {cur_epoch + 1} - Training"):
        """ Network Forward """
        model.train()
        criterion.train()
        decouple_loss.train()
        optimizer.zero_grad()
        
        if setup_conf['model_type'] == 'vit_b':
            outputs = model(batched_input=batch, multimask_output=False)
            
            stk_gt = torch.stack([b["gt_mask"].squeeze(0) for b in batch], dim=0).long()
            stk_out = torch.stack([out["low_res_logits"].squeeze(1) for out in outputs], dim=0)
            loss, dice_loss, ce_loss = criterion(stk_out, stk_gt)
            
        elif setup_conf['model_type'] == 'DReAM-SAM':
            outputs, pred_cls, decoupled_res = model(batched_input=batch, multimask_output=False)

            stk_gt = torch.stack([b["gt_mask"].squeeze(0) for b in batch], dim=0).long()
            stk_out = torch.stack([out["low_res_logits"].squeeze(1) for out in outputs], dim=0)
            # Segmentation loss
            loss, dice_loss, ce_loss = criterion(stk_out, stk_gt)
            # Classification loss
            stk_cls_lbl = torch.stack([get_class_vector(b["gt_mask"], setup_conf['cls']) for b in batch]).to(device)
            cls_loss = classification_loss(pred_cls, stk_cls_lbl)
            loss += cls_loss
            # Region Decoupling loss
            decoup_loss = decouple_loss(decoupled_res, stk_gt)
            loss += decoup_loss
        
        """ Back-propagation """
        loss.backward()
        optimizer.step()
        scheduler.step()
        torch.cuda.empty_cache()

        one_epoch_loss += loss
        one_epoch_trn_dice += soft_dice(stk_out, stk_gt)
        local_count += 1
    
    trn_loss = one_epoch_loss / local_count
    trn_dice = one_epoch_trn_dice / local_count
    print(f'Epoch {cur_epoch+1}) Train Loss: {trn_loss:.4f}, Train Dice: {trn_dice:.4f}')

    
    """ Validation """
    model.eval()
    criterion.eval()
    torch.cuda.empty_cache()

    validation_loss, one_epoch_val_dice, val_local_count = 0.0, 0.0, 0.0
    with torch.no_grad():
        val_dataset = DatasetSAM(val_list, data_path, processor, net_conf["Prompt"])
        valid_dataloader = DataLoader(val_dataset, batch_size=optim_conf["batch_size"], shuffle=True, collate_fn=collate_fn)

        for batch in tqdm(valid_dataloader, desc=f"Current Epoch: {cur_epoch + 1} - Validation"):
            if setup_conf['model_type'] == 'vit_b':
                outputs = model(batched_input=batch,multimask_output=False)
            
            elif setup_conf['model_type'] == 'DReAM-SAM':
                outputs, pred_cls, _ = model(batched_input=batch, multimask_output=False)

            stk_gt = torch.stack([b["gt_mask"].squeeze(0) for b in batch], dim=0).long()
            stk_out = torch.stack([out["low_res_logits"].squeeze(1) for out in outputs], dim=0)
            
            val_loss, _, _ = criterion(stk_out, stk_gt)
            
            validation_loss += val_loss
            one_epoch_val_dice += soft_dice(stk_out, stk_gt)
            val_local_count += 1

        validation_loss = validation_loss / val_local_count
        val_dice = one_epoch_val_dice / val_local_count
        print(f'Epoch {cur_epoch+1}) Valid_Dice: {val_dice:.4f}')

        if best_val_dice <= val_dice:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': cur_epoch,
                'test_dice': val_dice,},
                os.path.join(model_save_path, f"best_model_val_{cur_epoch}.pth"))
            best_val_dice = val_dice
