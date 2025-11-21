import torch
from torch import nn
from torch.nn import functional as F
from typing import Any, Dict, List, Tuple

from .image_encoder import ImageEncoderViT
from .mask_decoder import MaskDecoder
from .prompt_encoder import PromptEncoder
from .self_prompting import TAPG
from .acnn_branch import ACNN_Branch
from .dna_module import DNA_Module


class DReAM_SAM(nn.Module):
    def __init__(
        self,
        image_encoder: ImageEncoderViT,
        cnn_encoder: ACNN_Branch,
        fusions: DNA_Module,
        prompt_generator: TAPG,
        prompt_encoder: PromptEncoder,
        mask_decoder: MaskDecoder,
        fusion_layers: List[int] = [1, 2, 3, 4],
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
        num_regions: int = 2,
    ) -> None:

        super().__init__()
        self.image_encoder = image_encoder
        self.cnn_encoder = cnn_encoder
        self.fusions = fusions
        self.fusion_layers = fusion_layers
        self.prompt_generator = prompt_generator
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        self.default_embed_dense = nn.Embedding(1, 256).weight.reshape(1, -1, 1, 1).expand(1, -1, 64, 64)
        self.num_regions = num_regions
        self.pwconv = nn.ModuleDict({
            f"M_{i}": nn.Conv2d(in_channels=4, out_channels=1, kernel_size=1)
            for i in range(num_regions)
        })

    @property
    def device(self) -> Any:
        return self.pixel_mean.device

    def forward(
        self,
        batched_input: List[Dict[str, Any]],
        multimask_output: bool = False,
    ) -> List[Dict[str, torch.Tensor]]:

        input_images = torch.stack([self.preprocess(x["image"]) for x in batched_input], dim=0)

        ### Stage 1: patch embed and stem block ###
        sam_feat = self.image_encoder.patch_embed(input_images)
        if self.image_encoder.pos_embed is not None:
            sam_feat = sam_feat + self.image_encoder.pos_embed
            
        cnn_feat = self.cnn_encoder.stem(input_images)
        
        ### Stage 2: ViTEncoder & CNN+DyRA-Block & DNA-Block ###
        decoupled_list = []
        for layer in range(4):
            # SAM layer forward #
            for i in range(layer*3, layer*3+3):
                sam_feat = self.image_encoder.blocks[i](sam_feat)
            
            # CNN layer forward #
            cnn_layer = getattr(self.cnn_encoder, f'layer{layer+1}')
            cnn_feat = cnn_layer(cnn_feat)
            dyra_layer = getattr(self.cnn_encoder, f'dyra{layer+1}')
            cnn_feat, decoupled = dyra_layer(cnn_feat)
            decoupled_list.append(decoupled)
            
            # DNA-Block forward #
            if (layer+1) in self.fusion_layers:
                sam_feat, cnn_feat = self.fusions[layer](sam_feat, cnn_feat)
            
        ### Stage 3: neck block and TEC classifier ###
        sam_feat = self.image_encoder.neck(sam_feat.permute(0, 3, 1, 2))
        
        pred_cls = self.cnn_encoder.avgpool(cnn_feat).squeeze(2).squeeze(2)
        pred_cls = self.cnn_encoder.fc(pred_cls)
        pred_cls = nn.Sigmoid()(pred_cls)
        
        ### Stage 4: Resize CNN features and Decoupled masks ###
        cnn_feat = F.interpolate(cnn_feat, size=(64, 64), mode='bilinear', align_corners=False)

        region_dict = {f"M_{i}": [] for i in range(self.num_regions)}
        for decoupled in decoupled_list:
            for key in region_dict.keys():
                M = decoupled[key]
                M = F.interpolate(M, size=(64,64), mode='bilinear', align_corners=False)
                region_dict[key].append(M)
        # Decoupled masks aggregation # 
        for k in region_dict:
            region_dict[k] = torch.cat(region_dict[k], dim=1)
            region_dict[k] = self.pwconv[k](region_dict[k])
        decoupled_feat = []
        for b in range(sam_feat.shape[0]):
            decoupled_embed = {k: region_dict[k][b] for k in region_dict}
            decoupled_feat.append(decoupled_embed)
        
        
        outputs = []
        for image_record, sam_embed, decoupled_embed, class_prob in zip(batched_input, sam_feat, decoupled_feat, pred_cls):
            ### Stage 5: Prompt generation ###
            sparse_embeddings, dense_embeddings = [], []
            for n, cls_prob in enumerate(class_prob):
                k = f"M_{n+1}"
                if k in decoupled_embed.keys():
                    dense, sparse = self.prompt_generator(decoupled_embed[k], class_prob=cls_prob)
                    dense_embeddings.append(dense)
                    sparse_embeddings.append(sparse)
            # For background #
            dense, sparse = self.prompt_generator(decoupled_embed['M_0'], class_prob=0.5)
            dense_embeddings.append(dense)
            sparse_embeddings.append(sparse)
                
            sparse_embeddings = torch.stack(sparse_embeddings, dim=0).squeeze(1)
            dense_embeddings = torch.stack(dense_embeddings, dim=0).squeeze(1)
            
            ### Stage 6: SAM mask decoder ###
            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=sam_embed.unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )
            masks = self.postprocess_masks(
                low_res_masks,
                input_size=image_record["image"].shape[-2:],
                original_size=image_record["original_size"],
            )
            square_masks = self.postprocess_masks(
                low_res_masks,
                input_size=image_record["image"].shape[-2:],
                original_size=(1024, 1024),
            )
            outputs.append(
                {
                    "masks": masks,
                    "iou_predictions": iou_predictions,
                    "low_res_logits": square_masks,
                }
            )
        return outputs, pred_cls, decoupled_list

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x
