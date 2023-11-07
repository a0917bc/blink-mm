import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.jit import Final
import numpy as np
import torchsnooper

from timm.loss import LabelSmoothingCrossEntropy
from timm.data import Mixup
from timm import create_model
from timm.layers import use_fused_attn
from timm.scheduler import CosineLRScheduler

from pprint import pprint

from qat.export.utils import replace_module_by_name, fetch_module_by_name
# from operations.amm_linear import AMMLinear, PQLinear, TrivenLinear
from blink_mm.ops.amm_linear import AMMLinear
from .losses import DistillationLoss, DistillationLoss_v2
from .vision_transformer import VisionTransformer

def create_target(start_replaced_layer_idx, end_replaced_layer_idx, model_name, subvec_len=32, k=16):
    # model = torch.load(f"/home/u1887834/Research/notebook/{model_name}_120000_base_0_12.pth") # deit3_small_patch16_224.fb_in22k_ft_in1k
                        # deit3_small_patch16_224.fb_in22k_ft_in1k_120000_base_0_12
    # model = torch.load(f"/home/u1887834/Research/notebook/{model_name}.pth")
    # model = create_model("deit3_small_patch16_224.fb_in1k", pretrained=False)
    model = create_model(model_name=model_name, pretrained=False)
    ncodebooks = {
        "attn.qkv": 384 // subvec_len,
        # "attn.q_linear": 384 // subvec_len, 
        # "attn.k_linear": 384 // subvec_len, 
        "mlp.fc1": 384 // subvec_len
        # "mlp.fc2": 1536 // subvec_len
        
        # "linear_tokens": 196 // 14, 
        # "mlp_channels.fc1": 384 // subvec_len,
        # "mlp_channels.fc2":1536 // subvec_len
    }
    
    for i in range(start_replaced_layer_idx, end_replaced_layer_idx): 
        for name in ncodebooks:
            layer = model.blocks[i]
            module = fetch_module_by_name(layer, name)
            amm_linear = AMMLinear(
            # amm_linear = PQLinear(
                ncodebooks[name],
                module.in_features,
                module.out_features,
                module.bias is not None,
                k=k
            )
            amm_linear.inverse_temperature_logit.data.copy_(
                torch.tensor(10)
            )
            # print(amm_linear.weight.data.shape)
            # print(module.weight.data.shape)
            # weight = rearrange(module.weight.data, 'o i -> i o')
            # weight = rearrange(weight, '(c v) o -> c v o', c=ncodebooks[name], v=subvec_len)
            # amm_linear.weight.data.copy_(weight.data)
            # amm_linear.bias.data.copy_(module.bias.data)
            replace_module_by_name(layer, name, amm_linear)
    return model

class LUT_DeiT(L.LightningModule):
    def __init__(self, 
                 model = None,
                 kmeans_init=False, 
                 start_replaced_layer_idx=9, 
                 end_replaced_layer_idx=12, 
                 num=1024,
                 model_name="deit3_small_patch16_224.fb_in22k_ft_in1k",
                 distillation_type="hard",
                 tau=1/2,
                 alpha=1/2,
                 max_iters = 100,
                 smoothing=0.1,
                 adam_epsilon: float = 1e-8,
                 lr=5e-4, 
                 weight_decay=0.1
                 ):
        super().__init__()
        float_model = create_model(model_name, pretrained=True)
        float_model.eval()
        for param in float_model.parameters():
            param.requires_grad = False
        self.float_model = torch.compile(float_model)
        
        self.save_hyperparameters(ignore=['model'])
        self.model = create_target(9, 12, model_name=model_name)
        # print(self.model)
        # exit()
        # if kmeans_init:
        #     from pathlib import Path   
        #     save_path = Path('/home/u1887834/Research/old_base_model') # TODO .... 
        #     # self.model.load_state_dict(torch.load(save_path / f"{num}_base_{start_replaced_layer_idx}_{end_replaced_layer_idx}.pt"))
        #     self.model.load_state_dict(torch.load(f"/home/u1887834/Research/base_model_qk/{model_name}_120000_base_0_12.pt"))
        #                                            #/home/u1887834/Research/base_model_qk/deit3_small_patch16_224.fb_in22k_ft_in1k_120000_base_0_12.pt
        # from pathlib import Path   
        # save_path = Path('/home/u1887834/Research/base_model_qk')
        # if model is not None:
        #     self.model = model
            
        # for l in range(12):
        #     for name, param in self.model.blocks[l].named_parameters():
        #         # if 'attn.q_linear' in name or 'attn.k_linear' in name:
        #         #     param.requires_grad = True
        #         if 'attn.q_linear.centroids' in name or 'attn.k_linear.centroids' in name or 'inverse_temperature_logit' in name:
        #             param.requires_grad = True
                
        #         elif "patch_embed" in name:
        #             param.requires_grad = False
                
        #         elif "norm" in name:
        #             param.requires_grad = False
                
        #         elif "head" in name:
        #             param.requires_grad = False
                
        #         else:
        #             param.requires_grad = False
        # self.model.patch_embed.proj.weight.requires_grad = False
        # self.model.cls_token.requires_grad = False
        # self.model.pos_embed.requires_grad = False
        # self.model.patch_embed.proj.bias.requires_grad = False
        # self.model.norm.weight.requires_grad = False
        # self.model.norm.bias.requires_grad = False
        # self.model.head.weight.requires_grad = False
        # self.model.head.bias.requires_grad = False
        # # print(self.model.patch_embed.requires_grad)
        # for name, param in self.model.named_parameters():
        #     if param.requires_grad:
        #         print(f"Parameter: {name}, Requires Gradient: {param.requires_grad}")

        # loss
        self.criterion = LabelSmoothingCrossEntropy(smoothing=smoothing)
        # self.distil_loss = DistillationLoss(base_criterion=self.criterion, 
        #                                     teacher_model=float_model, 
        #                                     distillation_type=distillation_type,
        #                                     alpha=alpha,
        #                                     tau=tau)
        self.distil_loss = DistillationLoss_v2(distillation_type=distillation_type, tau=tau)
    def forward(self, x):
        return self.model(x)

    def common_step_v2(self, x, y, stage):
        student_logits = self(x)
        teacher_logits = self.float_model(x)
        distill_loss = self.distil_loss(student_logits, teacher_logits) 
        base_loss = self.criterion(student_logits, y)
        acc = (student_logits.argmax(dim=-1) == y).float().mean()
        self.log(f"{stage}_distill_loss", distill_loss, 
                 on_step=True, on_epoch=True, sync_dist=True)
        self.log(f"{stage}_base_loss", base_loss, 
                 on_step=True, on_epoch=True, sync_dist=True)
        self.log(f"{stage}_acc", acc, 
                 on_step=True, on_epoch=True, sync_dist=True)
        return 0.2*base_loss + 0.8*distill_loss
    def common_step(self, x, y, stage):
        logits = self(x)
        # pprint(logits.shape)
        # pprint(y.shape)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=-1) == y).float().mean()
        self.log(f"{stage}_loss", loss, on_epoch=True, sync_dist=True)
        self.log(f"{stage}_acc", acc, on_epoch=True, sync_dist=True)
        return loss
    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.common_step_v2(x, y, "train")
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss = self.common_step(x, y, "val")
    
    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        # Initialize lists to hold parameters for each group
        default_params = []
        special_linear_params = []
        no_decay_params = []

        # Define the names or substrings that should not have weight decay
        no_decay = ["bias", "LayerNorm.weight"]

        # Iterate through all named parameters
        for name, param in self.model.named_parameters():
            if any(nd in name for nd in no_decay):
                # Parameters that should not have weight decay
                no_decay_params.append(param)
            elif 'inverse_temperature_logit' in name:  # Replace with the actual name or substring
                # Parameters in nn.Linear that should have a special learning rate
                special_linear_params.append(param)
            else:
                # All other parameters
                default_params.append(param)

        # Create the list of parameter groups
        optimizer_grouped_parameters = [
            {'params': default_params, 'weight_decay': self.hparams.weight_decay},  # Default learning rate and weight decay
            {'params': no_decay_params, 'weight_decay': 0.0},  # No weight decay
            {'params': special_linear_params, 'lr': 0.1}  # Special learning rate for specific nn.Linear parameters
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.hparams.lr, eps=self.hparams.adam_epsilon)
        scheduler = CosineLRScheduler(optimizer, cycle_limit=1, t_initial = self.hparams.max_iters)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]
    
    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step(epoch=self.current_epoch)  # timm's scheduler need the epoch value


# +
class Argmax_DeiT(L.LightningModule):
    def __init__(self, 
                 model = None,
                 kmeans_init=False, 
                 start_replaced_layer_idx=0, 
                 end_replaced_layer_idx=12, 
                 num=1024,
                 model_name="deit3_small_patch16_224.fb_in22k_ft_in1k",
                 distillation_type="soft",
                 tau=1/2,
                 alpha=1/2,
                 max_iters = 100,
                 smoothing=0.1,
                 adam_epsilon: float = 1e-8,
                 lr=5e-4, 
                 weight_decay=0.1
                 ):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        if model is not None:
            self.model = model
            # torch.load(f"/home/u1887834/Research/notebook/{model_name}.pth") # deit3_small_patch16_224.fb_in22k_ft_in1k
                               # "/home/u1887834/Research/notebook/argmax_deit3_small_patch16_224.fb_in22k_ft_in1k.pth"
        # self.model = torch.load(f"/home/u1887834/Research/{model_name}.pth")
        # self.model = VisionTransformer(weight_init="skip", embed_dim=384, num_heads=6, class_token=True)
        
        
        # self.model.load_state_dict(torch.load(f"/home/u1887834/Research/{model_name}.pth"))
        # print(self.model)
        float_model = create_model(model_name, pretrained=True)
        float_model.eval()
        for param in float_model.parameters():
            param.requires_grad = False
        float_model = torch.compile(float_model)
        self.float_model = float_model
        # self.model = create_model(f"{model_name}", pretrained=True)
        
        if kmeans_init:
            from pathlib import Path   
            save_path = Path('/home/u1887834/Research/old_base_model') # TODO .... 
            # new_model_state_dict = torch.load(save_path / f"{num}_base_{start_replaced_layer_idx}_{end_replaced_layer_idx}.pt")
            self.model.load_state_dict(torch.load(save_path / f"{num}_base_{start_replaced_layer_idx}_{end_replaced_layer_idx}.pt"))
#         for param in self.model.parameters():
#             param.requires_grad = False
            
#         self.model.head.weight.requires_grad = True
#         self.model.head.bias.requires_grad = True
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(f"Parameter: {name}, Requires Gradient: {param.requires_grad}")
                
        # for l in range(start_replaced_layer_idx, end_replaced_layer_idx):
        #     for param in self.model.blocks[l].parameters():
        #         param.requires_grad = True
        
        # loss
        self.criterion = LabelSmoothingCrossEntropy(smoothing=smoothing)
        self.distil_loss = DistillationLoss_v2(distillation_type=distillation_type, tau=tau)
    
    def forward(self, x):
        return self.model(x)

    def common_step_v2(self, x, y, stage):
        student_logits = self(x)
        teacher_logits = self.float_model(x)
        distill_loss = self.distil_loss(student_logits, teacher_logits) 
        base_loss = self.criterion(student_logits, y)
        acc = (student_logits.argmax(dim=-1) == y).float().mean()
        self.log(f"{stage}_distill_loss", distill_loss, 
                 on_step=True, on_epoch=True, sync_dist=True)
        self.log(f"{stage}_base_loss", base_loss, 
                 on_step=True, on_epoch=True, sync_dist=True)
        self.log(f"{stage}_acc", acc, 
                 on_step=True, on_epoch=True, sync_dist=True)
        return 0.2*base_loss + 0.8*distill_loss
    def common_step(self, x, y, stage):
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=-1) == y).float().mean()
        self.log(f"{stage}_loss", loss, on_epoch=True, sync_dist=True)
        self.log(f"{stage}_acc", acc, on_epoch=True, sync_dist=True)
        return loss
    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.common_step_v2(x, y, "train")
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss = self.common_step(x, y, "val")
    
    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        # Initialize lists to hold parameters for each group
        default_params = []
        special_linear_params = []
        no_decay_params = []

        # Define the names or substrings that should not have weight decay
        no_decay = ["bias", "LayerNorm.weight"]

        # Iterate through all named parameters
        for name, param in self.model.named_parameters():
            if any(nd in name for nd in no_decay):
                # Parameters that should not have weight decay
                no_decay_params.append(param)
            elif 'inverse_temperature_logit' in name:  # Replace with the actual name or substring
                # Parameters in nn.Linear that should have a special learning rate
                special_linear_params.append(param)
            else:
                # All other parameters
                default_params.append(param)

        # Create the list of parameter groups
        optimizer_grouped_parameters = [
            {'params': default_params, 'weight_decay': self.hparams.weight_decay},  # Default learning rate and weight decay
            {'params': no_decay_params, 'weight_decay': 0.0},  # No weight decay
            {'params': special_linear_params, 'lr': 0.1}  # Special learning rate for specific nn.Linear parameters
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.hparams.lr, eps=self.hparams.adam_epsilon)
        # optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.hparams.lr, eps=self.hparams.adam_epsilon)

        scheduler = CosineLRScheduler(optimizer, cycle_limit=1, t_initial = self.hparams.max_iters)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]
    
    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step(epoch=self.current_epoch)  # timm's scheduler need the epoch value


# -

class LUT_Distilled_DeiT(L.LightningModule):
    def __init__(self, 
                 kmeans_init=True, 
                 start_replaced_layer_idx=0, 
                 end_replaced_layer_idx=12, 
                 current_layer = 0,
                 num=1024,
                 model_name="deit3_small_patch16_224.fb_in22k_ft_in1k",
                 distillation_type="hard",
                 tau=1,
                 alpha=1/2,
                 smoothing=0.1,
                 adam_epsilon: float = 1e-8,
                 lr=5e-4, 
                 weight_decay=0.1,
                 max_iters=100
                 ):
        super().__init__()
        float_model = create_model(model_name, pretrained=True)
        float_model.eval()
        for param in float_model.parameters():
            param.requires_grad = False
        float_model = torch.compile(float_model)
        self.float_model = float_model
        self.save_hyperparameters()
        
        
        
        
        self.model = create_target(start_replaced_layer_idx, end_replaced_layer_idx, model_name=model_name)
        for param in self.model.parameters():
            param.requires_grad = False
        for l in range(start_replaced_layer_idx, end_replaced_layer_idx):
            for param in self.model.blocks[l].parameters():
                param.requires_grad = True
            
        if kmeans_init:
            from pathlib import Path   
            save_path = Path('/home/u1887834/Research/base_model')
            self.model.load_state_dict(torch.load(save_path / f"{num}_base_{start_replaced_layer_idx}_{end_replaced_layer_idx}.pt"))
        # loss
        self.mse = nn.MSELoss()
        # self.criterion = LabelSmoothingCrossEntropy(smoothing=smoothing)
        # self.distil_loss = DistillationLoss(base_criterion=self.criterion, 
        #                                     teacher_model=float_model, 
        #                                     distillation_type=distillation_type,
        #                                     alpha=alpha,
        #                                     tau=tau)
        self.distil_loss_v2 = DistillationLoss_v2(
                                            distillation_type=distillation_type,
                                            tau=tau)
    def forward(self, x):
        return self.model(x)

    def distill_step(self, x, layer, stage):
        student_logits = self.model.get_intermediate_layers(x, n=[layer])[0] # return tuple...
        teacher_logits = self.float_model.get_intermediate_layers(x, n=[layer])[0]
        # distill_loss = self.distil_loss_v2(student_logits, teacher_logits) 
        mse_loss = self.mse(student_logits, teacher_logits)
        # self.log(f"{stage}_distill_loss", distill_loss, 
        #          on_step=True, on_epoch=True, sync_dist=True)
        self.log(f"{stage}_mse_loss", mse_loss, 
                 on_step=True, on_epoch=True, sync_dist=True)
        return mse_loss
    
    def common_step_v2(self, x, y, stage):
        student_logits = self(x)
        base_loss = self.criterion(student_logits, y)
        loss = self.distil_loss(x, student_logits, y) # def forward(self, inputs, outputs, labels):
        # for l in range(self.hparams.start_replaced_layer_idx, self.hparams.end_replaced_layer_idx):
        #     loss += self.distill_step(x, l, stage)
        acc = (student_logits.argmax(dim=-1) == y).float().mean()
        self.log(f"{stage}_loss", loss, 
                 on_step=True, on_epoch=True, sync_dist=True)
        self.log(f"{stage}_base_loss", base_loss, 
                 on_step=True, on_epoch=True, sync_dist=True)
        self.log(f"{stage}_acc", acc, 
                 on_step=True, on_epoch=True, sync_dist=True)
        self.current_epoch
        return loss
    
    def training_step(self, batch, batch_idx):
        self.batch_idx = batch_idx
        x, y = batch
        # loss = self.common_step_v2(x, y, "train")
        # for l in range(self.hparams.layer, self.hparams.end_replaced_layer_idx):
        loss = self.distill_step(x, self.hparams.current_layer, "train")
        return loss
    
    def common_step(self, x, y, stage):
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=-1) == y).float().mean()
        self.log(f"{stage}_loss", loss, on_epoch=True, sync_dist=True)
        self.log(f"{stage}_acc", acc, on_epoch=True, sync_dist=True)
        return loss
        
    def validation_step(self, batch, batch_idx):
        x, y = batch
        # loss = self.common_step(x, y, "val")
        loss = self.distill_step(x, self.hparams.current_layer, "val")
    
    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        # Initialize lists to hold parameters for each group
        default_params = []
        special_linear_params = []
        no_decay_params = []

        # Define the names or substrings that should not have weight decay
        no_decay = ["bias", "LayerNorm.weight"]

        # Iterate through all named parameters
        for name, param in model.named_parameters():
            if any(nd in name for nd in no_decay):
                # Parameters that should not have weight decay
                no_decay_params.append(param)
            elif 'inverse_temperature_logit' in name:  # Replace with the actual name or substring
                # Parameters in nn.Linear that should have a special learning rate
                special_linear_params.append(param)
            else:
                # All other parameters
                default_params.append(param)

        # Create the list of parameter groups
        optimizer_grouped_parameters = [
            {'params': default_params, 'weight_decay': self.hparams.weight_decay},  # Default learning rate and weight decay
            {'params': no_decay_params, 'weight_decay': 0.0},  # No weight decay
            {'params': special_linear_params, 'lr': 0.1}  # Special learning rate for specific nn.Linear parameters
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.hparams.lr, eps=self.hparams.adam_epsilon)
        scheduler = CosineLRScheduler(optimizer, cycle_limit=1, t_initial = self.hparams.max_iters)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]
    
    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step(epoch=self.current_epoch)  # timm's scheduler need the epoch value

class Attention2(nn.Module):
    fused_attn: Final[bool]

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        # self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_linear = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_linear = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_linear = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_score = AttnScore(scale=self.scale, attn_drop=attn_drop)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        # q, k, v = qkv.unbind(0)
        q = self.q_linear(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_linear(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_linear(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        q, k = self.q_norm(q), self.k_norm(k)

        # if self.fused_attn:
        #     x = F.scaled_dot_product_attention(
        #         q, k, v,
        #         dropout_p=self.attn_drop.p,
        #     )
        # else:
        #     # q = q * self.scale
        #     # attn = q @ k.transpose(-2, -1)
        #     # attn = attn.softmax(dim=-1)
        #     # attn = self.attn_drop(attn)
        #     attn = self.attn_score(q, k)
        #     x = attn @ v
        attn = self.attn_score(q, k)
        x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class AttnScore(nn.Module):
    def __init__(self, scale, attn_drop):
        super().__init__()
        self.scale = scale
        self.attn_drop = nn.Dropout(attn_drop) # one is func. another one is probability.
    def forward(self, q, k):
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        return attn

class Attention3(nn.Module):
    fused_attn: Final[bool]
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()
        self.attn_score = AttnScore(scale=self.scale, attn_drop=attn_drop)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.register_parameter(
            "inverse_temperature_logit",
            nn.Parameter(torch.randn(1))
        )
        
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    # @torchsnooper.snoop()
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        # if self.fused_attn:
        #     x = F.scaled_dot_product_attention(
        #         q, k, v,
        #         dropout_p=self.attn_drop.p,
        #     )
        # else:
            # q = q * self.scale
            # attn = q @ k.transpose(-2, -1)
            # attn = attn.softmax(dim=-1)
            # attn_argmax = attn.argmax(dim=-1)
            # attn = self.attn_drop(attn)
            # output = attn - (attn - attn_argmax).detach() 
            # x = attn_argmax @ v
        # attn = self.attn_score(q, k)
        
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        # attn = attn.softmax(dim=-1)
        attn = F.softmax(attn * (F.softplus(self.inverse_temperature_logit) + 1), dim=-1) # TODO
        # attn = F.softmax(attn * ), dim=-1) # TODO How to implement annealing?
        # attn = F.softmax(attn, dim=-1)
        attn_argmax = F.one_hot(attn.argmax(dim=-1), num_classes=N).float()
        attn = self.attn_drop(attn)
        output = attn - (attn - attn_argmax).detach()
        # x = attn @ v
        x = output @ v
        
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
