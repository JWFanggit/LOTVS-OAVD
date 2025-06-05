import argparse
import datetime
import logging
import inspect
import math
import os
import random
import gc
import time
import copy
from dada import DADA2KS3

from typing import Dict, Optional, Tuple
from omegaconf import OmegaConf

import cv2
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torchvision.transforms as T
import diffusers
import transformers
from torchvision import transforms
from tqdm.auto import tqdm
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from models.unet_3d_condition import UNet3DConditionModel
from diffusers.models import AutoencoderKL
from diffusers import DPMSolverMultistepScheduler, DDPMScheduler
from OAVD_Pipeline import TextToVideoSDPipeline
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from diffusers.models.attention_processor import AttnProcessor2_0
from models.attentions import BasicTransformerBlock
from transformers import CLIPTextModel, CLIPTokenizer
from transformers.models.clip.modeling_clip import CLIPEncoder
from Fourier import PositionNet
from einops import rearrange, repeat

from torch.utils.tensorboard import SummaryWriter

from utils.lora import (
    extract_lora_ups_down,
    inject_trainable_lora,
    inject_trainable_lora_extended,
    save_lora_weight,
    train_patch_pipe,
    monkeypatch_or_replace_lora,
    monkeypatch_or_replace_lora_extended
)

already_printed_trainables = False

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.10.0.dev0")

logger = get_logger(__name__, log_level="INFO")

def create_logging(logging, logger, accelerator):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)


def accelerate_set_verbose(accelerator):
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

def extend_datasets(datasets, dataset_items, extend=False):
    biggest_data_len = max(x.__len__() for x in datasets)
    extended = []
    for dataset in datasets:
        if dataset.__len__() == 0:
            del dataset
            continue
        if dataset.__len__() < biggest_data_len:
            for item in dataset_items:
                if extend and item not in extended and hasattr(dataset, item):
                    print(f"Extending {item}")

                    value = getattr(dataset, item)
                    value *= biggest_data_len
                    value = value[:biggest_data_len]

                    setattr(dataset, item, value)

                    print(f"New {item} dataset length: {dataset.__len__()}")
                    extended.append(item)

def create_output_folders(output_dir, config):
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    out_dir = os.path.join(output_dir, f"train_{now}")
    
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(f"{out_dir}/samples", exist_ok=True)
    OmegaConf.save(config, os.path.join(out_dir, 'config.yaml'))

    return out_dir


def unet_and_text_g_c(unet, text_encoder, unet_enable, text_enable):
    unet._set_gradient_checkpointing(value=unet_enable)
    text_encoder._set_gradient_checkpointing(CLIPEncoder, value=text_enable)

def freeze_models(models_to_freeze):
    for model in models_to_freeze:
        if model is not None: model.requires_grad_(False) 
            
def is_attn(name):
   return ('attn1' or 'attn2' == name.split('.')[-1])

def set_processors(attentions):
    for attn in attentions: attn.set_processor(AttnProcessor2_0()) 

def set_torch_2_attn(unet):
    optim_count = 0
    
    for name, module in unet.named_modules():
        if is_attn(name):
            if isinstance(module, torch.nn.ModuleList):
                for m in module:
                    if isinstance(m, BasicTransformerBlock):
                        set_processors([m.attn1, m.attn2])
                        optim_count += 1
    if optim_count > 0: 
        print(f"{optim_count} Attention layers using Scaled Dot Product Attention.")

def handle_memory_attention(enable_xformers_memory_efficient_attention, enable_torch_2_attn, unet): 
    # try:
    #     is_torch_2 = hasattr(F, 'scaled_dot_product_attention')

    if enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            from xformers.ops import MemoryEfficientAttentionFlashAttentionOp
            unet.enable_xformers_memory_efficient_attention(attention_op=MemoryEfficientAttentionFlashAttentionOp)
            print("xformers has injected to unet")
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    #     if enable_torch_2_attn and is_torch_2:
    #         set_torch_2_attn(unet)
    # except:
    #     print("Could not enable memory efficient attention for xformers or Torch 2.0.")

def inject_lora(use_lora, model, replace_modules, is_extended=False, dropout=0.0, lora_path='', r=16):    
    injector = (
        inject_trainable_lora if not is_extended 
    else 
        inject_trainable_lora_extended
    )

    params = None
    negation = None

    if os.path.exists(lora_path):
        try:
            for f in os.listdir(lora_path):
                if f.endswith('.pt'):
                    lora_file = os.path.join(lora_path, f)
                    
                    if 'text_encoder' in f and isinstance(model, CLIPTextModel):
                        monkeypatch_or_replace_lora(
                            model,
                            torch.load(lora_file),
                            target_replace_module=replace_modules,
                            r=r
                        )
                        print("Successfully loaded Text Encoder LoRa.")

                    if 'unet' in f and isinstance(model, UNet3DConditionModel):
                        monkeypatch_or_replace_lora_extended(
                            model,
                            torch.load(lora_file),
                            target_replace_module=replace_modules,
                            r=r
                        )
                        print("Successfully loaded UNET LoRa.")

        except Exception as e:
            print(e)
            print("Could not load LoRAs. Injecting new ones instead...")

    if use_lora:
        REPLACE_MODULES = replace_modules
        injector_args = {
            "model": model, 
            "target_replace_module": REPLACE_MODULES,
            "r": r
        }
        if not is_extended: injector_args['dropout_p'] = dropout

        params, negation = injector(**injector_args)   
        for _up, _down in extract_lora_ups_down(
            model, 
            target_replace_module=REPLACE_MODULES):

            if all(x is not None for x in [_up, _down]):
                print(f"Lora successfully injected into {model.__class__.__name__}.")

            break
        
    return params, negation

def save_lora(model, name, condition, replace_modules, step, save_path): 
    if condition and replace_modules is not None:
        save_path = f"{save_path}/{step}_{name}.pt"
        save_lora_weight(model, save_path, replace_modules)

def handle_lora_save(
    use_unet_lora, 
    use_text_lora, 
    model, 
    save_path, 
    checkpoint_step, 
    unet_target_modules, 
    text_encoder_target_modules
):
    
    save_path = f"{save_path}/lora"
    os.makedirs(save_path, exist_ok=True)

    save_lora(
        model.unet, 
        'unet', 
        use_unet_lora, 
        unet_target_modules, 
        checkpoint_step,
        save_path, 
    )
    save_lora(
        model.text_encoder, 
        'text_encoder', 
        use_text_lora, 
        text_encoder_target_modules, 
        checkpoint_step, 
        save_path
    )

    train_patch_pipe(model, use_unet_lora, use_text_lora)

def param_optim(model, condition, extra_params=None, is_lora=False, negation=None):
    return {
        "model": model, 
        "condition": condition, 
        'extra_params': extra_params,
        'is_lora': is_lora,
        "negation": negation
    }
    

def create_optim_params(name='param', params=None, lr=5e-6, extra_params=None):
    params = {
        "name": name, 
        "params": params, 
        "lr": lr
    }

    if extra_params is not None:
        for k, v in extra_params.items():
            params[k] = v
    
    return params

def negate_params(name, negation):
    # We have to do this if we are co-training with LoRA.
    # This ensures that parameter groups aren't duplicated.
    if negation is None: return False
    for n in negation:
        if n in name and 'temp' not in name:
            return True
    return False


def create_optimizer_params(model_list, lr):
    import itertools
    optimizer_params = []

    for optim in model_list:
        model, condition, extra_params, is_lora, negation = optim.values()
        # Check if we are doing LoRA training.
        if is_lora and condition:
            params = create_optim_params(
                params=itertools.chain(*model),
                extra_params=extra_params
            )
            optimizer_params.append(params)
            continue

        # If this is true, we can train it.
        if condition:
            for n, p in model.named_parameters():
                should_negate = 'lora' in n
                if should_negate: continue

                params = create_optim_params(n, p, lr, extra_params)
                optimizer_params.append(params)

    return optimizer_params

def get_optimizer(use_8bit_adam):
    if use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        return bnb.optim.AdamW8bit
    else:
        return torch.optim.AdamW

def is_mixed_precision(accelerator):
    weight_dtype = torch.float32

    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16

    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    return weight_dtype

def cast_to_gpu_and_type(model_list, accelerator, weight_dtype):
    for model in model_list:
        if model is not None: model.to(accelerator.device, dtype=weight_dtype)


def handle_cache_latents(
        should_cache,
        output_dir,
        train_dataloader,
        train_batch_size,
        vae,
        cached_latent_dir=None
    ):

    # Cache latents by storing them in VRAM.
    # Speeds up training and saves memory by not encoding during the train loop.
    if not should_cache: return None
    device=torch.device("cuda",2)
    # vae.to('cuda', dtype=torch.float16)
    vae=vae.to(device,dtype=torch.float16)
    vae.enable_slicing()



def handle_trainable_modules(model, trainable_modules=None, is_enabled=True, negation=None):
    global already_printed_trainables

    # This can most definitely be refactored :-)
    unfrozen_params = 0
    if trainable_modules is not None:
        for name, module in model.named_modules():
            for tm in tuple(trainable_modules):
                if tm == 'all':
                    model.requires_grad_(is_enabled)
                    unfrozen_params =len(list(model.parameters()))
                    break
                    
                if tm in name and 'lora' not in name:
                    for m in module.parameters():
                        m.requires_grad_(is_enabled)
                        if is_enabled: unfrozen_params +=1

    if unfrozen_params > 0 and not already_printed_trainables:
        already_printed_trainables = True 
        print(f"{unfrozen_params} params have been unfrozen for training.")

def tensor_to_vae_latent(t, vae):
    video_length = t.shape[1]
    t = rearrange(t, "b f c h w -> (b f) c h w")
    latents = vae.encode(t).latent_dist.sample()
    latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)
    latents = latents * 0.18215
    return latents

def sample_noise(latents, noise_strength, use_offset_noise):
    b ,c, f, *_ = latents.shape
    noise_latents = torch.randn_like(latents, device=latents.device)
    offset_noise = None

    if use_offset_noise:
        offset_noise = torch.randn(b, c, f, 1, 1, device=latents.device)
        noise_latents = noise_latents + noise_strength * offset_noise

    return noise_latents

def should_sample(global_step, validation_steps, validation_data):
    return (global_step % validation_steps == 0 or global_step == 1)  \
    and validation_data.sample_preview


def save_pipe(
        path,
        global_step,
        accelerator,
        unet,
        text_encoder,
        vae,
        output_dir,
        use_unet_lora,
        use_text_lora,
        unet_target_replace_module=None,
        text_target_replace_module=None,
        is_checkpoint=False,
):
    if is_checkpoint:
        save_path = os.path.join(output_dir, f"checkpoint-{global_step}")
        os.makedirs(save_path, exist_ok=True)
    else:
        save_path = output_dir

    # Save the dtypes so we can continue training at the same precision.
    u_dtype, t_dtype, v_dtype = unet.dtype, text_encoder.dtype, vae.dtype

    # Copy the model without creating a reference to it. This allows keeping the state of our lora training if enabled.
    unet_out = copy.deepcopy(accelerator.unwrap_model(unet, keep_fp32_wrapper=False))
    text_encoder_out = copy.deepcopy(accelerator.unwrap_model(text_encoder, keep_fp32_wrapper=False))
    pipeline = TextToVideoSDPipeline.from_pretrained(
        path,
        unet=unet_out,
        text_encoder=text_encoder_out,
        vae=vae,
    ).to(torch_dtype=torch.float32)

    handle_lora_save(
        use_unet_lora,
        use_text_lora,
        pipeline,
        output_dir,
        global_step,
        unet_target_replace_module,
        text_target_replace_module
    )
    pipeline.save_pretrained(save_path)
    if is_checkpoint:
        unet, text_encoder = accelerator.prepare(unet, text_encoder)
        models_to_cast_back = [(unet, u_dtype), (text_encoder, t_dtype), (vae, v_dtype)]
        [x[0].to(accelerator.device, dtype=x[1]) for x in models_to_cast_back]

    logger.info(f"Saved model at {save_path} on step {global_step}")

    del pipeline
    del unet_out
    del text_encoder_out
    torch.cuda.empty_cache()
    gc.collect()


def replace_prompt(prompt, token, wlist):
    for w in wlist:
        if w in prompt: return prompt.replace(w, token)
    return prompt


def shape_change(bbx):
    B,F,N,C=bbx.shape
    bbx=bbx.view(B*F, N, -1)
    return bbx



def main(
    pretrained_model_path: str,
    output_dir: str,
    train_data: Dict,
    validation_data: Dict,
    dataset_types: Tuple[str] = ('json'),
    validation_steps: int = 100,
    trainable_modules: Tuple[str] = ("attn1","attn1.5","attn2"),
    trainable_text_modules: Tuple[str] = ("all"),
    extra_unet_params = None,
    extra_text_encoder_params = None,
    train_batch_size: int = 1,
    max_train_steps: int = 500,
    learning_rate: float = 5e-5,
    scale_lr: bool = False,
    lr_scheduler: str = "constant",
    lr_warmup_steps: int = 0,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_weight_decay: float = 1e-2,
    adam_epsilon: float = 1e-08,
    max_grad_norm: float = 1.0,
    gradient_accumulation_steps: int = 1,
    gradient_checkpointing: bool = False,
    text_encoder_gradient_checkpointing: bool = False,
    checkpointing_steps: int = 500,
    resume_from_checkpoint: Optional[str] = None,
    mixed_precision: Optional[str] = "fp16",
    use_8bit_adam: bool = False,
    enable_xformers_memory_efficient_attention: bool = True,
    enable_torch_2_attn: bool = False,
    seed: Optional[int] = None,
    train_text_encoder: bool = False,
    use_offset_noise: bool = False,
    offset_noise_strength: float = 0.1,
    extend_dataset: bool = False,
    cache_latents: bool = False,
    cached_latent_dir = None,
    use_unet_lora: bool = False,
    use_text_lora: bool = False,
    unet_lora_modules: Tuple[str] = ["ResnetBlock2D"],
    text_encoder_lora_modules: Tuple[str] = ["CLIPEncoderLayer"],
    lora_rank: int = 16,
    lora_path: str = '',
    **kwargs
):

    *_, config = inspect.getargvalues(inspect.currentframe())
    global_step = 0
    first_epoch = 0
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        log_with="tensorboard",
        logging_dir=output_dir
    )

    # Make one log on every process with the configuration for debugging.
    create_logging(logging, logger, accelerator)

    # Initialize accelerate, transformers, and diffusers warnings
    accelerate_set_verbose(accelerator)

    # If passed along, set the training seed now.
    if seed is not None:
        set_seed(seed)

    # Handle the output folder creation
    if accelerator.is_main_process:
       output_dir = create_output_folders(output_dir, config)

    # Load scheduler, tokenizer and models.
    pretrained_model_path=r" "
    noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    unet = UNet3DConditionModel.from_pretrained_2d(pretrained_model_path, subfolder="unet")
    p_net = PositionNet(64, 1024)

    # Freeze any necessary models
    freeze_models([vae, text_encoder, unet])

    # or open unet models
    # freeze_models([vae, text_encoder])
    
    # Enable xformers if available
    handle_memory_attention(enable_xformers_memory_efficient_attention, enable_torch_2_attn, unet)

    if scale_lr:
        learning_rate = (
            learning_rate * gradient_accumulation_steps * train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    optimizer_cls = get_optimizer(use_8bit_adam)

    # Use LoRA if enabled.
    unet_lora_params, unet_negation = inject_lora(
        use_unet_lora, unet, unet_lora_modules, is_extended=True,
        r=lora_rank, lora_path=lora_path
        )

    text_encoder_lora_params, text_encoder_negation = inject_lora(
        use_text_lora, text_encoder, text_encoder_lora_modules,
        r=lora_rank, lora_path=lora_path
        )

    # Create parameters to optimize over with a condition (if "condition" is true, optimize it)
    optim_params = [
        param_optim(unet, trainable_modules is not None, extra_params=extra_unet_params, negation=unet_negation),

        param_optim(text_encoder, train_text_encoder and not use_text_lora, extra_params=extra_text_encoder_params,
                    negation=text_encoder_negation
                   ),
        param_optim(text_encoder_lora_params, use_text_lora, is_lora=True, extra_params={"lr": 1e-5}),
        param_optim(unet_lora_params, use_unet_lora, is_lora=True, extra_params={"lr": 1e-5}),
    ]

    params = create_optimizer_params(optim_params, learning_rate)
    
    # Create Optimizer
    optimizer = optimizer_cls(
        params,
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        weight_decay=adam_weight_decay,
        eps=adam_epsilon,
    )

    # Scheduler
    lr_scheduler = get_scheduler(
        lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
        num_training_steps=max_train_steps * gradient_accumulation_steps,
    )

    # Get the training dataset based on types (json, single_video, image)

    train_dataset= DADA2KS3(root_path=r"", interval=1, phase="train")

    val_dataset = DADA2KS3(root_path=r"", interval=1, phase="val")
    # val_dataset = RAADataset(origin_h5_file=r"/media/work/My Passport/h5/min-origin")
    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=2, shuffle=True,
        pin_memory=True,num_workers=4,drop_last=True)

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        pin_memory=True, drop_last=True)
    # DataLoaders creation:

    unet,optimizer,lr_scheduler, text_encoder= accelerator.prepare(
        unet,
        optimizer,
        lr_scheduler,
        text_encoder,
 )

    # Use Gradient Checkpointing if enabled.
    unet_and_text_g_c(
        unet,
        text_encoder, 
        gradient_checkpointing, 
        text_encoder_gradient_checkpointing
    )
    
    # Enable VAE slicing to save memory.
    vae.enable_slicing()

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = is_mixed_precision(accelerator)

    # Move text encoders, and VAE to GPU
    models_to_cast = [text_encoder,vae,p_net]
    cast_to_gpu_and_type(models_to_cast, accelerator, weight_dtype)

    if accelerator.is_main_process:
        accelerator.init_trackers("text2video-fine-tune")


    progress_bar = tqdm(range(global_step, max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    def finetune_unet(batch, train_encoder=False):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Check if we are training the text encoder
        text_trainable = (train_text_encoder or use_text_lora)

        # Unfreeze UNET Layers
        if global_step == 0:
            already_printed_trainables = False
            unet.train()
            handle_trainable_modules(
                unet,
                trainable_modules,
                is_enabled=True,
                negation=unet_negation
            )
        # Convert videos to latent space
        pixel_values = batch["pixel_values"].to(device,dtype=weight_dtype)
        mask_latents = batch["mask"].to(device,dtype=weight_dtype)
        # if not cache_latents:
        if cache_latents:
            latents = tensor_to_vae_latent(pixel_values, vae)
            mask_latents= tensor_to_vae_latent(mask_latents,vae)
        else:
            latents = pixel_values
            mask_latents=mask_latents
        # Get video length
        video_length = latents.shape[2]
        # Sample noise that we'll add to the latents
        noise = sample_noise(latents, offset_noise_strength, use_offset_noise)
        bsz = latents.shape[0]
        mask = (mask_latents >= 0.5).float()
        noise_mask =mask_latents * mask
        # Sample a random timestep for each video
        timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
        timesteps = timesteps.long()
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
        if text_trainable:
            text_encoder.train()
            if use_text_lora:
                text_encoder.text_model.embeddings.requires_grad_(True)
            if global_step == 0 and train_text_encoder:
                handle_trainable_modules(
                    text_encoder,
                    trainable_modules=trainable_text_modules,
                    negation=text_encoder_negation
            )
            cast_to_gpu_and_type([text_encoder], accelerator, torch.float32)
        # Fixes gradient checkpointing training.
        # See: https://github.com/prigoyal/pytorch_memonger/blob/master/tutorial/Checkpointing_for_PyTorch_models.ipynb
        if gradient_checkpointing or text_encoder_gradient_checkpointing:
            unet.eval()
            text_encoder.eval()
        selected_text = batch["prompt_ids"]
        texts=[]
        for i in range(len(selected_text)):
            text = tokenizer(
                selected_text[i], max_length=tokenizer.model_max_length, padding="max_length",
                truncation=True,
                return_tensors="pt"
            ).input_ids[0]
            texts.append(text)
        texts = torch.stack(texts).to(device)
        encoder_hidden_states=text_encoder(texts)[0]
        bbx=batch["bbx"].to(device)
        bbx = p_net(bbx)
        # Get the target for loss depending on the prediction type
        if noise_scheduler.prediction_type == "epsilon":
            target = noise
        elif noise_scheduler.prediction_type == "v_prediction":
            target = noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {noise_scheduler.prediction_type}")
        losses = []
        should_truncate_video = (video_length > 1 and text_trainable)
        # We detach the encoder hidden states for the first pass (video frames > 1)
        # Then we make a clone of the initial state to ensure we can train it in the loop.
        detached_encoder_state = encoder_hidden_states.clone().detach()
        trainable_encoder_state = encoder_hidden_states.clone()
        for i in range(2):
            should_detach = noisy_latents.shape[2] > 1 and i == 0
            if should_truncate_video and i == 1:
                noisy_latents = noisy_latents[:,:,1,:,:].unsqueeze(2)
                # noisy_latents_map = noisy_latents_map[:, :, 1, :, :].unsqueeze(2)
                target = target[:,:,1,:,:].unsqueeze(2)
                noise_mask=noise_mask[:,:,1,:,:].unsqueeze(2)
            bbx=shape_change(bbx)
            encoder_hidden_states = (
                detached_encoder_state if should_detach else trainable_encoder_state
            )
            l1=torch.nn.L1Loss()
            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states,bbx=bbx).sample
            reconstruction_mask=1-noise_mask

            loss1 = F.mse_loss(model_pred.float(), target.float(), reduction="mean")+ 0.3*l1(
                model_pred.float() * reconstruction_mask.float(), target.float() * reconstruction_mask.float())
            losses.append(loss1)

            if video_length == 1 and i == 0: break

        loss = losses[0] if len(losses) == 1 else losses[0] + losses[1]

        return loss, latents

    writer=SummaryWriter(log_dir=r"log_new1")
    for epoch in range(first_epoch, 20):
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue
            with accelerator.accumulate(unet) ,accelerator.accumulate(text_encoder):
                with accelerator.autocast():
                    loss, latents = finetune_unet(batch, train_encoder=train_text_encoder)

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(train_batch_size)).mean()
                train_loss += avg_loss.item() / gradient_accumulation_steps

                # Backpropagate
                try:
                    accelerator.backward(loss)
                    params_to_clip = (
                        unet.parameters() if not train_text_encoder
                    else
                        list(p_net.parameters()) +list(unet.parameters()) + list(text_encoder.parameters())
                        # list(unet.parameters()) + list(text_model.parameters())
                    )
                    accelerator.clip_grad_norm_(params_to_clip, max_grad_norm)

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                except Exception as e:
                    print(f"An error has occured during backpropogation! {e}")
                    continue
            print(global_step,"loss:",loss)
            writer.add_scalar("loss",loss.detach(),global_step)
            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

            if global_step % checkpointing_steps == 0:
                save_pipe(
                    pretrained_model_path,
                    global_step,
                    accelerator,
                    unet,
                    text_encoder,
                    vae,
                    output_dir,
                    use_unet_lora,
                    use_text_lora,
                    unet_lora_modules,
                    text_encoder_lora_modules,
                    is_checkpoint=True,
                )
                save_dir = os.path.join(r""+ "/"+f"{global_step}_bbx.pt")
                accelerator.save(p_net.state_dict(),save_dir)

    if global_step % validation_steps == 0:
        if accelerator.is_main_process:
            with accelerator.autocast():
                unet.eval()
                text_encoder.eval()
                p_net.eval()
                unet_and_text_g_c(unet, text_encoder, False, False)

                pretrained_model_path=r"your trained model path"

                pipeline = TextToVideoSDPipeline.from_pretrained(
                    pretrained_model_path,
                    text_encoder=text_encoder,
                    vae=vae,
                    unet=unet
                )

                p_net.load_state_dict(torch.load(r"your trained p_net path"),strict=False)
                diffusion_scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
                pipeline.scheduler = diffusion_scheduler

                for idx, batch in enumerate(val_dataloader):
                    prompts = batch["prompt_ids"]
                    latents=batch["pixel_values"]
                    num_frames = latents.shape[2]
                    latents = rearrange(latents, "b c f h w -> (b f) c h w")
                    latents = vae.encode(latents).latent_dist.sample()
                    latents = rearrange(latents, "(b f) c h w -> b c f h w", f=num_frames)
                    id1 = str(idx)
                    save_folder = os.path.join(output_dir,id1)
                    if not os.path.exists(save_folder):
                        os.makedirs(save_folder)
                    bbx = batch["bbx"].to(device=accelerator.device)
                    bbx = p_net(bbx)

                    with torch.no_grad():
                        videos = pipeline(
                            prompt=prompts,
                            negative_prompt=None,
                            num_frames=16,
                            height=224,
                            width=224,
                            num_inference_steps=50,
                            guidance_scale=9.0,
                            output_type="pt",
                            latents=latents.to(device=accelerator.device,dtype=torch.half),
                            bbx=bbx,
                        )

                        for video in videos:
                            video = rearrange(video, "c f h w -> f h w c").clamp(-1, 1).add(1).mul(127.5)
                            video = video.byte().cpu().numpy()
                            for f, frame in enumerate(video):
                                frame_path = os.path.join(save_folder, f"frame_{f + 1}.jpg")
                                cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

                del pipeline
                torch.cuda.empty_cache()
                unet_and_text_g_c(
                    unet,
                    text_encoder,
                    gradient_checkpointing,
                    text_encoder_gradient_checkpointing
                )

    accelerator.wait_for_everyone()
    accelerator.end_training()

# writer.close()
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/lora_training_config.yaml")
    args = parser.parse_args()
    main(**OmegaConf.load(args.config))

