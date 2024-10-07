# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path

import os
import sys
import subprocess

import gc
import time
import random

import torch
import torch.nn.functional as F
import numpy as np

import cv2
import PIL
from PIL import Image

from diffusers import TCDScheduler
from diffusers.models.model_loading_utils import load_state_dict

from safetensors.torch import load_file

from transformers import AutoModelForCausalLM

from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
from deepseek_vl.utils.io import load_pil_images


# Allow faster download
os.environ["HF_HUB_ENABLE_HF_TRANSFER"]="1"


# Deal with PIL.Image.DecompressionBombError
Image.MAX_IMAGE_PIXELS = None


# GPU global variables
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if str(DEVICE).__contains__("cuda") else torch.float32


# AI global variables
TOTAL_CACHE = "./cache"
MODEL_ID = "deepseek-ai/deepseek-vl-1.3b-chat"


def flush():
    gc.collect()
    torch.cuda.empty_cache()


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.load_model()


    def load_model(self):
        print("[~] Setup pipeline")
        # 1. Setup pipeline
        self.processor : VLChatProcessor = VLChatProcessor.from_pretrained(TOTAL_CACHE)
        self.tokenizer = self.processor.tokenizer
        
        self.model : MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(TOTAL_CACHE, trust_remote_code=True)
        self.model = self.model.to(device=DEVICE, dtype=DTYPE)


    @torch.inference_mode()
    def fill_image(
        self,
        image,
        mask_image,
        control_image,
        prompt,
        negative_prompt,
        width,
        height,
        num_steps,
        num_outputs,
        guidance_scale,
        controlnet_scale,
        seed,
    ):
        flush()
        
        # generator_list = []
        # for i in range(num_outputs):
        #     generator = torch.Generator(device=DEVICE).manual_seed(seed+i)
        #     generator_list.append(generator)
        generator = torch.Generator(device=DEVICE).manual_seed(seed)
        
        # Convert prompt, negative_prompt to embeddings
        # conditioning, pooled = self.compel(prompt)
        # neg_conditioning, neg_pooled = self.compel(negative_prompt)

        print("Start inference...")
        print(f"[Debug] Prompt: {prompt}")
        
        image_list = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            mask_image=mask_image,
            control_image_list=[0, 0, 0, 0, 0, 0, 0, control_image],
            width=width,
            height=height,
            num_images_per_prompt=num_outputs,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            union_control=True,
            union_control_type=torch.Tensor([0, 0, 0, 0, 0, 0, 0, controlnet_scale]),
            generator=generator,
        ).images

        return image_list


    @torch.inference_mode()
    def predict(
        self,
        image: Path = Input(
            description="Input image for inpainting.",
            default=None,
        ),
        prompt: str = Input(
            description="Input prompt.",
            default=None,
        ),
        sample: bool = Input(
            description="Whether do sample or not.",
            default=False,
        ),
        top_k: int = Input(
            description="Top k, if sample is enabled.",
            default=0.5,
            ge=0,
            le=1,
        ),
        top_p: float = Input(
            description="Top p, if sample is enabled.",
            default=3,
            ge=0,
            le=50,
        ),
        temperature: float = Input(
            description="Temperature, if sample is enabled.",
            default=0.1,
            ge=0,
            le=1,
        ),
        max_tokens: int = Input(
            description="Maximum number of tokens.",
            default=512,
            ge=256,
            le=512,
        ),
    ) -> str:
        """Run a single prediction on the model"""
        start1 = time.time()
        
        # if init_image is None or mask_image is None:
        if image is None or prompt is None:
            msg = "No input, Save money"
            return msg

        else:
            print(f"DEVICE: {DEVICE}")
            print(f"DTYPE: {DTYPE}")

            # Set prompt for template
            conversation = [
                {
                    "role": "User",
                    "content": f"<image_placeholder>{prompt}",
                    "images": [str(image)]
                },
                {
                    "role": "Assistant",
                    "content": ""
                }
            ]
            
            # load images and prepare for inputs
            pil_images = load_pil_images(conversation)
            prepare_inputs = self.processor(
                conversations=conversation,
                images=pil_images,
                force_batchify=True
            ).to(self.model.device)
            
            # run image encoder to get the image embeddings
            inputs_embeds = self.model.prepare_inputs_embeds(
                **prepare_inputs
            )
            
            print("[~] Finish setup in " + str(time.time()-start1) + " secs.")
            
            start2 = time.time()
            
            # run the model to get the response
            if sample is True:
                outputs = self.model.language_model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=prepare_inputs.attention_mask,
                    pad_token_id=self.tokenizer.eos_token_id,
                    bos_token_id=self.tokenizer.bos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    top_k=top_k,
                    top_p=top_p,
                    temperature=temperature,
                    use_cache=True
                )
            else:
                outputs = self.model.language_model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=prepare_inputs.attention_mask,
                    pad_token_id=self.tokenizer.eos_token_id,
                    bos_token_id=self.tokenizer.bos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    use_cache=True
                )
            
            answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
            
            print("[~] Finish generation in " + str(time.time()-start2) + " secs.")
            
            return answer