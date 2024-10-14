# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path

import os
import sys
import subprocess

import gc
import time
import random
import requests

import torch
import torch.nn.functional as F
import numpy as np

import cv2
import PIL
from PIL import Image, ImageStat

from safetensors.torch import load_file

from transformers import AutoModel, AutoTokenizer


# Allow faster download
os.environ["HF_HUB_ENABLE_HF_TRANSFER"]="1"


# Deal with PIL.Image.DecompressionBombError
Image.MAX_IMAGE_PIXELS = None


# GPU global variables
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if str(DEVICE).__contains__("cuda") else torch.float32


# AI global variables
TOTAL_CACHE = "./cache"
MODEL_ID = "openbmb/MiniCPM-V-2_6"


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
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID, 
            trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            MODEL_ID, 
            trust_remote_code=True,
            attn_implementation='sdpa', 
            torch_dtype=DTYPE
        ) # sdpa or flash_attention_2, no eager
        self.model = self.model.eval().cuda()


    @torch.inference_mode()
    def predict(
        self,
        image: Path = Input(
            description="Input image.",
            default=None,
        ),
        prompt: str = Input(
            description="Input prompt.",
            default="Describe this image.",
        ),
        # sample: bool = Input(
        #     description="Whether do sample or not.",
        #     default=False,
        # ),
        # top_k: int = Input(
        #     description="Top k, if sample is enabled.",
        #     default=5,
        #     ge=0,
        #     le=50,
        # ),
        # top_p: float = Input(
        #     description="Top p, if sample is enabled.",
        #     default=0.5,
        #     ge=0,
        #     le=1,
        # ),
        # temperature: float = Input(
        #     description="Temperature, if sample is enabled.",
        #     default=0.3,
        #     ge=0,
        #     le=1,
        # ),
        max_tokens: int = Input(
            description="Maximum number of tokens.",
            default=200,
            ge=0,
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
            
            # Convert the image to grayscale to calculate brightness
            original_image = Image.open(str(image))
            gray_image = original_image.convert('L')  # Convert to grayscale

            # Calculate the average brightness
            stat = ImageStat.Stat(gray_image)
            average_brightness = stat.mean[0]  # Get the average value

            # Define background color based on brightness (threshold can be adjusted)
            bg_color = (0, 0, 0) if average_brightness > 127 else (255, 255, 255)

            # Create a new image with the same size as the original, filled with the background color
            new_image = Image.new('RGB', original_image.size, bg_color)

            # Paste the original image on top of the background (use image as a mask if needed)
            new_image.paste(original_image, (0, 0), original_image if original_image.mode == 'RGBA' else None)

            # Set inputs
            inputs = [
                {
                    'role': 'user', 'content': [new_image, prompt]
                }
            ]
            
            print("[~] Finish setup in " + str(time.time()-start1) + " secs.")
            
            start2 = time.time()
            
            # generate output; maximum 200 new tokens; stop generation when <|endoftext|> is generated
            with torch.autocast(device_type=DEVICE, enabled=True, dtype=DTYPE):
                answer = self.model.chat(
                    image=None,
                    msgs=inputs,
                    tokenizer=self.tokenizer
                )

            # delete blank in both sides
            answer = answer.strip()
            
            print("[~] Finish generation in " + str(time.time()-start2) + " secs.")
            
            return answer