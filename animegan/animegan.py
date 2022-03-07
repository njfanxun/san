import cv2
import os
from basicsr.utils import imwrite
import numpy
import numpy as np
import torch
from PIL import Image
from animegan.model import Generator
from animegan.face_detector import get_dlib_face_detector
from torchvision.transforms.functional import to_tensor, to_pil_image


class AnimeGANer:

    def __init__(self, model_path, face_model_path, upscale=2):
        super().__init__()
        self.upscale = upscale
        # initialize model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gen = Generator()
        self.gen.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.gen.to(self.device).eval()
        self.face_detector = get_dlib_face_detector(face_model_path)

    @torch.no_grad()
    def enhance(self, image: Image):
        image_rgb = image.convert("RGB")
        landmarks = self.face_detector(image_rgb)
        landmarks = np.array(landmarks)
        mean_face_size = (landmarks.max(axis=1) - landmarks.min(axis=1)).mean()

        resize_ratio = 256 / max(mean_face_size, 32)
        image_inference_size = (np.array(image.size) * resize_ratio).astype(int)
        image_inference_size -= image_inference_size % 4

        image_rgb = image_rgb.resize(image_inference_size, Image.LANCZOS)

        # Model inference
        model_input = to_tensor(image_rgb).unsqueeze(0) * 2 - 1
        with torch.inference_mode():
            image_output = self.gen(model_input.to(self.device))
        image_output = (image_output * 0.5 + 0.5).clip(0, 1)
        image_output = to_pil_image(image_output.cpu()[0])

        # Resize image back to the original size & mode
        image_output = image_output.resize(image.size, Image.LANCZOS).convert(image.mode)
        if image.mode == "RGBA":  # <<<<< THIS PART HANDLES WHAT YOU WANTED
            alpha = image.split()[-1]
            image_output.putalpha(alpha)

        return image_output
