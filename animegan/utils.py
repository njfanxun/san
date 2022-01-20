import cv2
import os

import numpy
import numpy as np
import torch
from basicsr.utils import img2tensor, tensor2img
from facexlib.utils.face_restoration_helper import FaceRestoreHelper
from torchvision.transforms.functional import normalize
from animegan.animegan import AnimeGAN2
from torchvision.transforms.functional import to_tensor, to_pil_image


class AnimeGANer():

    def __init__(self, model_path, upscale=2):
        super().__init__()
        self.upscale = upscale
        # initialize model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.animegan = AnimeGAN2()
        self.face_helper = FaceRestoreHelper(
            upscale,
            face_size=512,
            crop_ratio=(1, 1),
            det_model='retinaface_resnet50',
            save_ext='jpg',
            device=self.device)
        self.animegan.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.animegan.to(self.device).eval()

    @torch.no_grad()
    def enhance(self, img, ):
        self.face_helper.clean_all()
        self.face_helper.read_image(img)
        self.face_helper.get_face_landmarks_5(only_center_face=True, eye_dist_threshold=5)
        self.face_helper.align_warp_face()
        if len(self.face_helper.cropped_faces) > 0:
            cropped_face = self.face_helper.cropped_faces[0]
            # cropped_face_t  'torch.Tensor'
            output = self.process(cropped_face)
            restored_face = tensor2img(output.squeeze(0), rgb2bgr=True, min_max=(0, 1)).astype('uint8')
            self.face_helper.add_restored_face(restored_face)
            # # paste_back
            #
            bg_img = self.process(img)
            bg_img = tensor2img(bg_img.squeeze(0), rgb2bgr=True, min_max=(0, 1)).astype('uint8')
            self.face_helper.get_inverse_affine(None)
            restored_img = self.restore_faces(face_img=restored_face)
            return restored_img
        else:
            return None

    def process(self, input_image) -> torch.Tensor:
        img = img2tensor(input_image / 255., bgr2rgb=True, float32=True)
        normalize(img, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
        img = img.unsqueeze(0) * 2 - 1
        out = self.animegan(img.to(self.device), False).cpu()
        out = out.squeeze(0).clip(-1, 1) * 0.5 + 0.5
        return out


