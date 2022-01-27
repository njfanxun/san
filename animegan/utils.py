import cv2
import os
from basicsr.utils import imwrite
import numpy
import numpy as np
import torch
from basicsr.utils import img2tensor, tensor2img
from facexlib.utils.face_restoration_helper import FaceRestoreHelper
from torchvision.transforms.functional import normalize
from animegan.animegan import AnimeGAN2
from torchvision.transforms.functional import to_tensor, to_pil_image


class AnimeGANer:

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
    def enhanceFace(self, img, ):
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
            # paste_back
            self.face_helper.get_inverse_affine()
            face_restored_img = self.paste_faces_to_input_image()
            return face_restored_img
        else:
            return None

    def enhance(self, img, ):
        out = self.process(img)
        restored_img = tensor2img(out.squeeze(0), rgb2bgr=True, min_max=(0, 1)).astype('uint8')
        return restored_img

    def process(self, input_image) -> torch.Tensor:
        img = img2tensor(input_image, bgr2rgb=True, float32=True)
        normalize(img, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
        img = img.unsqueeze(0) * 2 - 1
        out = self.animegan(img.to(self.device), False).cpu()
        out = out.squeeze(0).clip(-1, 1) * 0.5 + 0.5
        return out

    def paste_faces_to_input_image(self, ):
        h, w, _ = self.face_helper.input_img.shape
        h_up, w_up = int(h * self.face_helper.upscale_factor), int(w * self.face_helper.upscale_factor)

        bg_img = np.zeros((h_up, w_up, 4))
        bg_img = cv2.resize(bg_img, (w_up, h_up), interpolation=cv2.INTER_LANCZOS4)
        for restored_face, inverse_affine in zip(self.face_helper.restored_faces,
                                                 self.face_helper.inverse_affine_matrices):
            # Add an offset to inverse affine matrix, for more precise back alignment
            if self.upscale > 1:
                extra_offset = 0.5 * self.upscale
            else:
                extra_offset = 0
            inverse_affine[:, 2] += extra_offset
            inv_restored = cv2.warpAffine(restored_face, inverse_affine, (w_up, h_up), )
            mask = np.ones(self.face_helper.face_size, dtype=np.float32)
            inv_mask = cv2.warpAffine(mask, inverse_affine, (w_up, h_up))
            # remove the black borders
            inv_mask_erosion = cv2.erode(
                inv_mask,
                np.ones((int(2 * self.face_helper.upscale_factor), int(2 * self.face_helper.upscale_factor)), np.uint8))
            pasted_face = inv_mask_erosion[:, :, None] * inv_restored

            img2gray = cv2.cvtColor(pasted_face, cv2.COLOR_BGR2GRAY)
            __, thresh = cv2.threshold(img2gray, 1, 255, cv2.THRESH_BINARY)
            b, g, r = cv2.split(pasted_face)
            rgba = [b, g, r, thresh]
            bg_img = cv2.merge(rgba, 4)
            bg_img = cv2.resize(bg_img, None, fx=1 / self.upscale, fy=1 / self.upscale, interpolation=cv2.INTER_LINEAR)
        if np.max(bg_img) > 256:  # 16-bit image
            bg_img = bg_img.astype(np.uint16)
        else:
            bg_img = bg_img.astype(np.uint8)

        return bg_img
