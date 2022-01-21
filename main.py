import cv2
import glob
from PIL import Image
import os
import torch
from animegan.utils import AnimeGANer
from torchvision.transforms.functional import to_tensor, to_pil_image
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from gfpgan import GFPGANer
import ssl
from basicsr.utils import imwrite
import options

ssl._create_default_https_context = ssl._create_unverified_context

torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
opt = options.Options()


def load_image(image_path, x32=False):
    img = Image.open(image_path).convert("RGB")

    if x32:
        def to_32s(x):
            return 256 if x < 256 else x - x % 32

        w, h = img.size
        img = img.resize((to_32s(w), to_32s(h)))

    return img


def runGFP():
    os.makedirs(opt.output_dir, exist_ok=True)
    # background upsampler
    bg_upsampler = RealESRGANer(
        scale=opt.upscale,
        model_path=opt.realesr_model_path,
        model=RRDBNet(num_in_ch=3,
                      num_out_ch=3,
                      num_feat=64,
                      num_block=23,
                      num_grow_ch=32,
                      scale=2),
        tile=opt.bg_tile,
        tile_pad=0,
        pre_pad=0,
        half=False)  # need to set False in CPU mode

    # set up GFPGAN restorer
    restorer = GFPGANer(model_path=opt.gfp_model_path,
                        upscale=opt.upscale,
                        arch=opt.arch,
                        channel_multiplier=opt.channel,
                        bg_upsampler=bg_upsampler)

    img_list = sorted(glob.glob(os.path.join(opt.input_dir, '*')))
    for img_path in img_list:
        # read image
        img_name = os.path.basename(img_path)
        print(f'GFPGan Processing {img_name} ...')
        basename, ext = os.path.splitext(img_name)
        input_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if opt.ext == 'auto':
            extension = ext[1:]
        else:
            extension = opt.ext
        # restore faces and background if necessary
        __, __, restored_img = restorer.enhance(
            input_img,
            has_aligned=opt.aligned,
            only_center_face=opt.only_center_face,
            paste_back=opt.paste_back)
        # save restored img
        if restored_img is not None:
            save_restore_path = os.path.join(opt.output_dir, f'{basename}_r.{extension}')
            imwrite(restored_img, save_restore_path)
            print(f'Results are in the [{save_restore_path}]')
            return save_restore_path
        return None


def runAnime(path: str):
    animer = AnimeGANer(opt.anime_model_path, upscale=opt.upscale)

    # read image
    img_name = os.path.basename(path)
    print(f'AnimeGan Processing {img_name} ...')
    basename, ext = os.path.splitext(img_name)
    if opt.ext == 'auto':
        extension = ext[1:]
    else:
        extension = opt.ext
    input_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    face_img = animer.enhanceFace(input_img)
    if face_img is not None:
        save_face_path = os.path.join(opt.output_dir, f'anime_face_{basename}.png')
        imwrite(face_img, save_face_path)
        print(f'Results are in the [{save_face_path}]')
    restored_img = animer.enhance(input_img)
    if restored_img is not None:
        save_anime_path = os.path.join(opt.output_dir, f'anime_full_{basename}.{extension}')
        imwrite(restored_img, save_anime_path)
        print(f'Results are in the [{save_anime_path}]')


if __name__ == '__main__':
    img_path = runGFP()
    if img_path is not None:
        runAnime(img_path)
