import cv2
import glob
from PIL import Image
import os
import torch
from animegan import Generator
from torchvision.transforms.functional import to_tensor, to_pil_image
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from gfpgan import GFPGANer
import ssl
import options

ssl._create_default_https_context = ssl._create_unverified_context


# torch.backends.cudnn.enabled = False
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True


def main():
    opt = options.Options()
    os.makedirs(opt.output_dir, exist_ok=True)
    # background upsampler
    bg_upsampler = RealESRGANer(
        scale=4,
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

    anime = Generator(opt.anime_model_path)

    img_list = sorted(glob.glob(os.path.join(opt.input_dir, '*')))
    for img_path in img_list:
        # read image
        img_name = os.path.basename(img_path)
        print(f'Processing {img_name} ...')
        basename, ext = os.path.splitext(img_name)
        input_img = cv2.imread(img_path, cv2.IMREAD_COLOR)

        # restore faces and background if necessary
        _, _, restored_img = restorer.enhance(
            input_img,
            has_aligned=opt.aligned,
            only_center_face=opt.only_center_face,
            paste_back=opt.paste_back)

        if opt.ext == 'auto':
            extension = ext[1:]
        else:
            extension = opt.ext

        # save restored img
        if restored_img is not None:
            image = Image.fromarray(cv2.cvtColor(restored_img, cv2.COLOR_BGR2RGB)).convert("RGB")
            with torch.no_grad():
                image = to_tensor(image).unsqueeze(0) * 2 - 1
                out = anime(image.to(opt.device), False).cpu()
                out = out.squeeze(0).clip(-1, 1) * 0.5 + 0.5
                out = to_pil_image(out)
            save_restore_path = os.path.join(opt.output_dir, f'{basename}_r.{extension}')
            out.save(save_restore_path)
            print(f'Results are in the [{save_restore_path}]')


if __name__ == '__main__':
    main()
