import time

import cv2
import glob

import numpy
from PIL import Image
import os
import torch
from animegan import AnimeGANer
from torchvision.transforms.functional import to_tensor, to_pil_image
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from gfpgan import GFPGANer
import ssl
from basicsr.utils import imwrite
import options
from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler

ssl._create_default_https_context = ssl._create_unverified_context
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

opt = options.Options()


def get_time(f):
    def inner(*arg, **kwarg):
        s_time = time.time()
        res = f(*arg, **kwarg)
        e_time = time.time()
        print('耗时：{}秒'.format(e_time - s_time))
        return res

    return inner


class FileWatchHandler(PatternMatchingEventHandler):

    def __init__(self, patterns=None, ignore_patterns=None, ignore_directories=False, case_sensitive=False):
        super().__init__(patterns, ignore_patterns, ignore_directories, case_sensitive)
        self.restorer = None
        self.animeGen = None
        self.restorer_image = 'gfp_'
        self.anime_image = 'anime_'

    def prepareGANs(self, ):
        # background upsampler
        net = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        bg_upsampler = RealESRGANer(scale=opt.upscale, model_path=opt.realesr_model_path,
                                    model=net, tile=opt.bg_tile, tile_pad=0, pre_pad=0, half=False)

        self.restorer = GFPGANer(model_path=opt.gfp_model_path, upscale=opt.upscale, arch=opt.arch,
                                 channel_multiplier=opt.channel, bg_upsampler=bg_upsampler)
        self.animeGen = AnimeGANer(opt.anime_model_path, opt.face_model_path, upscale=opt.upscale)
        print('完成对抗网络模型加载...')

    def restorer_img_path(self, img_name: str):
        basename, ext = os.path.splitext(img_name)
        if opt.ext == 'auto':
            extension = ext[1:]
        else:
            extension = opt.ext
        return os.path.join(opt.output_dir, f'{self.restorer_image}{basename}.{extension}'), extension

    def anime_img_path(self, img_name: str):
        basename, ext = os.path.splitext(img_name)
        if opt.ext == 'auto':
            extension = ext[1:]
        else:
            extension = opt.ext
        return os.path.join(opt.output_dir, f'{self.anime_image}{basename}.{extension}'), extension

    @get_time
    def gfp_process(self, img_path: str) -> (bool, str):
        img_name = os.path.basename(img_path)
        input_img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        __, __, restored_img = self.restorer.enhance(input_img, has_aligned=opt.aligned,
                                                     only_center_face=opt.only_center_face, paste_back=opt.paste_back)
        if restored_img is not None:
            restored_img_path, _ = self.restorer_img_path(img_name)
            imwrite(restored_img, restored_img_path)
            print(f'完成图像增强:{img_name}')
            return True, restored_img_path
        else:
            print(f'图像增强处理失败:{img_path}')
            return False, None

    @get_time
    def anime_process(self, img_path: str):
        img_name = os.path.basename(img_path)

        image = Image.open(img_path)
        full_img = self.animeGen.enhance(image)
        if full_img is not None:
            save_full_path, ext = self.anime_img_path(img_name)
            full_img.save(save_full_path, ext)

        print(f'完成漫画风格化{img_name}')

    def on_created(self, event):
        print('===============================================================================')
        print(f'开始处理图像:{event.src_path}')
        success, img_path = self.gfp_process(event.src_path)
        if success:
            self.anime_process(img_path)
        print('===============================================================================')

    def on_deleted(self, event):
        img_name = os.path.basename(event.src_path)
        restorer_img_path, _ = self.restorer_img_path(img_name=img_name)
        img_name = os.path.basename(restorer_img_path)
        anime_full_img_path, _ = self.anime_img_path(img_name=img_name)
        if os.path.exists(restorer_img_path):
            os.remove(restorer_img_path)
        if os.path.exists(anime_full_img_path):
            os.remove(anime_full_img_path)


if __name__ == '__main__':
    os.makedirs(opt.output_dir, exist_ok=True)

    event_handler = FileWatchHandler(patterns=['*.jpg', '*.png'], ignore_patterns=None,
                                     ignore_directories=True, case_sensitive=True)
    event_handler.prepareGANs()
    observer = Observer()
    observer.schedule(event_handler, opt.input_dir, recursive=False)
    observer.start()
    print("开始监视文件夹:%s" % opt.input_dir)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
