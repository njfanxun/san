import os.path


class Options(object):
    @property
    def upscale(self) -> int:
        return 4

    @property
    def arch(self) -> str:
        return 'clean'

    @property
    def channel(self) -> int:
        return 2

    @property
    def gfp_model_path(self) -> str:
        return 'weights/GFPGANCleanv1-NoCE-C2.pth'

    @property
    def realesr_model_path(self) -> str:
        return 'weights/RealESRGAN_x2plus.pth'

    @property
    def bg_tile(self) -> int:
        return 0

    @property
    def input_dir(self) -> str:
        return './inputs'

    @property
    def only_center_face(self) -> bool:
        return True

    @property
    def aligned(self) -> bool:
        return False

    @property
    def paste_back(self) -> bool:
        return True

    @property
    def output_dir(self) -> str:
        return './outputs'
    @property
    def ext(self) -> str:
        return 'auto'

    @property
    def device(self) -> str:
        return 'cpu'

    @property
    def anime_model_path(self) -> str:
        return 'weights/face_paint_512_v2.pt'
