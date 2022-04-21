import random
from PIL import ImageFilter, ImageOps


class Solarization(object):

    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


class GaussianBlur(object):

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class MultiCropTransform:

    def __init__(self, trans_list):
        self.trans_list = trans_list

    def __call__(self, x):
        x_out = []
        for trans in self.trans_list:
            x_out.append(trans(x))
        return x_out