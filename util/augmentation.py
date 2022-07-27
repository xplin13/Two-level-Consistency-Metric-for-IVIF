import numpy as np
import cv2


class RandomFlip():
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, label):
        if np.random.rand() < self.prob:
            image = image[:,::-1]
            label = label[:,::-1]
        return image, label

class RandomFlip_three():
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, image_ir):
        if np.random.rand() < self.prob:
            image = image[:,::-1]
            image_ir = image_ir[:,::-1]

        return image, image_ir

class RandomCrop():
    def __init__(self, crop_rate=0.1, prob=1.0):
        self.crop_rate = crop_rate
        self.prob      = prob

    def __call__(self, image, label):
        if np.random.rand() < self.prob:
            w, h, c = image.shape

            h1 = np.random.randint(0, h*self.crop_rate)
            w1 = np.random.randint(0, w*self.crop_rate)
            h2 = np.random.randint(h-h*self.crop_rate, h+1)
            w2 = np.random.randint(w-w*self.crop_rate, w+1)

            image = image[w1:w2, h1:h2]
            label = label[w1:w2, h1:h2]

        return image, label



class RandomCrop_three():
    def __init__(self, crop_rate=0.1, prob=1.0):
        self.crop_rate = crop_rate
        self.prob      = prob

    def __call__(self, image, image_ir, label):
        if np.random.rand() < self.prob:
            w, h  = image.shape

            h1 = np.random.randint(0, h*self.crop_rate)
            w1 = np.random.randint(0, w*self.crop_rate)
            h2 = np.random.randint(h-h*self.crop_rate, h+1)
            w2 = np.random.randint(w-w*self.crop_rate, w+1)

            image = image[w1:w2, h1:h2]
            image_ir = image_ir[w1:w2, h1:h2]
            label = label[w1:w2, h1:h2]

        return image, image_ir, label


class RandomScale2_three():
    def __init__(self, low=0.8, high=1.2, prob=1.0):
        self.low = low
        self.high = high
        self.prob = prob

    def __call__(self, image, image_ir, label):
        scale = np.random.uniform(self.low,self.high)
        h, w = image.shape
        image = cv2.resize(image,(int(w*scale),int(h*scale)))
        image_ir = cv2.resize(image_ir,(int(w*scale),int(h*scale)))
        label = cv2.resize(label,(int(w*scale),int(h*scale)),cv2.INTER_NEAREST)
        return image, image_ir, label



class RandomCrop2_three():
    def __init__(self, crop_w = 256,crop_h=256, prob=1.0):
        self.crop_h = crop_w
        self.crop_w = crop_h
        self.prob      = prob

    def __call__(self, image, image_ir):

        if np.random.rand() < self.prob:
            h, w  = image.shape
            if self.crop_h > h or self.crop_w > w:
                image = cv2.resize(image, (self.crop_w,self.crop_h))
                image_ir = cv2.resize(image_ir, (self.crop_w,self.crop_h))
                return image, image_ir
            else:

                h1 = np.random.randint(0, h-self.crop_h)
                w1 = np.random.randint(0, w-self.crop_w)


                image = image[h1:h1+self.crop_h, w1:w1+self.crop_w]
                image_ir = image_ir[h1:h1+self.crop_h, w1:w1+self.crop_w]

                return image, image_ir

class RandomCropOut():
    def __init__(self, crop_rate=0.2, prob=1.0):
        self.crop_rate = crop_rate
        self.prob      = prob

    def __call__(self, image, label):
        if np.random.rand() < self.prob:
            w, h, c = image.shape

            h1 = np.random.randint(0, h*self.crop_rate)
            w1 = np.random.randint(0, w*self.crop_rate)
            h2 = int(h1 + h*self.crop_rate)
            w2 = int(w1 + w*self.crop_rate)

            image[w1:w2, h1:h2] = 0
            label[w1:w2, h1:h2] = 0

        return image, label


class RandomBrightness():
    def __init__(self, bright_range=0.15, prob=0.9):
        self.bright_range = bright_range
        self.prob = prob

    def __call__(self, image, label):
        if np.random.rand() < self.prob:
            bright_factor = np.random.uniform(1-self.bright_range, 1+self.bright_range)
            image = (image * bright_factor).astype(image.dtype)

        return image, label


class RandomNoise():
    def __init__(self, noise_range=5, prob=0.9):
        self.noise_range = noise_range
        self.prob = prob

    def __call__(self, image, label):
        if np.random.rand() < self.prob:
            w, h, c = image.shape

            noise = np.random.randint(
                -self.noise_range,
                self.noise_range,
                (w,h,c)
            )

            image = (image + noise).clip(0,255).astype(image.dtype)

        return image, label
        


