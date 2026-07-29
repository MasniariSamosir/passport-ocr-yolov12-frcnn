import torchvision.transforms as T

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self, image):
        for t in self.transforms:
            image = t(image)
        return image
