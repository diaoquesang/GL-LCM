from config import config
from torchvision import transforms
import cv2 as cv



class myTransformMethod():
    def __call__(self, img):

        img = cv.resize(img, (config.image_size, config.image_size))
        if img.shape[-1] == 3:  # HWC
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        return img


myTransform = {
    'trainTransform': transforms.Compose([
        myTransformMethod(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ]),
    'testTransform': transforms.Compose([
        myTransformMethod(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ]),

}
