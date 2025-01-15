import torchio as tio


"""
Standard class with some augmentations from torchio
"""
class Augmenter:
    def __init__(self,):
        self.augmentations = tio.Compose(self.get_augmentations())

    def get_augmentations(self):
        augs = [
            tio.RandomAffine(p=0.5),
            tio.RandomGamma(p=0.1),
            tio.RandomNoise(p=0.1),
            tio.RandomBlur(p=0.1),
            tio.RandomFlip(axes=(0,1,2), p=0.2) # does this make sense? i mean i removed left/right stuff, should be ok for a pretraining
        ]
        return augs
    
    def augment(self, image, label):
        # check that this does not makes copy of the data
        image = tio.ScalarImage(tensor=image)
        label = tio.LabelMap(tensor=label)
        subject = tio.Subject(image=image, label=label)
        subject = self.augmentations(subject)
        return subject["image"][tio.DATA], subject["label"][tio.DATA]