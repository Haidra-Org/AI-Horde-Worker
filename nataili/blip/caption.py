from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import torch
from nataili.util.voodoo import performance

class Caption:
    def __init__(self,
        model,
        device
    ):
        self.model = model
        self.device = device
    
    @performance
    def __call__(self, image):
        gpu_image = transforms.Compose([
            transforms.Resize((512, 512), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ])(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            caption = self.model.generate(gpu_image, max_length=20, min_length=5, num_beams=3)[0]
        return caption