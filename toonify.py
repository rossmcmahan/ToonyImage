import torch
from torchvision import transform
from PIL import Image
from toonify import Toonify

model = Toonify()

image = Image.open('meg.JPEG')
preprocess = transform.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
])

input_tensor = preprocess(image).unsqueeze(0)

with torch.no_grad():
    toonified_image = model(input_tensor)

output_image = transform.ToPILImage()(toonified_image.squeeze())

output_image.save('toonified_photo.JPEG')