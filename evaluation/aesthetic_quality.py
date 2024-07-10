import os
import torch
import torch.nn as nn
from os.path import expanduser  # pylint: disable=import-outside-toplevel
from urllib.request import urlretrieve  # pylint: disable=import-outside-toplevel
import subprocess
import numpy as np
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, ToPILImage
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
    BILINEAR = InterpolationMode.BILINEAR
except ImportError:
    BICUBIC = Image.BICUBIC
    BILINEAR = Image.BILINEAR

def get_aesthetic_model(cache_folder):
    """load the aethetic model"""
    path_to_model = cache_folder + "/aesthetic_model/sa_0_4_vit_l_14_linear.pth"
    if not os.path.exists(path_to_model):
        os.makedirs(cache_folder, exist_ok=True)
        url_model = (
            "https://github.com/LAION-AI/aesthetic-predictor/blob/main/sa_0_4_vit_l_14_linear.pth?raw=true"
        )
        # download aesthetic predictor
        if not os.path.isfile(path_to_model):
            try:
                print(f'trying urlretrieve to download {url_model} to {path_to_model}')
                urlretrieve(url_model, path_to_model) # unable to download https://github.com/LAION-AI/aesthetic-predictor/blob/main/sa_0_4_vit_l_14_linear.pth?raw=true to pretrained/aesthetic_model/emb_reader/sa_0_4_vit_l_14_linear.pth 
            except:
                print(f'unable to download {url_model} to {path_to_model} using urlretrieve, trying wget')
                wget_command = ['wget', url_model, '-P', os.path.dirname(path_to_model)]
                subprocess.run(wget_command)
    m = nn.Linear(768, 1)
    s = torch.load(path_to_model)
    m.load_state_dict(s)
    m.eval()
    return m

def clip_transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        transforms.Lambda(lambda x: x.float().div(255.0)),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def EvaluateLaionAesthetic(aesthetic_model, clip_model, preprocess, store_image_folder, device):
    aesthetic_model.eval()
    aesthetic_model=aesthetic_model.to(clip_model.dtype)
    clip_model.eval()
    with torch.no_grad():
        tmp_paths = [os.path.join(store_image_folder, "frames_" + str(f) + ".png") for f in range(1,1+len(os.listdir(store_image_folder)))]
        images = []

        for tmp_path in tmp_paths:
            images.append(preprocess(Image.open(tmp_path)))
        images = torch.stack(images)
            
        images = images.to(device)
        scores=[]
        for i in range(images.shape[0]):
            image_features = clip_model.encode_image(images[[i]])
            image_features = F.normalize(image_features, dim=-1, p=2)
            aesthetic_scores = aesthetic_model(image_features).squeeze()
            scores.append(aesthetic_scores.item())
    return np.mean(scores)

