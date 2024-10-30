import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
import os
from sklearn.metrics.pairwise import cosine_similarity
from torchvision.models.resnet import ResNet18_Weights
from torchvision import transforms
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA

def extract_feature_vector(img_tensor):
    t_img = Variable(img_tensor)
    my_embedding = torch.zeros(1, 512, 1, 1)
    def copy_data(m, i, o):
        my_embedding.copy_(o.data)
    h = layer.register_forward_hook(copy_data)
    model(t_img)
    h.remove()
    return my_embedding.squeeze().numpy()

def process_and_save_features(input_base_folder, output_base_folder, dim):
    folder_counter = 1

    for subdir, dirs, files in os.walk(input_base_folder):
        if not files:
            continue
        all_feature_vectors = []
        for file in files:
            if file.lower().endswith('.jpeg'):
                img_path = os.path.join(subdir, file)
                img = Image.open(img_path)
                img = convert_to_3_channels(img)
                img_tensor = preprocess(img)
                img_tensor = img_tensor.unsqueeze(0)

                feature_vector = extract_feature_vector(img_tensor)
                all_feature_vectors.append(feature_vector)


        all_feature_vectors_np = np.array(all_feature_vectors)
        pca = PCA(n_components=dim)
        reduced_vectors = pca.fit_transform(all_feature_vectors_np)


        current_output_folder = os.path.join(output_base_folder, f"train_feature{folder_counter}")
        if not os.path.exists(current_output_folder):
            os.makedirs(current_output_folder)

        for i, reduced_vector in enumerate(reduced_vectors, start=1):
            output_path = os.path.join(current_output_folder, f"{i}.npy")
            np.save(output_path, reduced_vector)

        folder_counter += 1

def convert_to_3_channels(img):
    """Convert an image to 3 channels if it's not already."""
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return img

model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model.eval()
layer = model.avgpool
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
preprocess = transforms.Compose([
    transforms.Lambda(convert_to_3_channels),
    transforms.Resize(256, antialias=True),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])

dim = 300
input_base_folder = './train'
output_base_folder = f'./train_feature_{dim}'

process_and_save_features(input_base_folder, output_base_folder, dim)