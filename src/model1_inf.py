import torch
import torch.nn as nn
import numpy as np
import os
from args import get_parser
import pickle
from model import get_model
from torchvision import transforms
from utils.output_ing import prepare_output
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
map_loc = None if torch.cuda.is_available() else 'cpu'

def im2ingr(image, ingrs_vocab, model):
    transf_list_batch = []
    transf_list_batch.append(transforms.ToTensor())
    transf_list_batch.append(transforms.Normalize((0.485, 0.456, 0.406), 
                                                (0.229, 0.224, 0.225)))
    to_input_transf = transforms.Compose(transf_list_batch)

    greedy = True
    beam = -1
    temperature = 1.0

    transf_list = []
    transf_list.append(transforms.Resize(256))
    transf_list.append(transforms.CenterCrop(224))
    transform = transforms.Compose(transf_list)
    
    image_transf = transform(image)
    image_tensor = to_input_transf(image_transf).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model.sample(image_tensor, greedy=greedy, 
                                temperature=temperature, beam=beam, true_ingrs=None)
        
    ingr_ids = outputs['ingr_ids'].cpu().numpy()
    outs = prepare_output(ingr_ids[0], ingrs_vocab)

    return outs['ingrs']


