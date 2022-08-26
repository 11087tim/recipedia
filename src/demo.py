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
from tqdm import tqdm
import time
import glob


# Set ```data_dir``` to the path including vocabularies and model checkpoint
model_dir = '../data'
image_folder = '../data/demo_imgs'
output_file = "../data/predicted_ingr.pkl"

# code will run in gpu if available and if the flag is set to True, else it will run on cpu
use_gpu = False
device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
map_loc = None if torch.cuda.is_available() and use_gpu else 'cpu'

# code below was used to save vocab files so that they can be loaded without Vocabulary class
#ingrs_vocab = pickle.load(open(os.path.join(data_dir, 'final_recipe1m_vocab_ingrs.pkl'), 'rb'))
#ingrs_vocab = [min(w, key=len) if not isinstance(w, str) else w for w in ingrs_vocab.idx2word.values()]
#vocab = pickle.load(open(os.path.join(data_dir, 'final_recipe1m_vocab_toks.pkl'), 'rb')).idx2word
#pickle.dump(ingrs_vocab, open('../demo/ingr_vocab.pkl', 'wb'))
#pickle.dump(vocab, open('../demo/instr_vocab.pkl', 'wb'))

ingrs_vocab = pickle.load(open(os.path.join(model_dir, 'ingr_vocab.pkl'), 'rb'))
vocab = pickle.load(open(os.path.join(model_dir, 'instr_vocab.pkl'), 'rb'))

ingr_vocab_size = len(ingrs_vocab)
instrs_vocab_size = len(vocab)
output_dim = instrs_vocab_size

print (instrs_vocab_size, ingr_vocab_size)

t = time.time()

args = get_parser()
args.maxseqlen = 15
args.ingrs_only=True
model = get_model(args, ingr_vocab_size, instrs_vocab_size)
# Load the trained model parameters
model_path = os.path.join(model_dir, 'modelbest.ckpt')
model.load_state_dict(torch.load(model_path, map_location=map_loc))
model.to(device)
model.eval()
model.ingrs_only = True
model.recipe_only = False
print ('loaded model')
print ("Elapsed time:", time.time() -t)

transf_list_batch = []
transf_list_batch.append(transforms.ToTensor())
transf_list_batch.append(transforms.Normalize((0.485, 0.456, 0.406), 
                                              (0.229, 0.224, 0.225)))
to_input_transf = transforms.Compose(transf_list_batch)


greedy = True
beam = -1
temperature = 1.0

# import requests
# from io import BytesIO
# import random
# from collections import Counter
# use_urls = False # set to true to load images from demo_urls instead of those in test_imgs folder
# show_anyways = False #if True, it will show the recipe even if it's not valid
# image_folder = os.path.join(data_dir, 'demo_imgs')

# if not use_urls:
#     demo_imgs = os.listdir(image_folder)
#     random.shuffle(demo_imgs)

# demo_urls = ['https://food.fnr.sndimg.com/content/dam/images/food/fullset/2013/12/9/0/FNK_Cheesecake_s4x3.jpg.rend.hgtvcom.826.620.suffix/1387411272847.jpeg',
#             'https://www.196flavors.com/wp-content/uploads/2014/10/california-roll-3-FP.jpg']

files_path = glob.glob(f"{image_folder}/*/*/*.jpg")
print(f"total data: {len(files_path)}")

res = []
for idx, img_file in tqdm(enumerate(files_path)):
    # if use_urls:
    #     response = requests.get(img_file)
    #     image = Image.open(BytesIO(response.content))
    # else:
    image = Image.open(img_file).convert('RGB')
    
    transf_list = []
    transf_list.append(transforms.Resize(256))
    transf_list.append(transforms.CenterCrop(224))
    transform = transforms.Compose(transf_list)
    
    image_transf = transform(image)
    image_tensor = to_input_transf(image_transf).unsqueeze(0).to(device)
    
    # plt.imshow(image_transf)
    # plt.axis('off')
    # plt.show()
    # plt.close()

    with torch.no_grad():
        outputs = model.sample(image_tensor, greedy=greedy, 
                                temperature=temperature, beam=beam, true_ingrs=None)
        
    ingr_ids = outputs['ingr_ids'].cpu().numpy()
    print(ingr_ids)
        
    outs = prepare_output(ingr_ids[0], ingrs_vocab)
    # print(ingrs_vocab.idx2word)

    print(outs)
        
    # print ('Pic ' + str(idx+1) + ':')

    # print ('\nIngredients:')
    # print (', '.join(outs['ingrs']))

    # print ('='*20)
    
    res.append({
        "id": img_file,
        "ingredients": outs['ingrs']
    })

with open(output_file, "wb") as fp:   #Pickling
    pickle.dump(res, fp)
