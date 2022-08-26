from PIL import Image
import requests
import pickle
from io import BytesIO
import gradio as gr
from args import get_parser
from model import get_model
import torch
import os
from model1_inf import im2ingr
import numpy as np

response = requests.get("https://i.imgur.com/DwR24EM.jpeg")
dog_img = Image.open(BytesIO(response.content))

def img2ingr(image):
    # img_file = '../data/demo_imgs/1.jpg'
    # image = Image.open(img_file).convert('RGB')
    img = Image.fromarray(np.uint8(image)).convert('RGB')
    ingr = im2ingr(img, ingrs_vocab, model)
    return ' '.join(ingr)

def img_ingr2recipe(image, ingr):
    print(image.shape, ingr)
    return dog_img, "A delicious meme dog \n--------\n1. Cook it!\n2. GL&HF"

def change_checkbox(predicted_ingr):
    return gr.update(label="Ingredient required", interactive=True, choices=predicted_ingr.split(), value=predicted_ingr.split())

def add_ingr(new_ingr):
    print(new_ingr)
    return "hello"

def add_to_checkbox(old_ingr, new_ingr):
    # chack if in dict or not
    return gr.update(label="Ingredient required", interactive=True, choices=[*old_ingr, new_ingr], value=[*old_ingr, new_ingr])


""" load model1 """
args = get_parser()

# basic parameters
model_dir = '../data'
data_dir = '../data'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
map_loc = None if torch.cuda.is_available() else 'cpu'

# load ingredients vocab
ingrs_vocab = pickle.load(open(os.path.join(model_dir, 'ingr_vocab.pkl'), 'rb'))
vocab = pickle.load(open(os.path.join(data_dir, 'instr_vocab.pkl'), 'rb'))

ingr_vocab_size = len(ingrs_vocab)
instrs_vocab_size = len(vocab)

# model setting and loading
args.maxseqlen = 15
args.ingrs_only=True
model = get_model(args, ingr_vocab_size, instrs_vocab_size)
model_path = os.path.join(model_dir, 'modelbest.ckpt')
model.load_state_dict(torch.load(model_path, map_location=map_loc))
model.to(device)
model.eval()
model.ingrs_only = True
model.recipe_only = False

""" load model2 """




""" gradio """
# input image -> list all required ingrs -> checkbox for selecting ingrs / input_box for input more ingrs user want -> output: recipe and its image
example_dir = "../data/demo_imgs/"
with gr.Blocks() as demo:
    gr.Markdown(
    """
    # Recipedia
    Start finding the yummy recipe ...
    """)
    with gr.Tabs():
        with gr.TabItem("User"):
            # input image
            image_input = gr.Image(label="Upload the image of your yummy food", type='filepath')
            gr.Examples(examples=[example_dir+"1.jpg", example_dir+"2.jpg", example_dir+"3.jpg", example_dir+"4.jpg", example_dir+"5.jpg", example_dir+"6.jpg"], inputs=image_input)
            with gr.Row():
                # clear_img_btn = gr.Button("Clear")
                image_btn = gr.Button("Upload", variant="primary")
            # list all required ingrs -> checkbox for selecting ingrs / input_box for input more ingrs user want
            predicted_ingr = gr.Textbox(visible=False)

            with gr.Row():
                checkboxes = gr.CheckboxGroup(label="Ingredient required", interactive=True)
                new_ingr = gr.Textbox(label="Addtional ingredients", max_lines=1)
                    # with gr.Row():
                    #     new_btn_clear = gr.Button("Clear")
                    #     new_btn = gr.Button("Add", variant="primary")

            add_ingr = gr.Textbox(visible=False)

            with gr.Row():
                clear_ingr_btn = gr.Button("Reset")
                ingr_btn = gr.Button("Confirm", variant="primary")

            # output: recipe and its image
            with gr.Row():
                out_recipe = gr.Textbox(label="Your recipe", value="Spagetti ---\n1. cook it!")
                out_image = gr.Image(label="Looks yummy ><")

        with gr.TabItem("Example"):
            image_button = gr.Button("Flip")
        
        image_btn.click(img2ingr, inputs=image_input, outputs=predicted_ingr)
        predicted_ingr.change(fn=change_checkbox, inputs=predicted_ingr, outputs=checkboxes)

        # new_btn.click(img2ingr, inputs=new_ingr, outputs=predicted_ingr)
        new_ingr.submit(fn=add_to_checkbox, inputs=[checkboxes, new_ingr], outputs=checkboxes)

        ingr_btn.click(img_ingr2recipe, inputs=[image_input, checkboxes], outputs=[out_image, out_recipe])


demo.launch(debug=True, share=True)