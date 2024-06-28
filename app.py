# import spaces
import gradio as gr
import os
import sys
import argparse
import random
import time
from omegaconf import OmegaConf
import torch
import torchvision
from pytorch_lightning import seed_everything
from huggingface_hub import hf_hub_download
from einops import repeat
import torchvision.transforms as transforms
from utils.utils import instantiate_from_config
sys.path.insert(0, "scripts/evaluation")
from scripts.evaluation.funcs import (
    batch_ddim_sampling,
    load_model_checkpoint,
    get_latent_z,
    save_videos
)

def download_model():
    REPO_ID = 'GraceZhao/DynamiCrafter-CIL-512'
    ckpt_dir = './checkpoints/dynamicrafter_512_cil/'
    filename_list = ['timenoise.ckpt']
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    for filename in filename_list:
        local_file = os.path.join(ckpt_dir, filename)
        if not os.path.exists(local_file):
            hf_hub_download(repo_id=REPO_ID, filename=filename, local_dir=ckpt_dir, force_download=True)



download_model()
ckpt_path='checkpoints/dynamicrafter_512_cil/timenoise.ckpt'
config_file='configs/inference_512_v1.0.yaml'
config = OmegaConf.load(config_file)
model_config = config.pop("model", OmegaConf.create())
model_config['params']['unet_config']['params']['use_checkpoint']=False   
model = instantiate_from_config(model_config)
assert os.path.exists(ckpt_path), "Error: checkpoint Not Found!"
model = load_model_checkpoint(model, ckpt_path)
model.eval()
model = model.cuda()



# @spaces.GPU(duration=300)
def infer(image, prompt, steps=50, cfg_scale=7.5, eta=1.0, fs=3, seed=123, ddpm_from=1000):
    resolution = (320, 512)
    save_fps = 8
    seed_everything(seed)
    transform = transforms.Compose([
        transforms.Resize(resolution),
        ])
    torch.cuda.empty_cache()
    print('start:', prompt, time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
    start = time.time()
    if steps > 60:
        steps = 60 

    batch_size=1
    channels = model.model.diffusion_model.out_channels
    frames = model.temporal_length
    h, w = resolution[0] // 8, resolution[1] // 8
    noise_shape = [batch_size, channels, frames, h, w]

    # text cond
    with torch.no_grad(), torch.cuda.amp.autocast():
        text_emb = model.get_learned_conditioning([prompt])
    
        # img cond
        img_tensor = torch.from_numpy(image).permute(2, 0, 1).float().to(model.device)
        img_tensor = (img_tensor / 255. - 0.5) * 2
    
        image_tensor_resized = transform(img_tensor) #3,256,256
        videos = image_tensor_resized.unsqueeze(0) # bchw
        
        z = get_latent_z(model, videos.unsqueeze(2)) #bc,1,hw
        
        img_tensor_repeat = repeat(z, 'b c t h w -> b c (repeat t) h w', repeat=frames)
    
        cond_images = model.embedder(img_tensor.unsqueeze(0)) ## blc
        img_emb = model.image_proj_model(cond_images)
    
        imtext_cond = torch.cat([text_emb, img_emb], dim=1)
    
        fs = torch.tensor([fs], dtype=torch.long, device=model.device)
        cond = {"c_crossattn": [imtext_cond], "fs": fs, "c_concat": [img_tensor_repeat]}
        
        ## inference
        batch_samples = batch_ddim_sampling(model, cond, noise_shape, n_samples=1, ddim_steps=steps, ddim_eta=eta, cfg_scale=cfg_scale, ddpm_from=ddpm_from)
        ## b,samples,c,t,h,w
    
        video_path = './output.mp4'
        save_videos(batch_samples, './', filenames=['output'], fps=save_fps)
    return video_path


i2v_examples = [
    ['prompts/512/7.png', 'Donkeys in traditional attire gallop across a lush green meadow.', 50, 7.5, 1.0, 24, 123,900],
    ['prompts/512/41.png', 'Rabbits playing in a river.', 50, 7.5, 1.0, 24, 123,900],
    ['prompts/512/32.png', 'Mountains under the starlight.', 50, 7.5, 1.0, 24, 123,900],
    ['prompts/512/14.png', 'A duck swimming in the lake.', 50, 7.5, 1.0, 24, 123,900],
    ['prompts/512/30.png', 'A soldier riding a horse.', 50, 7.5, 1.0, 24, 123,900],
    ['prompts/512/52.png', 'Fireworks exploding in the sky.', 50, 7.5, 1.0, 24, 123,900],
]




css = """#input_img {max-width: 1024px !important} #output_vid {max-width: 1024px; max-height: 576px}"""

with gr.Blocks(analytics_enabled=False, css=css) as demo:
    gr.Markdown("<div align='center'> <h1> DynamiCrafter-CIL </span> </h1> \
                    <h2 style='font-weight: 450; font-size: 1rem; margin: 0rem'>\
                    <a href='https://gracezhao1997.github.io/'>Min Zhao</a>, \
                    <a href='https://zhuhz22.github.io/'>Hongzhou Zhu</a>, \
                    <a href='https://xiang-cd.github.io/'>Chendong Xiang</a>, \
                    <a href='https://scholar.google.com/citations?user=0d80xSIAAAAJ&hl=en'>Kaiwen Zheng</a>, \
                    <a href='https://zhenxuan00.github.io/'> Chongxuan Li</a>,\
                    <a href='https://ml.cs.tsinghua.edu.cn/~jun/software.shtml'>Jun Zhu</a>,\
                </h2> \
                <a style='font-size:18px;color: #000000'>If DynamiCrafter is useful, please help star the </a>\
                <a style='font-size:18px;color: #000000' href='https://github.com/thu-ml/cond-image-leakage/'>[Github Repo]</a>\
                <a style='font-size:18px;color: #000000'>, which is important to Open-Source projects. Thanks!</a>\
                    <a style='font-size:18px;color: #000000' href='https://arxiv.org/abs/2406.15735'> [ArXiv] </a>\
                    <a style='font-size:18px;color: #000000' href='https://cond-image-leak.github.io/'> [Project Page] </a> </div>")
    
    with gr.Tab(label='ImageAnimation_576x1024'):
        with gr.Column():
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        i2v_input_image = gr.Image(label="Input Image",elem_id="input_img")
                    with gr.Row():
                        i2v_input_text = gr.Text(label='Prompts')
                    with gr.Row():
                        i2v_seed = gr.Slider(label='Random Seed', minimum=0, maximum=10000, step=1, value=123)
                        i2v_eta = gr.Slider(minimum=0.0, maximum=1.0, step=0.1, label='ETA', value=1.0, elem_id="i2v_eta")
                        i2v_cfg_scale = gr.Slider(minimum=1.0, maximum=15.0, step=0.5, label='CFG Scale', value=7.5, elem_id="i2v_cfg_scale")
                    with gr.Row():
                        i2v_steps = gr.Slider(minimum=1, maximum=50, step=1, elem_id="i2v_steps", label="Sampling steps", value=30)
                        i2v_motion = gr.Slider(minimum=5, maximum=20, step=1, elem_id="i2v_motion", label="FPS", value=10)
                        i2v_ddpm_from = gr.Slider(minimum=840, maximum=1000, step=1, elem_id="i2v_motion", label="ddpm_from", value=900)
                        
                    i2v_end_btn = gr.Button("Generate")
                # with gr.Tab(label='Result'):
                with gr.Row():
                    i2v_output_video = gr.Video(label="Generated Video",elem_id="output_vid",autoplay=True,show_share_button=True)

            gr.Examples(examples=i2v_examples,
                        inputs=[i2v_input_image, i2v_input_text, i2v_steps, i2v_cfg_scale, i2v_eta, i2v_motion, i2v_seed],
                        outputs=[i2v_output_video],
                        fn = infer,
                        cache_examples=True,
            )
        i2v_end_btn.click(inputs=[i2v_input_image, i2v_input_text, i2v_steps, i2v_cfg_scale, i2v_eta, i2v_motion, i2v_seed, i2v_ddpm_from],
                        outputs=[i2v_output_video],
                        fn = infer
        )

demo.queue(max_size=12).launch(show_api=True)