import warnings, time, timeit, sys, os
warnings.filterwarnings('ignore')

import tensorflow as tf
from tqdm import tqdm
import torch as ch
import tensorflow_hub as hub
from PIL import Image
from argparse import ArgumentParser
from helpers import classifier_helpers
import helpers.data_helpers as dh

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

hub_handle = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
hub_module = hub.load(hub_handle)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--style-dir', type=str, default='./data/synthetic/styles') 
    parser.add_argument('--style-name', type=str, default='colorful_flowers.pt')
    parser.add_argument('--dataset-name', type=str, default='ImageNet')
    parser.add_argument('--Nstyles', type=int, default=3)
    parser.add_argument('--content-blending-ratio', type=float, default=0.8)
    parser.add_argument('--out-path', type=str, default='./data/synthetic/stylized')
    args = parser.parse_args()
    
    out_dir = os.path.join(args.out_path, args.dataset_name)
    if not os.path.exists(out_dir): 
        os.makedirs(out_dir)

    content_blending_ratio = args.content_blending_ratio
        
    ret = classifier_helpers.get_default_paths(args.dataset_name)
    base_dataset, train_loader, val_loader = dh.get_dataset(args.dataset_name, 
                                                            ret[0],
                                                            batch_size=1, workers=8)
    style_name  = args.style_name
    style_dir = os.path.join(out_dir, style_name.split('.pt')[0])
    if not os.path.exists(style_dir):
        os.makedirs(style_dir)

    style = ch.load(os.path.join(args.style_dir, style_name))
    style_images = tf.convert_to_tensor(style['imgs'].numpy().transpose(0, 2, 3, 1))[:args.Nstyles]

    for style_num in range(len(style_images)):
        style_image = style_images[style_num:style_num+1]
        print('Style number:', style_num)
        out_dir_curr = os.path.join(style_dir, str(style_num))
        if not os.path.exists(out_dir_curr): os.makedirs(out_dir_curr)

        it = tqdm(val_loader, total=len(val_loader))
        for imgno, (content_image, _) in enumerate(it):
            
            if os.path.exists(f"{out_dir_curr}/img_{imgno}.png"):
                continue
            content_image = tf.convert_to_tensor(content_image.numpy().transpose(0, 2, 3, 1))
            outputs = hub_module(tf.constant(content_image), tf.constant(style_image)) 
            stylized_image = content_image * (1 - content_blending_ratio) + \
                        outputs[0] * content_blending_ratio    

            stylized_image = stylized_image.numpy()[0]
            im = Image.fromarray((255 * stylized_image).astype('uint8'))
            im.save(f"{out_dir_curr}/img_{imgno}.png")
    
