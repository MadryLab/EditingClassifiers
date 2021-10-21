import os
import sys

import torch
import torch as ch
from torchvision.transforms import ToPILImage
import numpy as np
from tqdm import tqdm

import helpers.math_helpers as math
from tools import tally, pbar, renormalize, imgviz  

features = {}

def get_context_model(model, layernum, arch):

    def hook_feature(module, input, output):
        features['pre'] = input[0]
        features['post'] = output
        
    if arch.startswith('vgg'):
        model[layernum + 1].register_forward_hook(hook_feature)
        Nfeatures = model[layernum + 1][0].in_channels
    elif arch.startswith('clip'):
        model.visual[layernum + 1].final.register_forward_hook(hook_feature)
        Nfeatures = model.visual[layernum + 1].final.conv3.module.in_channels
    elif arch == 'resnet50':
        model[layernum + 1].final.register_forward_hook(hook_feature)
        Nfeatures = model[layernum + 1].final.conv3.module.in_channels
    elif arch == 'resnet18':
        model[layernum + 1].final.register_forward_hook(hook_feature)
        Nfeatures = model[layernum + 1].final.conv2.module.in_channels

    context_model = model

    return context_model, Nfeatures

def get_keys(batch, context_mod=None, device='cuda', 
             no_grad=True, loc='input'):
    assert len(batch.shape) == 4
    
    def get_keys_sub():
        context_mod(batch.to(device))
        if loc == 'input':
            if type(features['pre']) == tuple:
                return (features['pre'][0].detach().clone(), features['pre'][1].detach().clone())
            else:
                return features['pre'].detach().clone() 
        else:
            return features['post'].detach().clone()
    
    if no_grad:
        with torch.no_grad():
            return get_keys_sub()
    else:
        return get_keys_sub()

def get_cov_matrix(loader, context_model, batch_size=78400, 
                   key_method='zca', device='cuda', caching_dir=None,
                   force_recache=False):
   
    
    if caching_dir:
        paths = [os.path.join(caching_dir, p) 
                 for p in ['CM_k.pt', 'ZM_k.pt']]
        if all(os.path.exists(p) for p in paths) and not force_recache:
            print("Found precomputed cov matrices, returning...")
            ret = []
            for f in paths:
                ret.append(ch.load(f).to(device))
            return ret
                  
    print("Computing cov matrices...")
    CM_k = calculate_2nd_moment(loader, context_model, 
                                       batch_size=batch_size, device=device)
    
    assert not ch.any(ch.isnan(CM_k)) 
    
    if key_method == 'zca':
        dtype = CM_k.dtype
        
        if not math.is_PD(CM_k.cpu().numpy()):
            print("Making CM_k PD")
            CM_k = ch.tensor(math.get_nearest_PD(CM_k.cpu().numpy()), dtype=dtype).to(device)
        assert math.is_PD(CM_k.cpu().numpy()) 

        ZM_k = math.zca_from_cov(CM_k).to(device)
        assert not ch.any(ch.isnan(ZM_k)) 
    else:
        ZM_k = ch.eye(CM_k.shape[0]).to(device)
     
    if caching_dir:
        paths = [os.path.join(caching_dir, p) 
                 for p in ['CM_k.pt', 'ZM_k.pt']]
        os.makedirs(caching_dir, exist_ok=True)
        for t, p in zip([CM_k, ZM_k], paths):
            ch.save(t, p)
    
    return CM_k, ZM_k

def calculate_2nd_moment(val_loader, context_model, 
                                batch_size=78400, device='cuda'):
                      
    total_count = 0
    for batch_idx, (zbatch, _) in tqdm(enumerate(val_loader), total=len(val_loader)):
        acts = get_keys(zbatch, 
                        context_mod=context_model, 
                        device=device)
        if type(acts) == tuple:
            acts = acts[0]
            
        # acts is B x C x H x W
        # Reshape sep_pix to be (BHW) X C
        sep_pix = acts.permute(0, 2, 3, 1).reshape(-1, acts.shape[1])
        
        if batch_idx == 0:
            moment = ch.zeros((sep_pix.shape[1], sep_pix.shape[1])).to(sep_pix.device)
        
        total_count += sep_pix.shape[0]
        BC = int(np.ceil(sep_pix.shape[0] / batch_size))
        
        for iidx in range(BC):
            nc = sep_pix[iidx*batch_size:(iidx+1)*batch_size, :, None].shape[0]
            moment += ch.sum(ch.bmm(sep_pix[iidx*batch_size:(iidx+1)*batch_size, :, None], 
                                    sep_pix[iidx*batch_size:(iidx+1)*batch_size, None, :]), axis=0) 
                
    moment /= total_count
    assert not ch.any(ch.isnan(moment))
    
    return moment


def get_matches(context_k, ims, context_model, K=200, q=0.99):
    match_idx, match_im, match_mask, match_over = find_context_matches(
                                                       context_k, ims, 
                                                       context_model, 
                                                       k=K, 
                                                       q=q)
    
    nz_mask = np.where(np.sum(match_mask.cpu().numpy().reshape(match_mask.shape[0], -1), axis=1) != 0)[0]
    match_idx, match_im, match_mask, match_over = (match_idx[nz_mask], match_im[nz_mask], match_mask[nz_mask],
                                                   match_over[nz_mask])
    
    return match_idx, match_mask, match_over
    
    
def get_context_key(source_imgs,
                     source_masks,
                     context_model,
                     matrix,
                     rank=1,
                     device='cuda',
                     loc='input',
                     threshold=0.2):
    # Fairly ok
    with torch.no_grad():
        accumulated_obs = []
        for img, mask in zip(source_imgs, source_masks):
            k_acts = get_keys(img[None,...], 
                                  context_mod=context_model, 
                                  device=device, loc=loc)
            if type(k_acts) == tuple:
                k_acts = k_acts[0]
            area = renormalize.from_image(ToPILImage()(mask.cpu()), target='pt',
                                            size=k_acts.shape[2:])[0]
            
            
            accumulated_obs.append((
                k_acts.permute(0, 2, 3, 1).reshape(-1, k_acts.shape[1]),
                area.view(-1)[:, None].to(k_acts.device)))
        
        all_obs = torch.cat([obs[(w > 0).nonzero()[:, 0], :]
                             for obs, w in accumulated_obs])
        all_weight = torch.cat([w[w > 0]
                                for _, w in accumulated_obs])
        all_zca_k = torch.cat([(w * math.zca_whitened_query_key(matrix, obs))[(w > 0).nonzero()[:, 0], :]
                                for obs,  w in accumulated_obs])

        _, _, q = all_zca_k.svd(compute_uv=True)
        top_e_vec = q[:, :rank]
        row_dirs = math.zca_whitened_query_key(matrix, top_e_vec.t())
        just_avg = (all_zca_k).sum(0)
        q, r = torch.qr(row_dirs.permute(1, 0))
        signs = (q * just_avg[:, None]).sum(0).sign()
        q = q * signs[None, :]
        return q.permute(1, 0)
    
def find_context_matches(key, ims, context_model, k=12,  
                         device='cuda', loc='input', q=0.999):
    sel_idx, sel_imgs, query_rq = rank_using_context(key, ims, context_model, 
                                       k=k, device=device, loc=loc)    
    level = query_rq.quantiles(q)[0]
    masks, masked_imgs = find_matching_region_img(context_model,
                                            sel_imgs,
                                            key, 
                                            level,
                                            device=device,
                                            loc=loc,
                                            border_color=[255, 255, 255])
    return sel_idx, sel_imgs, masks, masked_imgs

def rank_using_context(key, ims, context_model, k=12, 
                       device='cuda', loc='input'):
    tensorkey = key.to(device).unsqueeze(2).unsqueeze(3)
    with pbar.quiet(), torch.no_grad():
        def image_max_sel(zbatch):
            acts = get_keys(zbatch, 
                                context_mod=context_model, 
                                device=device,
                                loc=loc)
            if type(acts) == tuple:
                acts = acts[0]
            heatmap = (acts * tensorkey).sum(dim=1)
            maxmap = heatmap.view(heatmap.shape[0], -1).max(1)[0]
            flatmap = heatmap.view(-1)[:, None]
            return maxmap, flatmap
        topk, rq = tally.tally_topk_and_quantile(
            image_max_sel, ims, k=k)
    sel_idx = topk.result()[1]
    return sel_idx, ims[sel_idx], rq


def find_matching_region_img(context_model, imgs, key, level, 
                             device='cuda', loc='input', **kwargs):
        batch_size = 3
        masks, masked_imgs = [], []
        for i in range(0, len(imgs), batch_size):
            img_batch = imgs[i:i + batch_size]
            
            with torch.no_grad():
                tensorkey = key.to(device).unsqueeze(2).unsqueeze(3)
                acts = get_keys(img_batch, 
                                context_mod=context_model, 
                                device=device, loc=loc)
                if type(acts) == tuple:
                    acts = acts[0]
                heatmap = (acts[...] * tensorkey).sum(dim=1)

                imgdata_batch = 2 * (img_batch - 0.5)
                iv = imgviz.ImageVisualizer(imgdata_batch.shape[2:])
                
                
                masks.extend([iv.pytorch_mask(h, unit=None, level=level,
                                     percent_level=None).cpu().float()
                                for h in heatmap])
                
                masked_imgs.extend(
                    [iv.masked_image(imgdata, heatmap[j], level=level,
                                     **kwargs)
                     for j, imgdata in enumerate(imgdata_batch)])

        masked_imgs = ch.stack([ch.tensor(np.asarray(r)).permute(2, 0, 1) for r in masked_imgs])
        return ch.stack(masks).cpu(), masked_imgs