import os
import torch
from argparse import ArgumentParser
from datasets import Sennet_image_pair
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from hf_img_networks import NICE_Trans_img
from matplotlib import pyplot as plt
import torch.nn.functional as F
import torch
import scanpy as sc
import cv2
from PIL import Image

def find_inverse_warp_coords(uv_coords, flow):
    """
    Args:
        uv_coords: [N, 2] (x, y) in the warped image
        flow: [2, H, W] forward flow (dy, dx)

    Returns:
        approx_orig_coords: [N, 2] — approximate original locations
    """
    device = flow.device
    dy, dx = flow[0], flow[1]  # shape: [H, W]
    H, W = dx.shape

    # 1. Build dense grid
    y, x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    base_grid = torch.stack([x, y], dim=-1).float()  # [H, W, 2]

    # 2. Apply flow
    warped_grid = base_grid + torch.stack([dx, dy], dim=-1)  # [H, W, 2]

    # 3. Flatten
    warped_flat = warped_grid.reshape(-1, 2)  # [H*W, 2]
    base_flat = base_grid.reshape(-1, 2)  # [H*W, 2]

    # 4. Nearest neighbor search
    dists = torch.cdist(uv_coords.unsqueeze(0), warped_flat.unsqueeze(0))  # [1, N, H*W]
    nn_indices = dists.argmin(dim=-1).squeeze(0)  # [N]

    # 5. Use the corresponding original positions
    approx_orig_coords = base_flat[nn_indices]  # [N, 2]

    return approx_orig_coords



def create_overlay(img1,img2):
    img1_rgb = np.stack([img1, np.zeros_like(img1), img1], axis=-1)
    # Image 2 → Cyan (G + B)
    img2_rgb = np.stack([np.zeros_like(img2), img2, img2], axis=-1)

    # Blend with maximum pixel value
    overlay = np.maximum(img1_rgb, img2_rgb)
    return overlay
def min_max_normalize(array):
    min = np.min(array)   # shape (C,)
    max = np.max(array)   # shape (C,)
    range = max - min

    normalized_data = (array - min) / range
    return normalized_data
def tensor2img(tensor):
    tensor = tensor.detach().cpu().numpy().squeeze()
    tensor = min_max_normalize(tensor)
    return tensor
def visualize_flow(flow, image=None,step=30, title="Flow field", ax=None,out_dir=None):
    """
    Visualizes a 2D flow field using quiver plot.
    flow: [2, H, W] or [1, 2, H, W]
    """
    if flow.ndim == 4:
        flow = flow[0]  # remove batch

    dy, dx = flow[0], flow[1]
    H, W = dy.shape

    y, x = torch.meshgrid(torch.arange(0, H, step), torch.arange(0, W, step), indexing='ij')
    y = y.cpu()
    x = x.cpu()

    u = dx[::step, ::step].cpu()
    v = dy[::step, ::step].cpu()

    if ax is None:
        ax = plt.gca()
    if image is not None:
        ax.imshow(image)
    ax.quiver(x, y, -u, -v, angles='xy', scale_units='xy', scale=1, color='red', width=0.002)
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_aspect('equal')


def resize_flow(flow, target_h, target_w):
    """
    Resize flow from [2, H, W] to [2, target_h, target_w] with proper value scaling.
    """
    H, W = flow.shape[1:]
    scale_h = target_h / H
    scale_w = target_w / W

    flow = flow.unsqueeze(0)  # [1, 2, H, W]
    flow_resized = F.interpolate(flow, size=(target_h, target_w), mode='bilinear', align_corners=True)
    flow_resized = flow_resized.squeeze(0)

    # Scale flow values accordingly
    flow_resized[0] *= scale_w  # x-direction
    flow_resized[1] *= scale_h  # y-direction

    return flow_resized

def rescale_coordinates(coords, original_size, new_size):
    """
    Rescale 2D coordinates from original image size to new image size.

    coords: numpy array of shape [N, 2], where coords[:, 0] is x and coords[:, 1] is y
    original_size: (H, W)
    new_size: (H', W')
    """
    orig_h, orig_w = original_size
    new_h, new_w = new_size

    scale_x = new_w / orig_w
    scale_y = new_h / orig_h

    coords_rescaled = coords.copy()
    coords_rescaled[:, 0] *= scale_x  # x-coordinate (width direction)
    coords_rescaled[:, 1] *= scale_y  # y-coordinate (height direction)

    return coords_rescaled

def save_single_dataset(image, gene_data, datatype, sampleid, regionid, output_dir):

    os.makedirs(output_dir, exist_ok=True)
    # Create base name
    base_name = f"{datatype}_{sampleid}_{regionid}"
    # Save AnnData object
    gene_data.write(os.path.join(output_dir, f"{base_name}_registered.h5ad"))
    # Save aligned image
    img_uint8 = (image * 255).astype('uint8') if image.max() <= 1.0 else image
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_registered.png"), img_uint8)
    print(f"Saved: {base_name} to {output_dir}")

def load_and_transform_image(image_path, device='cuda'):
    image = Image.open(image_path).convert('L')
    image_tensor = img_transforms(image)
    image_tensor = image_tensor.to(device)
    return image_tensor.unsqueeze(dim=0).unsqueeze(dim=0)  # shape: [1,1,1, 1024, 2048]

parser = ArgumentParser()


parser.add_argument("--load_model", type=str,
                    dest="load_model", default='/media/huifang/data/experiment/registration/sennet/sennet_enhanced_image_1st_regu_loss_large_image/model_950.pth',
                    help="load model file to initialize with")
args = parser.parse_args()

img_transforms = transforms.Compose([transforms.ToTensor(),
                 transforms.Resize((1024,2048),antialias=True),
                  transforms.Normalize((0.5), (0.5))])


test_dataloader = DataLoader(Sennet_image_pair(datalist="/media/huifang/data/sennet/xenium_codex_pairs.txt",image_transformer=img_transforms),batch_size=1, shuffle=False, num_workers=16)
# device handling
cuda = True if torch.cuda.is_available() else False
torch.cuda.set_device(0)
if cuda:
    device = 'cuda'
    os.environ['CUDA_VISIBLE_DEVICES'] = device[-1]
    torch.backends.cudnn.deterministic = True
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    device = 'cpu'

# prepare the model
model = NICE_Trans_img(in_channels=1)
checkpoint = torch.load(args.load_model, map_location="cpu")
model.load_state_dict(checkpoint)
model.to("cuda")
model.eval()


Img_warpper = model.SpatialTransformer

file_path = '/media/huifang/data/sennet/xenium_codex_pairs.txt'
file = open(file_path)
sennet_pairs =file.readlines()
for i, batch in enumerate(test_dataloader):
    print(i)
    line = sennet_pairs[i]
    xenium_sampleid, xenium_regionid, codex_sampleid, codex_regionid = line.rstrip().split(' ')
    with torch.no_grad():
        inputs = batch
        fixed_image = inputs['fixed_image'].to(device).float()
        moving_image = inputs['moving_image'].to(device).float()

        fixed_image = fixed_image.unsqueeze(dim=2)
        moving_image = moving_image.unsqueeze(dim=2)





        warped_image, flow, affined_image, affine_para = \
            model(fixed_image, moving_image)

        xenium_image = load_and_transform_image(
            '/media/huifang/data/sennet/hf_aligned_data/xenium' + f"_{xenium_sampleid}_{xenium_regionid}.png")
        codex_image = load_and_transform_image(
            '/media/huifang/data/sennet/hf_aligned_data/codex' + f"_{codex_sampleid}_{codex_regionid}.png")
        warped_codex_image = Img_warpper(codex_image,flow)


        fixed_image = tensor2img(xenium_image)
        moving_image = tensor2img(codex_image)
        warped_image = tensor2img(warped_codex_image)
        flow = flow.squeeze().detach().cpu()
        flow = flow[1:,:,:]




        f,a = plt.subplots(2,3,figsize=(16,8))
        a[0, 0].imshow(fixed_image)
        a[0, 0].set_title("Fixed Image")

        a[0, 1].imshow(moving_image)
        a[0, 1].set_title("Moving Image")

        a[0, 2].imshow(warped_image)
        a[0, 2].set_title("Warped Image")

        a[1, 0].imshow(create_overlay(fixed_image, moving_image))
        a[1, 0].set_title("Overlay: Fixed vs. Moving")

        a[1, 1].imshow(create_overlay(fixed_image, warped_image))
        a[1, 1].set_title("Overlay: Fixed vs. Warped")

        visualize_flow(flow, ax=a[1, 2], title="Flow Field")

        # plt.imshow(warped_image)
        plt.show()






