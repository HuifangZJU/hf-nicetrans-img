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
from sklearn.neighbors import NearestNeighbors
import matplotlib.cm as cm
import matplotlib.colors as mcolors

def find_inverse_warp_coords(uv_coords, flow):
    dy, dx = flow[0], flow[1]  # [H, W]
    H, W = dx.shape

    # 1. Create base grid (x, y) in [H, W]
    y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    base_grid = torch.stack([x, y], dim=-1).reshape(-1, 2)  # [H*W, 2]

    # 2. Add flow to base grid
    warped_grid = base_grid + torch.stack([dx, dy], dim=-1).reshape(-1, 2)  # [H*W, 2]

    # 3. Convert to NumPy
    warped_grid_np = warped_grid.cpu().numpy()
    base_grid_np = base_grid.cpu().numpy()
    uv_coords_np = uv_coords.cpu().numpy()

    # 4. KD-tree nearest neighbor
    tree = NearestNeighbors(n_neighbors=1).fit(warped_grid_np)
    _, indices = tree.kneighbors(uv_coords_np)

    # 5. Gather original coordinates
    approx_orig_coords_np = base_grid_np[indices[:, 0]]
    approx_orig_coords = torch.tensor(approx_orig_coords_np, dtype=torch.float32, device=flow.device)

    return approx_orig_coords  # [N, 2]



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


    # if out_dir:
    #     plt.savefig(out_dir, dpi=300)
    # else:
    #     plt.show()



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

def run_network():
    torch.cuda.set_device(0)
    device = 'cuda'
    os.environ['CUDA_VISIBLE_DEVICES'] = device[-1]
    torch.backends.cudnn.deterministic = True

    img_transforms = transforms.Compose([transforms.ToTensor(),
                     transforms.Resize((1024,2048),antialias=True),
                      transforms.Normalize((0.5), (0.5))])

    test_dataloader = DataLoader(Sennet_image_pair(datalist="/media/huifang/data/sennet/xenium_codex_pairs.txt",image_transformer=img_transforms),batch_size=1, shuffle=False, num_workers=16)

    # prepare the model
    model = NICE_Trans_img(in_channels=1)
    checkpoint = torch.load('/media/huifang/data/experiment/registration/sennet/sennet_enhanced_image_1st_regu_loss_large_image/model_950.pth', map_location="cpu")
    model.load_state_dict(checkpoint)
    model.to("cuda")
    model.eval()

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


            fixed_image = tensor2img(fixed_image)
            moving_image = tensor2img(moving_image)
            warped_image = tensor2img(warped_image)
            flow = flow.squeeze().detach().cpu()
            flow = flow[1:,:,:]
            # visualize_flow(flow)


            xenium_image = plt.imread('/media/huifang/data/sennet/hf_aligned_data/xenium'+f"_{xenium_sampleid}_{xenium_regionid}.png")
            xenium_gene_data = sc.read_h5ad("/media/huifang/data/sennet/hf_aligned_data/xenium"+ f"_{xenium_sampleid}_{xenium_regionid}.h5ad")
            codex_gene_data = sc.read_h5ad(
                "/media/huifang/data/sennet/hf_aligned_data/codex" + f"_{codex_sampleid}_{codex_regionid}.h5ad")

            xenium_coordinate = np.stack([
                xenium_gene_data.obs['x_trans'].values,
                xenium_gene_data.obs['y_trans'].values
            ], axis=1)
            codex_coordinate = np.stack([
                codex_gene_data.obs['x_trans'].values,
                codex_gene_data.obs['y_trans'].values
            ], axis=1)



            # flow = resize_flow(flow, xenium_image.shape[0], xenium_image.shape[1])
            # xenium_coordinate = rescale_coordinates(xenium_coordinate,xenium_image.shape,(1024,2048))
            codex_coordinate = rescale_coordinates(codex_coordinate,xenium_image.shape,(1024,2048))
            warped_codex_coor = find_inverse_warp_coords(torch.from_numpy(codex_coordinate).float(), flow)
            warped_codex_coor = rescale_coordinates(warped_codex_coor.detach().cpu().numpy(),(1024,2048),xenium_image.shape)

            xenium_gene_data.obs['x_aligned'] = xenium_coordinate[:, 0]
            xenium_gene_data.obs['y_aligned'] = xenium_coordinate[:, 1]
            codex_gene_data.obs['x_aligned'] = warped_codex_coor[:, 0]
            codex_gene_data.obs['y_aligned'] = warped_codex_coor[:, 1]

            # plt.scatter(xenium_coordinate[:, 0], xenium_coordinate[:, 1], s=5)
            # plt.scatter(warped_codex_coor[:, 0], warped_codex_coor[:, 1], s=5)
            # plt.show()

            save_single_dataset(fixed_image, xenium_gene_data, 'xenium', xenium_sampleid, xenium_regionid, '/media/huifang/data/sennet/registered_data/')
            save_single_dataset(warped_image, codex_gene_data, 'codex', codex_sampleid, codex_regionid,
                                '/media/huifang/data/sennet/registered_data/')
            # visualize_flow(flow,out_dir='/media/huifang/data/sennet/registered_data/pair'+str(i)+'_flow.png')



# run_network()
# print('all done')
# test = input()


file_path = '/media/huifang/data/sennet/xenium_codex_pairs.txt'
file = open(file_path)
sennet_pairs = file.readlines()
for i in range(0,len(sennet_pairs)):
    print(i)
    line = sennet_pairs[i]
    xenium_sampleid, xenium_regionid, codex_sampleid, codex_regionid = line.rstrip().split(' ')
    # Xenium: use CDKN1A expression from .X

    # xenium_image = plt.imread(
    #     '/media/huifang/data/sennet/hf_aligned_data/xenium' + f"_{xenium_sampleid}_{xenium_regionid}.png")
    # codex_image = plt.imread(
    #     '/media/huifang/data/sennet/hf_aligned_data/codex' + f"_{codex_sampleid}_{codex_regionid}.png")
    #
    # warped_xenium_image = plt.imread(
    #     '/media/huifang/data/sennet/registered_data/xenium' + f"_{xenium_sampleid}_{xenium_regionid}_registered.png")
    # warped_codex_image = plt.imread(
    #     '/media/huifang/data/sennet/registered_data/codex' + f"_{codex_sampleid}_{codex_regionid}_registered.png")
    #
    # plt.imshow(create_overlay(warped_xenium_image,warped_codex_image))
    # plt.gca().invert_yaxis()
    # plt.show()

    xenium_gene_data = sc.read_h5ad(
        "/media/huifang/data/sennet/registered_data/xenium" + f"_{xenium_sampleid}_{xenium_regionid}_registered.h5ad")
    codex_gene_data = sc.read_h5ad(
        "/media/huifang/data/sennet/registered_data/codex" + f"_{codex_sampleid}_{codex_regionid}_registered.h5ad")

    xenium_coordinate = np.stack([
        xenium_gene_data.obs['x_trans'].values,
        xenium_gene_data.obs['y_trans'].values
    ], axis=1)
    codex_coordinate = np.stack([
        codex_gene_data.obs['x_trans'].values,
        codex_gene_data.obs['y_trans'].values
    ], axis=1)

    warped_xenium_coor = np.stack([
        xenium_gene_data.obs['x_aligned'].values,
        xenium_gene_data.obs['y_aligned'].values
    ], axis=1)

    warped_codex_coor= np.stack([
        codex_gene_data.obs['x_aligned'].values,
        codex_gene_data.obs['y_aligned'].values
    ], axis=1)


    xenium_gene_expr = xenium_gene_data[:, 'CDKN1A'].to_df()['CDKN1A']

    # CODEX: use p16 intensity from .obs
    codex_gene_expr = codex_gene_data.obs['p16'].values
    codex_gene_expr = codex_gene_expr - codex_gene_expr.min()

    xenium_gene_expr = np.log1p(xenium_gene_expr)
    codex_gene_expr = np.log1p(codex_gene_expr)



    f,a = plt.subplots(1,2,figsize=(16,8))
    # # # Plot original coordinates with gene marker color
    # a[0,0].scatter(xenium_coordinate[:,0],xenium_coordinate[:,1], s=5)
    # a[0,0].scatter(codex_coordinate[:, 0], codex_coordinate[:, 1], s=5)
    # a[0, 0].set_title("Aligned cell distributions",fontsize=22)
    # a[0,1].scatter(warped_xenium_coor[:, 0], warped_xenium_coor[:, 1], s=5)
    # a[0,1].scatter(warped_codex_coor[:, 0], warped_codex_coor[:, 1], s=5)
    # a[0, 1].set_title("Registered cell distributions",fontsize=22)


    # Normalize gene expression to [0, 1]
    norm_xenium = (xenium_gene_expr - xenium_gene_expr.min()) / (xenium_gene_expr.max() - xenium_gene_expr.min())
    norm_codex = (codex_gene_expr - codex_gene_expr.min()) / (codex_gene_expr.max() - codex_gene_expr.min())

    # Convert cmap to RGBA with alpha = normalized expression
    # cmap_red = cm.get_cmap('Reds')
    cmap_red = plt.colormaps['Reds']
    colors_xenium = cmap_red(norm_xenium)
    colors_xenium[:, -1] = norm_xenium  # set alpha channel

    # cmap_blue = cm.get_cmap('Blues')
    cmap_blue = plt.colormaps['Blues']
    colors_codex = cmap_blue(norm_codex)
    colors_codex[:, -1] = norm_codex  # set alpha channel


    a[0].scatter(xenium_coordinate[:, 0], xenium_coordinate[:, 1],
                    color=colors_xenium, s=10, label='Xenium CDKN1A')
    a[0].scatter(codex_coordinate[:, 0], codex_coordinate[:, 1],
                    color=colors_codex, s=10, label='CODEX p16')


    a[0].set_title("Unregistered Marker Levels",fontsize=22)

    # Scatter with RGBA color array
    a[1].scatter(warped_xenium_coor[:, 0], warped_xenium_coor[:, 1],
                    color=colors_xenium, s=10, label='Xenium CDKN1A')

    a[1].scatter(warped_codex_coor[:, 0], warped_codex_coor[:, 1],
                    color=colors_codex, s=10, label='CODEX p16')

    a[1].set_title("Registered CODEX to Xenium with Marker Levels",fontsize=22)

    # Create normalization
    norm_xenium = mcolors.Normalize(vmin=xenium_gene_expr.min(), vmax=xenium_gene_expr.max())
    norm_codex = mcolors.Normalize(vmin=codex_gene_expr.min(), vmax=codex_gene_expr.max())

    # Create ScalarMappables (used for colorbars)
    sm_xenium = cm.ScalarMappable(cmap='Reds', norm=norm_xenium)
    sm_xenium.set_array([])  # required for colorbar

    sm_codex = cm.ScalarMappable(cmap='Blues', norm=norm_codex)
    sm_codex.set_array([])

    # Add colorbars to the subplot
    # Colorbar 1 (e.g., CDKN1A)
    cbar_ax1 = f.add_axes([0.91, 0.30, 0.01, 0.3])  # [left, bottom, width, height]
    cbar = f.colorbar(sm_xenium, cax=cbar_ax1)
    cbar.set_label('CDKN1A', fontsize=14)  # Set label font size
    cbar.ax.tick_params(labelsize=10)  # Set
    # Colorbar 2 (e.g., p16) placed lower
    cbar_ax2 = f.add_axes([0.95, 0.30, 0.01, 0.3])
    cbar = f.colorbar(sm_codex, cax=cbar_ax2)
    cbar.set_label('p16', fontsize=14)  # Set label font size
    cbar.ax.tick_params(labelsize=10)  # Set
    for ax in a.flat:
        ax.set_aspect('equal')
    plt.show()


