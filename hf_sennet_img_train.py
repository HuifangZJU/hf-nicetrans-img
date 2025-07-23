import os
import sys
import seaborn as sns
import time
import torch
from argparse import ArgumentParser
from datasets import Sennet_image_pair
from matplotlib import pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import losses
import datetime
from tensorboardX import SummaryWriter
import numpy as np
from hf_img_networks import NICE_Trans_img
from itertools import cycle


def min_max_normalize(array):
    min = np.min(array)   # shape (C,)
    max = np.max(array)   # shape (C,)
    range = max - min

    normalized_data = (array - min) / range
    return normalized_data


def create_overlay(fixed, moving):
    fixed = min_max_normalize(fixed.detach().cpu().squeeze().numpy())
    moving = min_max_normalize(moving.detach().cpu().squeeze().numpy())
    # fixed = min_max_normalize(fixed)
    # moving = min_max_normalize(moving)
    overlay = np.zeros((fixed.shape[0], fixed.shape[1], 3))
    overlay[..., 0] = fixed  # Red
    overlay[..., 1] = moving  # Green
    overlay[..., 2] = moving  # Blue
    return overlay

def save_image_registrations(fixed,moving,affined,warpped,epoch,i):

    original = create_overlay(fixed,moving)
    after_affine = create_overlay(fixed,affined)
    final_result = create_overlay(fixed,warpped)

    # Plot results
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))

    axes[0].imshow(original, cmap='gray')
    axes[0].set_title("Original displacement")

    axes[1].imshow(after_affine)
    axes[1].set_title("Overlay After Affine transformation (Magenta/Cyan)")

    axes[2].imshow(final_result)
    axes[2].set_title("Overlay After Deformable Registration (Magenta/Cyan)")

    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(image_save_path + str(epoch)+'_'+str(i) +'_image.png')
    plt.close()
    # plt.show()

def validation(epoch,i, cyclic_loader):
    model.eval()
    test_batch = next(cyclic_loader)  # just one pair per call

    with torch.no_grad():
        fixed_image = test_batch['fixed_image'].to(device).float()
        moving_image = test_batch['moving_image'].to(device).float()

        fixed_image = fixed_image.unsqueeze(dim=2)
        moving_image = moving_image.unsqueeze(dim=2)

        warped_image, flow, affined_image, affine_para = \
            model(fixed_image, moving_image)

        warped_image = warped_image.squeeze(dim=2)
        affined_image = affined_image.squeeze(dim=2)

        save_image_registrations(fixed_image, moving_image, affined_image, warped_image, epoch,i)





parser = ArgumentParser()

parser.add_argument("--model_dir", type=str,
                    dest="model_dir", default='sennet_test',
                    help="models folder")
# parser.add_argument("--load_model", type=str,
#                     dest="load_model", default='/media/huifang/data/experiment/registration/DLPFC_all_pairwise_align_tissue_mask_hvg/saved_models/model_200.pth',
#                     help="load model file to initialize with")
parser.add_argument("--load_model", type=str,
                    dest="load_model", default='./',
                    help="load model file to initialize with")
parser.add_argument("--initial_epoch", type=int,
                    dest="initial_epoch", default=0,
                    help="initial epoch")
parser.add_argument("--epochs", type=int,
                    dest="epochs", default=200,
                    help="number of epoch")
parser.add_argument("--batch_size", type=int,
                    dest="batch_size", default=1,
                    help="batch size")
args = parser.parse_args()



experiment_path = '/media/huifang/data/experiment/registration/sennet/'
image_save_path = os.path.join(experiment_path,args.model_dir) + '/images/'
model_save_path = os.path.join(experiment_path,args.model_dir)+ '/saved_models/'
log_save_path = os.path.join(experiment_path,args.model_dir) + '/logs/'
os.makedirs(image_save_path, exist_ok=True)
os.makedirs(model_save_path, exist_ok=True)
os.makedirs(log_save_path, exist_ok=True)



# img_transforms = transforms.Compose([
#     transforms.Resize((512,512)),                      # Resize image
#     transforms.ToTensor(),                               # Convert to tensor (C, H, W)
#     transforms.Normalize(mean=(0.5, 0.5, 0.5),            # Normalize 3 channels
#                          std=(0.5, 0.5, 0.5))
# ])

img_transforms = transforms.Compose([transforms.ToTensor(),
                 transforms.Resize((512,512),antialias=True),
                  transforms.Normalize((0.5), (0.5))])


# debug_dataset = Sennet_image_pair(datalist="/media/huifang/data/sennet/xenium_codex_pairs.txt",image_transformer=img_transforms)
# test = debug_dataset.__getitem__(1)
# moving_image = np.asarray(test['moving_image'])
# fixed_image = np.asarray(test['fixed_image'])
# f,a = plt.subplots(1,2)
# a[0].imshow(moving_image)
# a[1].imshow(fixed_image)
# plt.show()
#
# plt.imshow(create_overlay(moving_image,fixed_image))
# plt.show()

train_dataloader = DataLoader(Sennet_image_pair(datalist="/media/huifang/data/sennet/xenium_codex_pairs.txt",image_transformer=img_transforms),batch_size=args.batch_size, shuffle=True, num_workers=16)
test_dataloader = DataLoader(Sennet_image_pair(datalist="/media/huifang/data/sennet/xenium_codex_pairs.txt",image_transformer=img_transforms),batch_size=args.batch_size, shuffle=False, num_workers=16)
cyclic_test_loader = cycle(test_dataloader)
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


if args.load_model != './':
    print('loading', args.load_model)
    # state_dict = torch.load(args.load_model, map_location=device)
    # model.load_state_dict(state_dict)

    checkpoint = torch.load(args.load_model, map_location="cpu")
    model.load_state_dict(checkpoint)
    model.to("cuda")
else:
    model.to(device)
# transfer model
SpatialTransformer = model.SpatialTransformer
SpatialTransformer.to(device)
SpatialTransformer.eval()



AffineTransformer = model.AffineTransformer
AffineTransformer.to(device)
AffineTransformer.eval()


# set optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=2*1e-4)
# prepare losses
# img_warpping_loss = losses.MaskedNCC(win=5).loss
img_warpping_loss = losses.NCC(win=51).loss
# displacement_loss = losses.Regu_loss
displacement_loss = losses.ReguLossCurvature(alpha=1.0, beta=0.1)


prev_time = time.time()
logger = SummaryWriter(log_save_path)
# training/validate loops
for epoch in range(args.initial_epoch, args.epochs):
    start_time = time.time()
    # training
    model.train()
    train_losses = []
    train_total_loss = []
    for i, batch in enumerate(train_dataloader):
        fixed_image = batch['fixed_image'].to(device).float()
        moving_image = batch['moving_image'].to(device).float()



        fixed_image = fixed_image.unsqueeze(dim=2)
        moving_image = moving_image.unsqueeze(dim=2)

        warped_image, flow, affined_image, affine_para = \
            model(fixed_image, moving_image)

        # calculate total loss
        deformable_image_loss = img_warpping_loss(fixed_image,warped_image)
        affine_image_loss = img_warpping_loss(fixed_image,affined_image)

        field_loss = 1*displacement_loss(flow)
        image_loss = 1*(deformable_image_loss + affine_image_loss)



        loss = image_loss + field_loss

        # backpropagate and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Determine approximate time left
        batches_done = epoch * len(train_dataloader) + i
        batches_left = args.epochs * len(train_dataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        # Print log
        sys.stdout.write(
            "\r" + args.model_dir + "---[Epoch %d/%d] [Batch %d/%d] [Total loss: %f] [Image loss: %f] [Regu loss: %f]  ETA: %s" %
            (epoch, args.epochs,
             i, len(train_dataloader),
             loss.item(),image_loss.item(),field_loss.item(), time_left))
        # # --------------tensor board--------------------------------#
        if batches_done % 20 == 0:
            info = {'loss': loss.item(),'deformable_image_loss':deformable_image_loss,'affine_image_loss':affine_image_loss}
            for tag, value in info.items():
                logger.add_scalar(tag, value, batches_done)
            for tag, value in model.named_parameters():
                tag = tag.replace('.', '/')
                logger.add_histogram(tag, value.data.cpu().numpy(), batches_done)
                # logger.add_histogram(tag+'grad', value.grad.data.cpu().numpy(),batches_done+1)
        if batches_done % 10 ==0:
            validation(epoch,i,cyclic_test_loader)
    # save model checkpoint
    if epoch % 50 == 0:
        torch.save(model.state_dict(), model_save_path+'/model_%d.pth' % (epoch))



