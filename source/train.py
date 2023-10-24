
import argparse
import time
from datetime import datetime
from itertools import product

import dataloader
import losses
import torch
import wet_dataloader
from wet_dataloader import ImageOrientation
from optimizer import Optimizer
from torch.utils.tensorboard import SummaryWriter

# a flag describing if we use the default synthetic dataloader (long distances of faces)
# or use WET-specific data loader which better illustrates example face landmark data recorded with a phone
# USE_WET_DATALOADER = True

# number of total epochs to run
EPOCHS_COUNT = 1

# number of synthetic batches generated in single epoch
# BATCHES_PER_EPOCH = 2000
BATCHES_PER_EPOCH = 500

# error component weights in summary loss
F_ERROR_WEIGHTS = [0.1, 1.0, 10.0]
S_ERROR_WEIGHTS = [0.1, 1.0, 10.0]

# learning rates for K calibration net & head pose estimation net
CALIB_LRS = [1e-2, 1e-3, 1e-4]
SFM_LRS = [1e-3, 1e-4, 1e-5]


def train(device: str = 'cuda',
          data_loader_type: str = 'legacy',
          orientation=ImageOrientation.PORTRAIT) -> None:
    """Main training loop.

    Args:
        device (str, optional): Device to use for training. Defaults to 'cuda'.
        data_loader_type (str, optional): Data loader to use. Defaults to 'legacy'.
        orientation (ImageOrientation, optional): Device orientation. Defaults to ImageOrientation.PORTRAIT.
    """

    # define hyper parameters dict
    hparameters = dict(
        f_error_weights=F_ERROR_WEIGHTS,
        s_error_weights=S_ERROR_WEIGHTS,
        calib_lrs=CALIB_LRS,
        sfm_lrs=SFM_LRS
    )

    # define hyper-parameters sets
    hparam_values = [v for v in hparameters.values()]
    print(hparam_values)

    # go through all permutations of the hyper parameters
    for run_id, (f_error_weight, s_error_weight, calib_lr, sfm_lr) in enumerate(product(*hparam_values)):

        # get current timestamp tag
        date_time = datetime.fromtimestamp(time.time(), tz=None)
        timestamp_tag = date_time.strftime("%d-%m-%Y_%H:%M:%S")

        # instantiate TensorBoard's SummaryWriter object to track training progress
        data_tag = 'wet' if data_loader_type=='wet' else 'legacy'
        comment = f'id_{run_id}_{timestamp_tag},data={data_tag},orient={orientation.value},f_w={f_error_weight:.02f},s_w={s_error_weight:.02f},calib_lr={calib_lr:.06f},sfm_lr={sfm_lr:.06f}'
        writer = SummaryWriter(comment=comment)

        # placeholders
        loader = None
        center = None

        if data_loader_type == 'legacy':
            loader = dataloader.SyntheticLoader()
            center = torch.tensor([loader.w / 2, loader.h / 2, 1])
        elif data_loader_type == 'wet':
            loader = wet_dataloader.WetSyntheticLoader(image_orientation=orientation)
            center = torch.tensor([loader.camera_frame_width_pixels / 2, loader.camera_frame_height_pixels / 2, 1])
        else:
            raise ValueError(f'Unsupported data loader type: {data_loader_type}')

        # instantiate optimizer
        optim = Optimizer(center, gt=None)
        optim.to_cuda()

        # setup parameters & learning rates
        optim.sfm_opt = torch.optim.Adam(optim.sfm_net.parameters(), lr=sfm_lr)
        optim.calib_opt = torch.optim.Adam(optim.calib_net.parameters(), lr=calib_lr)

        # start training
        for epoch in range(EPOCHS_COUNT):
            for i in range(BATCHES_PER_EPOCH):
                # batch is a dict which holds the following keys:
                # 1. 'alpha_gt', torch tensor of shape: (199, 1) holding ???
                # 2. 'x_w_gt', torch tensor of shape: (68, 3) holding 3D landmark coordinates in world coordinate system ???
                # 3. 'x_cam_gt', torch tensor of shape: (100, 3, 68) holding 3D landmark coordinates in camera coordinate system
                # 4. 'x_img', torch tensor of shape: (100, 2, 68) holding 2D landmark coordinates in camera image frame
                # 5. 'x_img_gt', torch tensor of shape: (100, 2, 68) holding 2D landmark coordinates in camera image
                #    frame (whats the difference with the previous one ???)
                # 6. 'f_gt', torch tensor of shape: (1) holding GT f

                batch = loader[i]
                optim.sfm_opt.zero_grad()
                optim.calib_opt.zero_grad()
                # extract 2D landmark locations in image frame
                x = batch['x_img'].float()

                # # TODO: Enable to dump some test data
                # torch.save(x, f'x_epoch_{epoch}_iter_{i}.pt')

                # extract GT f to calculate loss
                fgt = batch['f_gt'].float()

                # forward prediction
                # K is a set of predictions of camera matrix for each batch data point of shape (100, 3, 3)
                K = optim.predict_intrinsic(x)
                # S is a set of predictions of 3D face landmarks in world coordinate system of shape (100, 68, 3)
                # TODO: why not in camera coordinate system?
                S = optim.get_shape(x)

                # compute error and step
                # calculate f error relative to GT f (part of the loss)
                f_error = torch.abs(K.mean(0)[0, 0] - fgt) / fgt
                # compute reprojection error of 3D landmarks onto camera image frame
                # this uses EPnP algorithm to first find R & t of the camera and then use K to project the 3D
                # landmarks onto camera image frame - the error between this projection and x taken out of
                # the data batch is part of the loss
                s_error = losses.compute_reprojection_error(x.permute(0, 2, 1), S, K, show=False)

                # calculate total loss
                loss = f_error_weight*f_error + s_error_weight*s_error

                # log f error, s error and summary loss
                writer.add_scalar('error/f_error', f_error, epoch*BATCHES_PER_EPOCH + i)
                writer.add_scalar('error/s_error', s_error, epoch*BATCHES_PER_EPOCH + i)
                writer.add_scalar('loss/train', loss, epoch*BATCHES_PER_EPOCH + i)

                loss.backward()
                optim.sfm_opt.step()
                optim.calib_opt.step()

                print(f"epoch: {epoch} | iter: {i} | f_error: {f_error.item():.3f} | f/fgt: {K.mean(0)[0,0].item():.2f}/{fgt.item():.2f} | S_err: {s_error.item():.3f} ")

            # store the model on disk
            optim.save(f'{epoch:02d}_orient={orientation.value}_fw={f_error_weight:.02f}_sw={s_error_weight:.02f}_clr={calib_lr:06f}_slr={sfm_lr:.06f}_')

        # log hyper-parameters
        writer.add_hparams(
                {
                    'f_error_weight': f_error_weight,
                    's_error_weight': s_error_weight,
                    'calib_lr': calib_lr,
                    'sfm_lr': sfm_lr,
                },
                {
                    'f_error': f_error,
                    's_error': s_error,
                    'loss': loss,
                }
            )


if __name__ == '__main__':

    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-loader', type=str, default='legacy', help='Data loader to use: legacy or wet')
    parser.add_argument('--orientation', type=str, default='portrait', help='Device orientation: portrait or landscape')
    args = parser.parse_args()

    if args.data_loader == 'legacy':
        print('[WARNING] Desired image orientation will be ignored since you are not using WET data loader')
    elif args.data_loader == 'wet':
        print(f'Using device orientation: {args.orientation}')
    else:
        raise ValueError(f'Unsupported data loader: {args.data_loader}')

    # map device orientation tag to the right enum value
    if args.orientation == 'portrait':
        orientation = ImageOrientation.PORTRAIT
    elif args.orientation == 'landscape':
        orientation = ImageOrientation.LANDSCAPE
    else:
        raise ValueError(f'Unsupported device orientation: {args.orientation}')

    # run the main training loop
    train(data_loader_type=args.data_loader, orientation=orientation)
