
import itertools

import torch

import losses
import dataloader
from model2 import PointNet
from optimizer import Optimizer

def train(device='cuda'):

    # create our dataloader
    data = dataloader.Data(face3dmm='lm')

    # mean shape and eigenvectors for 3dmm
    # data3dmm = dataloader.SyntheticLoader()
    loader = dataloader.SyntheticLoader()
    center = torch.tensor([loader.w/2,loader.h/2,1])

    # optimizer
    optim = Optimizer(center,gt=None)
    optim.to_cuda()

    # TODO: verify if we should use different learning rates
    optim.sfm_opt = torch.optim.Adam(optim.sfm_net.parameters(),lr=1e-4)
    optim.calib_opt = torch.optim.Adam(optim.calib_net.parameters(),lr=1e-3)

    # start training
    # TODO: will this loop forever?
    for epoch in itertools.count():
        for i in range(2000):
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
            # TODO: why this isn't used during the training?
            shape_gt = batch['x_w_gt'].float()

            # forward prediction
            # K is a set of predictions of camera matrix for each batch data point of shape (100, 3, 3)
            K = optim.predict_intrinsic(x)
            # S is a set of predictions of 3D face landmarks in world coordinate system of shape (100, 68, 3)
            # TODO: why not in camera coordinate system?
            S = optim.get_shape(x)

            # compute error and step
            # calculate f error relative to GT f (part of the loss)
            f_error = torch.abs(K.mean(0)[0,0] - fgt) / fgt
            # compute reprojection error of 3D landmarks onto camera image frame
            # this uses EPnP algorithm to first find R & t of the camera and then use K to project the 3D
            # landmarks onto camera image frame - the error between this projection and x taken out of
            # the data batch is part of the loss
            s_error = losses.compute_reprojection_error(x.permute(0,2,1),S,K,show=False)

            # s_error = torch.mean(torch.pow(S - shape_gt,2).sum(1))
            loss = f_error + s_error
            loss.backward()
            optim.sfm_opt.step()
            optim.calib_opt.step()

            print(f"epoch: {epoch} | iter: {i} | f_error: {f_error.item():.3f} | f/fgt: {K.mean(0)[0,0].item():.2f}/{fgt.item():.2f} | S_err: {s_error.item():.3f} ")

        optim.save(f"{epoch:02d}_")

if __name__ == '__main__':

    train()

