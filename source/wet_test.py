import os
import sys
import time
from datetime import datetime
from itertools import product
from typing import List, Tuple

import numpy as np
import torch
from numpy.typing import NDArray
from optimizer import Optimizer
from torch.utils.tensorboard import SummaryWriter

# mapping between dlib landmark indexes (list index) and Face Blaze landmark index (list value)
DLIB_2_FACE_BLAZE_MAPPING = [
    # face outline
    127,
    234,
    93,
    132,
    58,
    172,
    150,
    176,
    152,
    400,
    379,
    397,
    288,
    361,
    323,
    454,
    356,
    # left eyebrow
    70,
    63,
    105,
    66,
    107,
    # right eyebrow
    336,
    296,
    334,
    293,
    300,
    # nose
    168,
    197,
    5,
    1,
    98,
    97,
    2,
    326,
    327,
    # left eye
    33,
    160,
    158,
    133,
    153,
    144,
    # right eye
    362,
    385,
    387,
    263,
    373,
    380,
    # lips
    61,
    40,
    37,
    0,
    267,
    270,
    291,
    321,
    314,
    17, 84,
    91,
    78,
    82,
    13,
    312,
    308,
    317,
    14,
    87,
]

# NOTE: Keypoints on few of the captured frames seem a little off, so we don't take those frames into account in calculations.
VALID_FRAME_INDEXES_DATA_1 = [1, 2, 3, 4, 5, 6, 7, 8, 9]

def load_wet_data() -> Tuple[List[NDArray[np.uint8]], List[NDArray[np.float32]]]:
    """Loads data dumped using face keypoint extraction script

    Returns:
        list of frames & list of dlib's keypoints related to those frames
    """

    # data1
    # DATA_FOLDER = 'data1'
    # VALID_FRAME_INDEXES = VALID_FRAME_INDEXES_DATA_1

    # data2 - from laptop
    # DATA_FOLDER = 'data2'
    # VALID_FRAME_INDEXES = [i for i in range(100)]

    # data3 - from phone (kjduzink)
    # DATA_FOLDER = 'data3'
    # VALID_FRAME_INDEXES = [i for i in range(100)]

    # data4 - from phone (jglinko)
    DATA_FOLDER = 'data4'
    VALID_FRAME_INDEXES = [i for i in range(100)]

    frames = []
    face_blaze_keypoints_list = []

    width = None
    height = None

    for i in VALID_FRAME_INDEXES:
        filepath = os.path.abspath(os.path.abspath(os.path.join(os.path.dirname(__file__), f'../wet/{DATA_FOLDER}/frame_kpts_{i}.npy')))

        with open(filepath, 'rb') as f:
            frame = np.load(f)
            face_keypoints = np.load(f)

        if i == 0:
            width = frame.shape[1]
            height = frame.shape[0]

        frames.append(frame)
        face_blaze_keypoints_list.append(face_keypoints)

    print(f'Loaded frames count: {len(face_blaze_keypoints_list)}')
    print(f'Frame width: {width}, height: {height} [pixels]')

    # go through keypoint list and extract a subset of 68 used by the FaceCalibration
    # based on https://github.com/google/mediapipe/blob/a908d668c730da128dfa8d9f6bd25d519d006692/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
    # NOTE: landmarks used by FaceCalibration are in fact dlib's face detector keypoints

    dlib_keypoints_list = []
    for single_frame_face_keypoints in face_blaze_keypoints_list:
        dlib_keypoints_list.append(face_blaze_2_dlib_2d(single_frame_face_keypoints, width=width, height=height))

    return frames, dlib_keypoints_list


def face_blaze_2_dlib_2d(fb_landmarks: NDArray[np.float32], width:int = 640, height: int = 480, flip_coordinates: bool = False) -> NDArray[np.float32]:
    """Extract dlib's landmarks out of Face Blaze landmarks and convert to the same coordinate system
    NOTE: For some reason dlib uses a different coordinate system then MediaPipe to report landmark positions in the image.

    Args:
        fb_landmarks: FaceBlaze 478 landmarks
        width: width of image in pixels
        height: height of image in pixels

    Returns:
        dlib landmarks
    """

    assert fb_landmarks.shape == (478, 2)

    assert len(DLIB_2_FACE_BLAZE_MAPPING) == 68

    # extract dlib's landmarks from Face Blaze landmarks
    dlib_landmarks = np.array([fb_landmarks[i] for i in DLIB_2_FACE_BLAZE_MAPPING], dtype=np.float32)

    if flip_coordinates:
        # convert coordinate system to match how the FaceCalibration optimizer was trained
        dlib_landmarks[:, 0] = width - dlib_landmarks[:, 0]
        dlib_landmarks[:, 1] = height - dlib_landmarks[:, 1]
    assert dlib_landmarks.shape == (68, 2)

    return dlib_landmarks


def load_gt_camera_parameters() -> NDArray[np.float32]:
    """Loads camera matrix and other parameters obtained using standard camera calibration
    procedure (i.e. using chessboard as calibration target)

    Returns:
        GT camera matrix
    """

    # filepath = os.path.abspath(os.path.abspath(os.path.join(os.path.dirname(__file__), f'../wet/asus_tuf_f15_calibration.npz')))
    # filepath = os.path.abspath(os.path.abspath(os.path.join(os.path.dirname(__file__), f'../wet/oppo_reno_6_5g_calibration.npz')))
    filepath = os.path.abspath(os.path.abspath(os.path.join(os.path.dirname(__file__), f'../wet/samsung_s10e_calibration.npz')))

    with open(filepath, 'rb') as f:
        npz_file = np.load(f, allow_pickle=True)
        legacy_camera_matrix = npz_file['camera_matrix']
        legacy_camera_distortion = npz_file['camera_distortion']
        legacy_reprojection_error = npz_file['reprojection_error']
        legacy_camera_resolution = npz_file['camera_resolution']

    # print(f'Measured camera matrix:\n{legacy_camera_matrix}')
    # print(f'Measured camera distortion coeffs:\n{legacy_camera_distortion}')
    # print(f'Measured camera resolution:\n{legacy_camera_resolution}')
    # print(f'Measured reprojection error:\n{legacy_reprojection_error}')

    return legacy_camera_matrix


def convert_keypoints_list_to_tensor(keypoints_list: List[NDArray[np.float32]]) -> torch.Tensor:
    """Convert list of face keypoints as NumPy arrays to torch tensor, to match input type
    of FaceCalibration NN

    Args:
        keypoints_list: list of NumPy arrays corresponding to dlib keypoints identified in input data

    Returns:
        torch tensor of shape (N, 2, 68)
    """

    assert isinstance(keypoints_list, list)
    assert all([isinstance(single_frame_keypoints, np.ndarray) and len(single_frame_keypoints.shape) == 2 and single_frame_keypoints.shape[0] == 68 for single_frame_keypoints in keypoints_list])

    # convert list of NumPy arrays to NumPy array
    keypoints = np.array(keypoints_list)
    print(f'keypoints.shape: {keypoints.shape}')

    # reorder axis order to match FaceCalibration input format (N, 2, 68)
    keypoints = np.swapaxes(keypoints, 1, 2)
    assert len(keypoints.shape) == 3 and keypoints.shape[1] == 2 and keypoints.shape[2] == 68

    # convert to torch tensor
    keypoints = torch.from_numpy(keypoints)

    # TODO: Do we need to move this to CUDA or not?
    # move tensor to CUDA
    # keypoints = keypoints.cuda()
    print(f'keypoints.device: {keypoints.device}')

    return keypoints


def main():
    print('*** FaceCalibration test on WET data ***')

    # USE_OPTIMIZATION = True
    USE_OPTIMIZATION = False

    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

    # hardcode the principal point
    # NOTE: It's not really used for any camera metric estimation - just for error calculation

    # for laptop data
    # center = torch.tensor([640/2, 480/2, 1])

    # for phone data
    center = torch.tensor([480/2, 640/2, 1])

    # learning rates for K calibration net & head pose estimation net
    CALIB_LRS = [1e-2, 1e-3, 1e-4, 1e-5]
    SFM_LRS = [1e-2, 1e-3, 1e-4, 1e-5]

    GLOBAL_ITER_COUNTS = [3, 5, 10]
    CALIB_ITER_COUNTS = [3, 5, 10]
    SFM_ITER_COUNTS = [3, 5, 10]

    # define hyper parameters dict
    hparameters = dict(
        calib_lrs=CALIB_LRS,
        sfm_lrs=SFM_LRS,
        global_iter_counts=GLOBAL_ITER_COUNTS,
        calib_iter_counts=CALIB_ITER_COUNTS,
        sfm_iter_counts=SFM_ITER_COUNTS,
    )

    # define hyper-parameters sets
    hparam_values = [v for v in hparameters.values()]
    print(hparam_values)

    # test various settings of the optimizers to see what gives the best results on real data from a phone
    # go through all permutations of the hyper parameters
    for run_id, (calib_lr, sfm_lr, global_iter_count, calib_iter_count, sfm_iter_count) in enumerate(product(*hparam_values)):

        # get current timestamp tag
        date_time = datetime.fromtimestamp(time.time(), tz=None)
        timestamp_tag = date_time.strftime("%d-%m-%Y_%H:%M:%S")

        # instantiate TensorBoard's SummaryWriter object to track training progress
        comment = f'{run_id}_wet_test_{timestamp_tag}_calib_lr={calib_lr:.06f},sfm_lr={sfm_lr:.06f}_gi={global_iter_count}_ci={calib_iter_count}_si={sfm_iter_count}'
        writer = SummaryWriter(comment=comment)

        # initialize FaceCalibration's optimizer
        # NOTE: 'gt' is a dictionary that holds ground truth camera matrix & ground truth 3D face landmark locations.
        #       It's only used for error calculation so you can pass an empty dict and still run optimization algorithm

        optim = Optimizer(center, gt={}, calib_lr=calib_lr, sfm_lr=sfm_lr, tb_writer=writer)
        assert optim is not None

        # load pre-trained weights
        # legacy pre-trained model
        # token = 'legacy_pre_trained_model/00_'

        # tokens
        tokens = [
            'wet_pre_trained_model/00_',  # 1st WET trained model
            '00_fw=1.00_sw=10.00_clr=0.001000_slr=0.001000_',  # best model according to TensorBoard (last iteration)
            '00_fw=10.00_sw=10.00_clr=0.001000_slr=0.001000_',  # best model according to TensorBoard (manual trend analysis)
        ]

        PRE_TRAINED_MODEL_INDEX = 2

        optim.load(tokens[PRE_TRAINED_MODEL_INDEX])
        print('FaceCalibration\'s optimizer is ready to use')

        # load face landmarks data from MediaPipe sample app
        _, face_keypoints_list = load_wet_data()

        # convert list of NumPy arrays to a tensor
        face_keypoints_tensor = convert_keypoints_list_to_tensor(face_keypoints_list)
        print(f'face_keypoints_tensor.shape: {face_keypoints_tensor.shape}')

        # TODO: Enable to dump some test data
        # torch.save(face_keypoints_tensor, 'x_wet.pt')

        # run FaceCalibration's optimizer to get camera matrix (K) and face location (S) prediction

        # default used in paper
        # SFM_LR = 6

        # # experimental values
        # # SFM_LR = 0.01
        # SFM_LR = 0.001  # according to best training on TensorBoard
        # CALIB_LR = 0.001  # according to best training on TensorBoard

        # number of iterations for dual-optimization
        # MAX_ITER = 25

        if USE_OPTIMIZATION:
            # optimize using Alternating Optimization (AO) approach - proven to be the best one according to paper

            # TODO: Subject for removal
            # set learning rate for optimizers
            # optim.sfm_opt.param_groups[0]['lr'] = SFM_LR
            # optim.calib_opt.param_groups[0]['lr'] = CALIB_LR

            try:
                S, K, R, T = optim.dualoptimization(
                    face_keypoints_tensor,
                    global_iter_count=global_iter_count,
                    calib_iter_count=calib_iter_count,
                    sfm_iter_count=sfm_iter_count
                )
            except:
                print("OOPS! SOMETHING WENT WRONG!")
                continue

            # TODO: verify if other optimization methods give better results
            # # JO approach
            # S,K,R,T = optim.jointoptimization(face_keypoints_tensor, max_iter=100)

            # # SO approach
            # S,K,R,T = optim.sequentialoptimization(face_keypoints_tensor)

            # AO approach
            # S, K, R, T = optim.dualoptimization(face_keypoints_tensor, max_iter=5)

        else:
            # run without optimization
            K = optim.predict_intrinsic(face_keypoints_tensor)
            assert len(K.shape) == 3 and K.shape[0] == face_keypoints_tensor.shape[0] and K.shape[1:] == (3, 3)
            S = optim.get_shape(face_keypoints_tensor)
            assert len(S.shape) == 3 and S.shape[0] == face_keypoints_tensor.shape[0] and S.shape[1:] == (68, 3)

        # get predicted camera's intrinsics by averaging predictions based on all processed data frames
        f = torch.mean(K[:, 0, 0])
        px = torch.mean(K[:, 0, 2])
        py = torch.mean(K[:, 1, 2])

        # get final prediction of camera matrix
        K_avg = np.zeros((3, 3))
        K_avg[0, 0] = f
        K_avg[1, 1] = f
        K_avg[0, 2] = px
        K_avg[1, 2] = py
        K_avg[2, 2] = 1.0

        print(f'K_avg:\n{K_avg}')

        # load GT camera parameters
        K_gt = load_gt_camera_parameters()
        assert K_gt is not None and K_gt.shape == (3, 3)
        print(f'K_gt:\n{K_gt}')

        # compare with what was calculated based on legacy camera calibration procedure using checkerboard
        f_gt = K_gt[0, 0]
        f_error = abs(f - f_gt) / f_gt
        print(f'f_error: {f_error:.03f}')

        writer.add_hparams(
                {
                    'calib_lr': calib_lr,
                    'sfm_lr': sfm_lr,
                    'global_iter_count': global_iter_count,
                    'calib_iter_count': calib_iter_count,
                    'sfm_iter_count': sfm_iter_count,
                },
                {
                    'f_error': f_error,
                }
            )


if __name__ == '__main__':
    main()
