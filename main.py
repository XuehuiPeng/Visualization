import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad
from pytorch_grad_cam.ablation_layer import AblationLayerVit
from torchvision import transforms
from timm.models import create_model
from utils import THRESH, show_cam_on_image
from model.resnet50 import ResNet50
from model.ViT import ViT
from model.SwinT import SwinTransformer
from model.Deit import deit_t_distilled


def subnet_to_dense(subnet_dict, p):
    """
        Convert a subnet state dict (with subnet layers) to dense i.e., which can be directly
        loaded in network with dense layers.
    """
    dense = {}

    # load dense variables
    for (k, v) in subnet_dict.items():
        if "popup_scores" not in k:
            dense[k] = v

    # update dense variables
    for (k, v) in subnet_dict.items():
        if "popup_scores" in k:
            s = torch.abs(subnet_dict[k])

            out = s.clone()
            _, idx = s.flatten().sort()
            j = int((1 - p) * s.numel())

            flat_out = out.flatten()
            flat_out[idx[:j]] = 0
            flat_out[idx[j:]] = 1
            dense[k.replace("popup_scores", "weight")] = (
                    subnet_dict[k.replace("popup_scores", "weight")] * out
            )
    return dense


def reshape_transform(tensor):
    height = args.W_H
    width = args.W_H
    if args.model_type == "SwinT":
        result = tensor.reshape(tensor.size(0),
                                height, width, tensor.size(2))
    else:
        result = tensor[:, args.token:, :].reshape(tensor.size(0),
                                                   height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def visualize(rgb, grayscale):
    # Here grayscale_cam has only one image in the batch
    mask = THRESH(rgb)
    grayscale = grayscale[0, :]
    image = show_cam_on_image(rgb, mask, grayscale, use_rgb=True)

    # cam_image is RGB encoded whereas "cv2.imwrite" requires BGR encoding.(cv2.imwrite保存格式为BGR)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--aug_smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument(
        '--eigen_smooth',
        action='store_true',
        help='Reduce noise by taking the first principle component'
             'of cam_weights*activations')
    parser.add_argument(
        '--image_path',
        type=str,
        default='input/Covid_2.png',
        help='Input image path')
    parser.add_argument(
        '--model_path',
        type=str,
        default="")
    parser.add_argument(
        '--model_type',
        type=str,
        default="DeiT_distilled",
        choices=["resnet50", "ViT", "DeiT", "DeiT_distilled", "SwinT"])
    parser.add_argument('--compress_K', type=int,
                        default=0.01,
                        help='base: compress_K=1')
    parser.add_argument('--W_H', type=int,
                        default=14,
                        help='W_H=7: ViT, deit_tiny_patch16_224 W_H=7, SwinT'
                             'W_H=14: deit_tiny_distilled_patch16_224')
    parser.add_argument('--token', type=int,
                        default=2,
                        help='token=1: ViT, deit_tiny_patch16_224 '
                             'token=2: deit_tiny_distilled_patch16_224')
    parser.add_argument(
        '--output_path',
        type=str,
        default='output',
        help='Output image path')

    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')
    return args


if __name__ == '__main__':
    """ python cam.py -image-path <path_to_image>
    Example usage of loading an image, and computing:
        1. CAM
        2. Guided Back Propagation
        3. Combining both
    """

    args = get_args()
    methods = \
        {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM,
         "layercam": LayerCAM,
         "fullgrad": FullGrad}

    if args.model_type == "resnet50":
        checkpoint = torch.load(args.model_path)
        model = ResNet50(nn.Conv2d, nn.Linear, 'kaiming_normal')  # Hydra resnet50
        model.load_state_dict(subnet_to_dense(checkpoint["state_dict"], args.compress_K), strict=False)
        target_layers = [model.layer4[-1]]
    elif args.model_type == "ViT":
        model = ViT(
            image_size=224,
            patch_size=32,  # image_size must be divisible by patch_size
            num_classes=2,
            dim=1024,  # Last dimension of output tensor after linear transformation nn.Linear(..., dim)
            depth=6,  # Number of Transformer blocks
            heads=16,  # Number of heads in Multi-head Attention layer
            mlp_dim=2048,  # Dimension of the MLP (FeedForward) layer
            dropout=0.1,
            emb_dropout=0.1  # Embedding dropout rate (0-1)
        )
        checkpoint = torch.load(args.model_path)
        model.load_state_dict(checkpoint["state_dict"], False)
        model.eval()
        target_layers = [model.transformer]
    elif args.model_type == "DeiT":
        model = create_model("deit_tiny_patch16_224", pretrained=False, num_classes=2, img_size=224)
        checkpoint = torch.load(args.model_path, map_location='cpu')
        model.load_state_dict(checkpoint["model_ema"], strict=False)
        model.eval()
        target_layers = [model.blocks[-1].norm1]
    elif args.model_type == "DeiT_distilled":
        model = deit_t_distilled(cl=nn.Conv2d, ll=nn.Linear, num_classes=2)
        checkpoint = torch.load(args.model_path, map_location='cpu')
        model.load_state_dict(subnet_to_dense(checkpoint["state_dict"], args.compress_K), strict=False)
        model.eval()
        target_layers = [model.blocks[-1].norm1]
    elif args.model_type == "SwinT":
        model = SwinTransformer(in_chans=3,
                                patch_size=4,
                                window_size=7,
                                embed_dim=96,
                                depths=(2, 2, 6, 2),
                                num_heads=(3, 6, 12, 24),
                                num_classes=2,
                                linear_layer=nn.Linear)
        checkpoint = torch.load(args.model_path, map_location='cuda')
        model.load_state_dict(checkpoint["state_dict"], False)
        model.eval()
        target_layers = [model.layers[-1].blocks[-1].norm2]

    if args.use_cuda:
        model = model.cuda()

    # Choose the target layer you want to compute the visualization for.
    # Usually this will be the last convolutional layer in the model.
    # Some common choices can be:
    # Resnet18 and 50: model.layer4
    # VGG, densenet161: model.features[-1]
    # mnasnet1_0: model.layers[-1]
    # You can print the model to help chose the layer
    # You can pass a list with several target layers,
    # in that case the CAMs will be computed per layer and then aggregated.
    # You can also try selecting all layers of a certain type, with e.g:
    # from pytorch_grad_cam.utils.find_layers import find_layer_types_recursive
    # find_layer_types_recursive(model, [torch.nn.ReLU])
    rgb_img = cv2.imread(args.image_path, 1)[:, :, ::-1]
    rgb_img = cv2.resize(rgb_img, dsize=(224, 224))
    rgb_img = np.float32(rgb_img) / 255
    preprocess = transforms.ToTensor()
    input_tensor = preprocess(rgb_img.copy()).unsqueeze(0)

    # We have to specify the target we want to generate
    # the Class Activation Maps for.
    # If targets is None, the highest scoring category (for every member in the batch) will be used.
    # You can target specific categories by
    # targets = [e.g ClassifierOutputTarget(281)]
    targets = None
    # Using the with statement ensures the context is freed, and you can
    # recreate different CAM objects in a loop.
    if args.model_type == "resnet50":
        method_choices = ['gradcam', 'gradcam++',
                          'xgradcam', 'ablationcam',
                          'eigencam', 'eigengradcam',
                          'layercam', 'fullgrad', 'scorecam']
        for method_name in method_choices:
            cam_algorithm = methods[method_name]
            with cam_algorithm(model=model,
                               target_layers=target_layers,
                               use_cuda=args.use_cuda) as cam:
                # AblationCAM and ScoreCAM have batched implementations.
                # You can override the internal batch size for faster computation.
                cam.batch_size = 32
                grayscale_cam = cam(input_tensor=input_tensor,
                                    targets=targets,
                                    aug_smooth=args.aug_smooth,
                                    eigen_smooth=args.eigen_smooth)
                cam_image = visualize(rgb_img, grayscale_cam)
                cv2.imwrite(f'{args.output_path}/{method_name}_cam.jpg', cam_image)

    else:
        method_choices = ['gradcam', 'gradcam++',
                          'xgradcam', 'eigencam',
                          'eigengradcam', 'layercam', 'scorecam']
        for method_name in method_choices:
            if method_name not in methods:
                raise Exception(f"Method {method_name} not implemented")

            if method_name == "ablationcam":
                cam = methods[method_name](model=model,
                                           target_layers=target_layers,
                                           use_cuda=args.use_cuda,
                                           reshape_transform=reshape_transform,
                                           ablation_layer=AblationLayerVit())
            else:
                cam = methods[method_name](model=model,
                                           target_layers=target_layers,
                                           use_cuda=args.use_cuda,
                                           reshape_transform=reshape_transform)
            cam.batch_size = 32

            grayscale_cam = cam(input_tensor=input_tensor,
                                targets=targets,
                                eigen_smooth=args.eigen_smooth,
                                aug_smooth=args.aug_smooth)
            cam_image = visualize(rgb_img, grayscale_cam)
            cv2.imwrite(f'{args.output_path}/{method_name}_cam.jpg', cam_image)
