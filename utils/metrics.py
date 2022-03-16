import sys
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import lpips

def PSNR(img1, img2):
    SE_map = (1. * img1 - img2) ** 2
    cur_MSE = torch.mean(SE_map)
    return float(20 * torch.log10(1. / torch.sqrt(cur_MSE)))


def SSIM(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()

    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 /float(2 * 1.5 ** 2)) for x in range(window_size)])
    gauss = gauss / gauss.sum()
    _1D_window = gauss.unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    mu1 = F.conv2d(img1, window, padding = window_size // 2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size // 2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding = window_size // 2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding = window_size // 2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding = window_size // 2, groups = channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return float(ssim_map.mean())
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def LPIPS(img1, img2):
    # Normalize to [-1, 1]
    batch_size = 10
    img1 = 2 * (img1 / 255.) - 1
    img2 = 2 * (img2 / 255.) - 1
    loss_fn = lpips.LPIPS(net='alex')
    result = 0
    for i in range(img1.shape[0] // batch_size):
        result += float(torch.mean(loss_fn.forward(img1[i*batch_size:(i+1)*batch_size], img2[i*batch_size:(i+1)*batch_size])))
    return result / (img1.shape[0] // batch_size)


def PCVC(img1, img2):
    pass


def colorfulness(imgs):
    """
    according to the paper: Measuring colourfulness in natural images
    input is batches of ab tensors in lab space
    """
    N, C, H, W = imgs.shape
    a = imgs[:, 0:1, :, :]
    b = imgs[:, 1:2, :, :]

    a = a.view(N, -1)
    b = b.view(N, -1)

    sigma_a = torch.std(a, dim=-1)
    sigma_b = torch.std(b, dim=-1)

    mean_a = torch.mean(a, dim=-1)
    mean_b = torch.mean(b, dim=-1)

    return torch.sqrt(sigma_a ** 2 + sigma_b ** 2) + 0.37 * torch.sqrt(mean_a ** 2 + mean_b ** 2)


def cosine_similarity(img1, img2):
    input_norm = torch.norm(img1, 2, 1, keepdim=True) + sys.float_info.epsilon
    target_norm = torch.norm(img2, 2, 1, keepdim=True) + sys.float_info.epsilon
    normalized_input = torch.div(img1, input_norm)
    normalized_target = torch.div(img2, target_norm)
    cos_similarity = torch.mul(normalized_input, normalized_target)
    return float(torch.mean(cos_similarity))