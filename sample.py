from tqdm import tqdm, trange
import matplotlib.pyplot as plt
from unet import *
from torchvision.transforms import ToPILImage
import imageio

import imageio
def compose_gif(img_paths):
    gif_images = []
    for path in img_paths:
        gif_images.append(path)
    imageio.mimsave("test.gif",gif_images,fps=1)


DEVICE = torch.device("cuda")


def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start,beta_end,timesteps).to(DEVICE)

timesteps = 1000
betas = linear_beta_schedule(timesteps = timesteps).to(DEVICE)
alphas = (1 - betas).to(DEVICE)
alphas_cumprod = torch.cumprod(alphas,axis = 0).to(DEVICE)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1],(1,0),value=1.0).to(DEVICE)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas).to(DEVICE)

sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).to(DEVICE)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1-alphas_cumprod).to(DEVICE)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod).to(DEVICE)

#生成正弦位置编码
embedding_dim = 512
temb = get_sin_enc_table(timesteps,embedding_dim)
temb = np.array(temb)
temb = torch.FloatTensor(temb).to(DEVICE)

def p_sample(model, img, t ,t_index):
    betas_t = betas[t].to(DEVICE)
    sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod[t].to(DEVICE)
    sqrt_recip_alphas_t = sqrt_recip_alphas[t].to(DEVICE)
    betas_t = betas_t.view(-1, 1, 1, 1).to(DEVICE)
    sqrt_recip_alphas_t = sqrt_recip_alphas_t.view(-1, 1, 1, 1).to(DEVICE)
    sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.view(-1, 1, 1, 1).to(DEVICE)
    temb_t = temb[t].to(DEVICE)

    model_mean = sqrt_recip_alphas_t*(
            img - betas_t*model(img,temb_t)/sqrt_one_minus_alphas_cumprod_t
    ).to(DEVICE)

    if t_index == 0:
        return  model_mean
    else:
        posterior_variance_t = posterior_variance[t].to(DEVICE)
        noise = torch.randn_like(img).to(DEVICE)
        posterior_sqrt_variance_t = torch.sqrt(posterior_variance_t).to(DEVICE)
        posterior_sqrt_variance_t = posterior_sqrt_variance_t.view(-1, 1, 1, 1).to(DEVICE)
        return model_mean + posterior_sqrt_variance_t * noise

def p_sample_loop(model,b,c,w,h):
    img = torch.randn((b,c,w,h),device= DEVICE)
    imgs = []

    for i in tqdm(reversed(range(0,timesteps)),desc="sampling loop time step",total=timesteps):
        img = p_sample(model , img, torch.full((b,),i,device=DEVICE,dtype=torch.long), i)
        imgs.append(img.cpu().detach().numpy())
        if i == 0:
            result = img.cpu().detach().numpy()
    return imgs , result

if __name__ == '__main__':
    b = 16; c = 1; w = 28; h = 28
    model = torch.load('./diffusion.pth')  # 加载模型
    model = model.to(DEVICE)
    model.eval()  # 把模型转为test模式
    with torch.no_grad():
        imgs , result = p_sample_loop(model,b,c,w,h)

    ims = []
    fig = plt.figure()
    n = 1

    for i in range(timesteps):
        if (i+1)%50 == 0:
            # plt.subplot(4, 5, n)
            # im = plt.imshow(imgs[i][3].reshape(28, 28, 1).squeeze(), cmap="gray", animated=True)
            # ims.append(imgs[i][3].reshape(28, 28, 1).squeeze())
            # n += 1
            plt.subplot(4, 5, n)
            a = imgs[i][3]
            b = np.transpose(a,(1,2,0))
            d = (b - np.min(b)) / (np.max(b) - np.min(b))
            _,_,c = d.shape
            if c == 1:
                im = plt.imshow(d.squeeze(), cmap="gray", animated=True)
            else:
                im = plt.imshow(d)
            n += 1
    plt.axis('off')
    plt.show()