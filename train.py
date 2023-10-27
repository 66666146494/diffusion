import torchvision
import torchvision.transforms as transformers
from dataset import *
from unet import *

DEVICE = torch.device("cuda")

#读取图像
# mnist_train = torchvision.datasets.FashionMNIST(root=r'./datasets', train=True, download=True, transform=transformers.ToTensor())
# mnist_test  = torchvision.datasets.FashionMNIST(root=r'./datasets', train=False, download=True, transform=transformers.ToTensor())

mnist_train = torchvision.datasets.CIFAR10(root=r'./datasets', train=True, download=True, transform=transformers.ToTensor())
mnist_test  = torchvision.datasets.CIFAR10(root=r'./datasets', train=False, download=True, transform=transformers.ToTensor())

train_dl = torch.utils.data.DataLoader(mnist_train, batch_size=16, shuffle=True)
test_dl = torch.utils.data.DataLoader(mnist_test, batch_size=256)

# transform = transforms.Compose([transforms.ToTensor()])  # 可以添加其他图像变换
#
# dataset = ImageDataset("./datasets/pokemon/images", transform=transform)
#
# data_tf = transforms.Compose(
#     [transforms.ToTensor(),
#     transforms.Normalize([0.5], [0.5])])
#
# batch_size = 4
# train_dl = DataLoader(dataset, batch_size=batch_size, shuffle=True)

def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start,beta_end,timesteps).to(DEVICE)

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

timesteps = 1000
betas = linear_beta_schedule(timesteps = timesteps).to(DEVICE)
alphas = (1 - betas).to(DEVICE)
alphas_cumprod = torch.cumprod(alphas,axis = 0).to(DEVICE)
alphas_cumprod_prev = torch.cat([torch.tensor([1]).float().to(DEVICE),alphas_cumprod[:-1]],0).to(DEVICE)
alphas_recip_alphas = torch.sqrt(1.0 / alphas).to(DEVICE)

sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).to(DEVICE)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1-alphas_cumprod).to(DEVICE)

#生成正弦位置编码
embedding_dim = 512
temb = get_sin_enc_table(timesteps,embedding_dim)

def p_losses(denoise_model,x_star,t,temb_Tensor,noise = None):
    x_star = x_star.to(DEVICE)
    t = t.to(DEVICE)
    if noise == None:
        noise = torch.randn_like(x_star)

    sqrt_alphas_cumprod_t = sqrt_alphas_cumprod[t].to(DEVICE)
    sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod[t].to(DEVICE)
    sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.view(-1, 1, 1, 1).to(DEVICE)
    sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.view(-1, 1, 1, 1).to(DEVICE)
    x_noise = (sqrt_alphas_cumprod_t * x_star + sqrt_one_minus_alphas_cumprod_t * noise).to(DEVICE)
    predicted_noise = denoise_model(x_noise,temb_Tensor)
    loss = F.mse_loss(noise,predicted_noise)

    return loss


def fit(model, trainloader, optimizer ,device):
    epochs = 50
    for epoch in range(epochs):
        print('Epoch: ', epoch + 1)
        for step, (images,_) in enumerate(trainloader):
            optimizer.zero_grad()
            batch_size = images.shape[0]
            images = images.to(device)
            # t = torch.randint(0,timesteps,(batch_size,),device=device).long()
            random_indices = random.sample(range(len(temb)), batch_size)
            t = torch.LongTensor(random_indices).to(DEVICE)
            temb_list = [temb[i] for i in random_indices]
            temb_array = np.array(temb_list)
            temb_Tensor = torch.FloatTensor(temb_array).to(DEVICE)
            loss = p_losses(denoise_model=model,x_star=images,t=t,temb_Tensor=temb_Tensor,noise=None)

            if step % 100 == 0:
                print("Step: ", step + 1, "Loss:", loss.item())
                print(loss.device)

            loss.backward()
            optimizer.step()


if __name__ == '__main__':
    DEVICE = torch.device("cuda")
    model = UNet().to(DEVICE)

    optim = torch.optim.Adam(model.parameters(), lr=5e-4)
    fit(model,train_dl,optim,DEVICE)
    torch.save(model, './diffusion_pokemon.pth')


