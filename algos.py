import torch
from torchvision.transforms import transforms


transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

pretransform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

def fgsm(image, model, loss_fn,label=None,targeted = False,eps = 0.01):
    #Fast Gradient sign methord to generate adversarial images
    if targeted:
        input_img =image
    else:
        input_tensor = transform(image)
        input_img = input_tensor.unsqueeze(0)
    if torch.cuda.is_available():
        input_img = input_img.to('cuda')
    input_img.requires_grad_(True)

    if label is None:
        _,label = torch.max(model(input_img),1)
    print(label)

    loss = loss_fn(model(input_img),label)
    if targeted:
        loss = -loss
    grad = torch.autograd.grad(loss, input_img)[0]
    perturbations = torch.sign(grad) 
    adv_x = image+eps*perturbations[0]

    return adv_x

def pgd(image,target,model,loss_fn,steps,alpha,eps=0.01):
    "Implementation of projected gradient descent to create adversarial images"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    target = torch.tensor([target],device=device )
    x = pretransform(image)
    x = x.unsqueeze(0)
    x = x.to(device)
    x_adv = copy.deepcopy(x)
    eta = torch.zeros_like(x_adv).to(device)
    for i in range(steps):
        x_adv = algos.fgsm(x_adv,model,loss_fn,label=target,targeted=True,eps=eps)
        eta = x_adv-x
        eta = clip_eps(eta,eps)
        x_adv = x+eta
    return x_adv