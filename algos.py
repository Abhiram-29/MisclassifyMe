import torch
from torchvision.transforms import transforms


transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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