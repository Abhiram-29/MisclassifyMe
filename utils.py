import torch
import PIL.Image as Image
import matplotlib.pyplot as plt


def predict(model,transform,img_path,class_names,transformed = False):
    if not transformed:
        img = Image.open(img_path).convert("RGB")
        input_tensor = transform(img)
        input_batch = input_tensor.unsqueeze(0)
    else:
        input_batch = img_path.unsqueeze(0)
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
    with torch.no_grad():
        output = model(input_batch)

    _,predicted_idx = torch.max(output,1)

    predicted_class = class_names[predicted_idx.item()]
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    predicted_probability = probabilities[predicted_idx.item()].item()
    print(f"Predicted class: {predicted_class}")
    print(f"Probability: {predicted_probability:.4f}")
    return predicted_idx.item()


def show_img(torchImg, title=""):
    fig, ax = plt.subplots()
    ax.set_title(title)
    img = torchImg.permute(1, 2, 0)
    img_numpy = img.numpy()
    ax.imshow(img_numpy)
    ax.axis('off')  # Turn off the axes of the subplot
    plt.show()
