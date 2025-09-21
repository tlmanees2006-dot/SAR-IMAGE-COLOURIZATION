import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import os

# Define the diffusion model architecture (EXACTLY as you provided)
class DiffusionModel(nn.Module):
    def __init__(self):
        super(DiffusionModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        bottleneck = self.bottleneck(encoded)
        decoded = self.decoder(bottleneck)
        return decoded

# Function to get all image paths in a directory (EXACTLY as you provided)
def get_image_paths(root_dir):
    image_paths = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".png"):
                image_paths.append(os.path.join(root, file))
    return image_paths

# Load the trained model
model = DiffusionModel()
model.load_state_dict(torch.load(r"/content/drive/MyDrive/Colab Notebooks/sar_colorization_model.pth"))
model.eval()  # Set the model to evaluation mode

# Set device (use CPU if CUDA is not available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Function to load grayscale images for testing (EXACTLY as you provided)
def load_grayscale_images(test_dir):
    image_paths = [os.path.join(test_dir, file) for file in os.listdir(test_dir) if file.endswith(".jpg") or file.endswith(".png")]
    transform = transforms.Compose([transforms.ToTensor()])
    images = [transform(Image.open(img_path).convert('L')).unsqueeze(0) for img_path in image_paths]  # Convert to 4D tensor (batch size 1)
    return images, image_paths

# Directory containing test grayscale images
test_grayscale_dir = r"/content/drive/MyDrive/CIFAR-10/bw"

# Load the test images
test_images, image_paths = load_grayscale_images(test_grayscale_dir)

# Run inference and save the colorized images
output_dir = r"/content/drive/MyDrive/CIFAR-10/colorized_output"
os.makedirs(output_dir, exist_ok=True)

with torch.no_grad():
    for img_tensor, img_path in zip(test_images, image_paths):
        img_tensor = img_tensor.to(device)
        output = model(img_tensor)
        output = output.squeeze(0)  # Remove batch dimension
        output_image_path = os.path.join(output_dir, os.path.basename(img_path))
        save_image(output, output_image_path)
        print(f"Colorized image saved to {output_image_path}")

print("Inference completed and colorized images saved.")