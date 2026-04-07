import os
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image

def main():
    print("Exporting the exact MNIST presentation images to PNGs...")
    
    transform = transforms.Compose([
        transforms.Pad(6),  # 28x28 -> 40x40
        transforms.ToTensor(),
        # We don't want to normalize here because we want to save natural-looking PNGs
    ])
    
    ds_test = torchvision.datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    # These are the exact 3 indices we baked into the ESP32 C headers
    indices_and_labels = [(18, 3), (0, 7), (3, 0)]
    
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'outputs', 'demo_images'))
    os.makedirs(output_dir, exist_ok=True)
    
    for count, (idx, expected_label) in enumerate(indices_and_labels):
        image, label = ds_test[idx]
        assert label == expected_label
        
        filename = f"sample{count + 1}_expected_{label}.png"
        out_path = os.path.join(output_dir, filename)
        
        save_image(image, out_path)
        print(f"✅ Saved visually: {out_path}")

    print("\nAwesome! You can now open these images during your presentation to show what the ESP32 is 'looking' at.")

if __name__ == "__main__":
    main()
