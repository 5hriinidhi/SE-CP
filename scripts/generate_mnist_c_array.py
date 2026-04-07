import os
import torch
import torchvision
import torchvision.transforms as transforms

def main():
    print("Downloading MNIST and generating C arrays...")
    
    transform = transforms.Compose([
        transforms.Pad(6),  # 28x28 -> 40x40
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    ds_test = torchvision.datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    # Let's pick three specific digits from the test set for the demo
    # In MNIST test set: idx 18 is a '3', idx 0 is a '7', idx 3 is a '0'
    indices_and_labels = [(18, 3), (0, 7), (3, 0)]
    
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'firmware', 'esp32_inference'))
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, 'mnist_samples.h')
    
    with open(out_path, 'w') as f:
        f.write("// Auto-generated MNIST test samples for ESP32 Presentation Demo\n")
        f.write("#ifndef MNIST_SAMPLES_H\n")
        f.write("#define MNIST_SAMPLES_H\n\n")
        
        f.write("const int NUM_TEST_SAMPLES = 3;\n")
        f.write("const int SAMPLE_SIZE = 40 * 40; // 1600 floats\n\n")
        
        f.write("const float test_samples[3][1600] = {\n")
        
        for count, (idx, expected_label) in enumerate(indices_and_labels):
            image, label = ds_test[idx]
            assert label == expected_label
            
            # Flatten the 1x40x40 tensor to a 1D array of 1600 elements
            flat = image.view(-1).numpy()
            
            f.write(f"  // Sample {count + 1}: A handwritten '{label}'\n  {{\n    ")
            for i, val in enumerate(flat):
                f.write(f"{val:.4f}f")
                if i < len(flat) - 1:
                    f.write(", ")
                if (i + 1) % 10 == 0:
                    f.write("\n    ")
            f.write("\n  }")
            if count < 2:
                f.write(",\n\n")
            else:
                f.write("\n")
                
        f.write("};\n\n")
        
        # Array storing the true labels
        f.write("const int test_labels[3] = {3, 7, 0};\n\n")
        
        f.write("#endif // MNIST_SAMPLES_H\n")
        
    print(f"✅ Generated 3 MNIST test samples in: {out_path}")

if __name__ == "__main__":
    main()
