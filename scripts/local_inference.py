import torch
import json
import os
import sys
from rich.console import Console
from rich.table import Table

# Add project root to path so we can import nas.*
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nas.architecture import Architecture
from nas.layers import LayerConfig

console = Console()

def run_local_demo(checkpoint_path='/tmp/best_candidate.pth', arch_path='outputs/best_arch.json'):
    console.print(f"[bold cyan]TinyML AutoNAS — Local Inference Demo[/bold cyan]\n")
    
    if not os.path.exists(arch_path):
        console.print(f"[bold red]Error:[/bold red] Architecture file not found: {arch_path}")
        console.print("Please run 'python run_nas.py search' first.")
        return

    if not os.path.exists(checkpoint_path):
        console.print(f"[bold red]Error:[/bold red] Checkpoint not found: {checkpoint_path}")
        console.print("Please run 'python run_nas.py search' first.")
        return

    # 1. Load Architecture
    with open(arch_path, 'r') as f:
        arch_dicts = json.load(f)
    arch = [LayerConfig(**d) for d in arch_dicts]
    
    # 2. Reconstruct Model
    # Note: Using 10 classes as default for demo, should match your search config
    model = Architecture(arch, num_classes=10)
    
    # 3. Load Weights
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))
        model.eval()
        console.print(f"[green]Successfully loaded model from {checkpoint_path}[/green]")
    except Exception as e:
        console.print(f"[bold red]Error loading weights:[/bold red] {e}")
        return

    # 4. Run Dummy Inference
    console.print("\n[bold yellow]Running Predictions on Sample Data:[/bold yellow]")
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Sample ID", justify="center")
    table.add_column("Predicted Class", justify="center")
    table.add_column("Confidence", justify="right")

    import torchvision
    import torchvision.transforms as transforms
    import random
    
    transform = transforms.Compose([
        transforms.Pad(6),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    console.print("\n[yellow]Downloading/Loading MNIST Test Dataset...[/yellow]")
    ds_test = torchvision.datasets.MNIST('./data', train=False, download=True, transform=transform)

    console.print("\n[bold yellow]Running Predictions on Real Data:[/bold yellow]")
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Sample ID", justify="center")
    table.add_column("True Class", justify="center")
    table.add_column("Predicted Class", justify="center")
    table.add_column("Confidence", justify="right")

    with torch.no_grad():
        # Pick 5 random images from the test set
        indices = random.sample(range(len(ds_test)), 5)
        for i, idx in enumerate(indices):
            image, true_label = ds_test[idx]
            
            # Add batch dimension: 1x40x40 -> 1x1x40x40
            dummy_input = image.unsqueeze(0)
            outputs = model(dummy_input)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            # Highlight if prediction is correct
            pred_str = str(predicted.item())
            if predicted.item() == true_label:
                pred_str = f"[green]{pred_str}[/green]"
            else:
                pred_str = f"[red]{pred_str}[/red]"
                
            table.add_row(
                f"#{i+1}", 
                str(true_label),
                pred_str, 
                f"{confidence.item()*100:.2f}%"
            )

    console.print(table)
    console.print("\n[dim]Note: Now using real images from the MNIST dataset![/dim]")

    console.print(table)
    console.print("\n[dim]Note: Predictions use dummy random data for demonstration.[/dim]")

if __name__ == "__main__":
    run_local_demo()
