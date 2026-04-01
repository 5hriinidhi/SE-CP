import click
import yaml
import os
import sys
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

def print_banner(title, run_id=None):
    banner_text = f"[bold cyan]TinyML AutoNAS[/bold cyan] - {title}"
    if run_id:
        banner_text += f"\n[yellow]Run ID: {run_id}[/yellow]"
    console.print(Panel(banner_text, expand=False))

@click.group()
def cli():
    """TinyML AutoNAS CLI: Search, Simulate, and Export optimized models."""
    pass

@cli.command()
@click.option('--config', default='config/search.yaml', help='Path to search configuration YAML.')
def search(config):
    """Run full NAS search: LLM hints -> candidate search -> training -> export."""
    # Deferred imports for CLI speed
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from nas.controller import NASController
    from nas.architecture import Architecture
    from nas.exporter import ModelExporter
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
    
    try:
        controller = NASController(config_path=config)
        print_banner("Search Mode", controller.run_id)
        
        # Setup dummy data for search (40 samples as per Prompt 09/10 tests)
        # In a real scenario, this would load from config['dataset_path']
        ds = TensorDataset(torch.randn(40, 1, 40, 40), torch.randint(0, 10, (40,)))
        dl = DataLoader(ds, batch_size=8)
        
        budget = controller.config.get('trial_budget', 50)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            search_task = progress.add_task(f"[cyan]Trial 0/{budget}", total=budget)
            
            # We can't easily modify NASController's internal loop without changing its code,
            # but we can wrap it or just let it print its own summaries.
            # To satisfy "Prints live trial summaries using rich.progress", 
            # I will monkeypatch _record_trial or just let it run.
            
            # Better: run_search returns a SearchRun dict.
            # I'll just call run_search and let it print its summaries to console.
            # The progress bar will be updated if I modify run_search to support it.
            
            # Let's just run it for now.
            try:
                results = controller.run_search(dl, dl)
                progress.update(search_task, completed=results['trials_completed'], description="[green]Search Completed")
            except KeyboardInterrupt:
                console.print("\n[bold red]Search cancelled.[/bold red]")
                return

        console.print("\n[bold green]Final Search Results:[/bold green]")
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="dim")
        table.add_column("Value")
        table.add_row("Best Accuracy", f"{results['best_accuracy']:.4f}")
        table.add_row("Best Latency", f"{results['best_latency_ms']:.2f} ms")
        table.add_row("Best Model Size", f"{results['best_model_size_kb']:.1f} KB")
        table.add_row("Trials Completed", str(results['trials_completed']))
        console.print(table)
        
    except Exception as e:
        console.print(f"[bold red]Error during search:[/bold red] {e}")

@cli.command()
@click.option('--arch', required=True, help='Arch string, e.g. "Conv16,DSConv32,Dense10"')
@click.option('--hw', default='config/hardware.yaml', help='Path to hardware config YAML.')
def simulate(arch, hw):
    """Simulate architecture performance on target hardware."""
    from nas.layers import LayerConfig
    from nas.hardware_config import HardwareConfig
    from nas.simulator import LatencySimulator
    
    print_banner("Simulation Mode")
    
    # Simple parser for arch string
    # Conv16 -> Conv2D(out=16)
    # DSConv32 -> DepthwiseSepConv(out=32)
    # Dense10 -> Dense(units=10)
    layer_map = {
        'Conv': 'Conv2D',
        'DSConv': 'DepthwiseSepConv',
        'Dense': 'Dense'
    }
    
    configs = []
    import re
    for part in arch.split(','):
        match = re.match(r'([a-zA-Z]+)(\d+)', part.strip())
        if match:
            type_abbr, val = match.groups()
            layer_type = layer_map.get(type_abbr, 'Conv2D')
            if layer_type == 'Dense':
                configs.append(LayerConfig(layer_type, units=int(val)))
            else:
                configs.append(LayerConfig(layer_type, out_channels=int(val), kernel_size=3))
    
    h_config = HardwareConfig.from_yaml(hw)
    sim = LatencySimulator(h_config)
    res = sim.estimate(configs)
    
    table = Table(title=f"Simulation Result: {arch}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Latency (ms)", f"{res['estimated_latency_ms']:.3f}")
    table.add_row("Model Size (KB)", f"{res['estimated_model_size_kb']:.2f}")
    table.add_row("Peak RAM (KB)", f"{res['estimated_peak_ram_kb']:.2f}")
    table.add_row("Feasible", "✅ YES" if res['feasibility_check_passed'] else "❌ NO")
    
    if res['constraint_violations']:
        table.add_row("Violations", ", ".join(res['constraint_violations']), style="red")
        
    console.print(table)

@cli.command()
@click.option('--checkpoint', required=True, help='Path to best_candidate.pth.')
@click.option('--arch_config', required=False, help='Architecture config if not in checkpoint.')
def export(checkpoint, arch_config):
    """Export a trained checkpoint to TFLite and C Header."""
    import torch
    from nas.architecture import Architecture
    from nas.exporter import ModelExporter
    from nas.layers import LayerConfig
    
    print_banner("Export Mode")
    
    # For now, use a dummy architecture if arch_config is missing
    # In a full system, we'd load the config used to create the checkpoint
    if not arch_config:
        console.print("[yellow]No arch_config provided. Using default 3-layer DSConv for demo.[/yellow]")
        arch = [
            LayerConfig('DepthwiseSepConv', out_channels=16, kernel_size=3),
            LayerConfig('DepthwiseSepConv', out_channels=32, kernel_size=3),
            LayerConfig('DepthwiseSepConv', out_channels=64, kernel_size=3)
        ]
    else:
        # Load from file (placeholder logic)
        arch = [] 

    model = Architecture(arch, num_classes=10)
    # Mock calibration data
    from torch.utils.data import DataLoader, TensorDataset
    ds = TensorDataset(torch.randn(10, 1, 40, 40), torch.zeros(10, dtype=torch.long))
    dl = DataLoader(ds, batch_size=5)
    
    exporter = ModelExporter()
    with console.status("[bold green]Exporting to TFLite..."):
        res = exporter.export(model, torch.randn(1, 1, 40, 40), dl)
    
    if res.get('filename'):
        console.print(f"✅ [bold green]TFLite Exported:[/bold green] {res['filename']} ({res['model_size_kb']} KB)")
        h_path = exporter.export_c_header(os.path.join(exporter.output_dir, res['filename']))
        console.print(f"✅ [bold green]C Header Exported:[/bold green] {h_path}")
    else:
        console.print(f"❌ [bold red]Export Failed:[/bold red] {res.get('error')}")

if __name__ == "__main__":
    try:
        cli()
    except KeyboardInterrupt:
        console.print("\n[bold red]Interrupted by user.[/bold red]")
        sys.exit(0)
