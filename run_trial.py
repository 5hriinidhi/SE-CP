import nni
import torch
import random
from torch.utils.data import DataLoader, TensorDataset
from nas.hardware_config import HardwareConfig
from nas.search_space import SearchSpace
from nas.simulator import LatencySimulator
from nas.architecture import Architecture
from nas.trainer import Trainer

def run():
    # 1. Get parameters from NNI
    # standalone run support
    params = nni.get_next_parameter()
    num_layers = params.get('num_layers', 2)
    
    # 2. Setup dummy data (40 samples, 40x40, 10 classes)
    ds = TensorDataset(torch.randn(40, 1, 40, 40), torch.randint(0, 10, (40,)))
    dl = DataLoader(ds, batch_size=8)
    
    # 3. NAS Pipeline componentes instantiation
    hw = HardwareConfig.from_yaml('config/hardware.yaml')
    hints = [] # No real LLM call as per constraints
    ss = SearchSpace(hw, hints)
    sim = LatencySimulator(hw)
    
    # Sample a candidate with the requested number of layers
    valid_candidates = [c for c in ss.candidates if len(c) == num_layers]
    if not valid_candidates:
        candidate = ss.sample()
    else:
        candidate = random.choice(valid_candidates)
        
    # 4. Latency/Fesibility Simulation
    res = sim.estimate(candidate)
    if not res['feasibility_check_passed']:
        nni.report_final_result(0.0)
        print(f"infeasible: {res['constraint_violations']}")
        return

    # 5. Architecture Build & Training
    model = Architecture(candidate, num_classes=10)
    trainer = Trainer(epochs=2)
    metrics = trainer.train(model, dl, dl)
    
    # 6. Report Result to NNI
    nni.report_final_result(metrics['val_accuracy'])
    print(f"val_accuracy: {metrics['val_accuracy']}")

if __name__ == "__main__":
    run()
