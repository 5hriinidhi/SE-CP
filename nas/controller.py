import yaml
import secrets
import nni
import time
from nas.hardware_config import HardwareConfig
from nas.llm_advisor import LLMAdvisor
from nas.search_space import SearchSpace
from nas.simulator import LatencySimulator
from nas.architecture import Architecture
from nas.trainer import Trainer

class NASController:
    """
    NASController orchestrates the full pipeline:
    LLM Hints -> Search Space -> Simulator -> Training -> Results.
    """
    def __init__(self, config_path: str = 'config/search.yaml'):
        # 1. Load Search Config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # 2. Load Hardware Config
        self.hw = HardwareConfig.from_yaml('config/hardware.yaml')
        
        # 3. Instantiate Sub-modules
        self.advisor = LLMAdvisor(
            model=self.config.get('llm_model', 'claude-3-haiku-20240307')
        )
        self.simulator = LatencySimulator(self.hw)
        self.trainer = Trainer(epochs=self.config.get('epochs', 20))
        
        # 4. State
        self.run_id = f"run_{secrets.token_hex(4)}"
        self.best_accuracy = 0.0
        self.best_latency = 0.0
        self.best_size = 0.0
        self.best_candidate_id = None
        self.best_candidate_cfg = None
        self.trials_completed = 0
        self.hints = []


    def run_search(self, train_loader, val_loader) -> dict:
        print(f"Starting NAS Search Run: {self.run_id}")
        
        # Step 1: Get LLM Hints with fallback
        try:
            self.hints = self.advisor.get_hints(
                domain=self.config['domain'],
                hw=self.hw
            )
            print(f"LLM hints received: {[h['hint'] for h in self.hints]}")
        except Exception as e:
            print(f"LLMAdvisor failed ({e}), falling back to empty hints.")
            self.hints = []
            
        # Step 2: Build Search Space
        ss = SearchSpace(self.hw, self.hints)
        
        # Step 3: Trial Loop
        status = "completed"
        budget = self.config.get('trial_budget', 50)
        target_acc = self.config.get('target_accuracy', 1.0)
        
        for i in range(budget):
            self.trials_completed += 1
            trial_id = f"trial_{i+1:03d}"
            
            # a. Sample
            candidate_cfg = ss.sample()
            
            # b. Simulate
            sim_res = self.simulator.estimate(candidate_cfg)
            if not sim_res['feasibility_check_passed']:
                self._record_trial(i+1, sim_res, None, "infeasible")
                continue
                
            # c. Train
            try:
                model = Architecture(
                    candidate_cfg, 
                    num_classes=self.config['num_classes']
                )
                metrics = self.trainer.train(model, train_loader, val_loader)
                acc = metrics['val_accuracy']
                
                # d. Update best
                if acc > self.best_accuracy:
                    self.best_accuracy = acc
                    self.best_latency = sim_res['estimated_latency_ms']
                    self.best_size = sim_res['estimated_model_size_kb']
                    self.best_candidate_id = trial_id
                    self.best_candidate_cfg = candidate_cfg
                    
                    # Backup the absolute best checkpoint
                    import shutil
                    import os
                    if os.path.exists('/tmp/best_candidate.pth'):
                        shutil.copy('/tmp/best_candidate.pth', '/tmp/global_best.pth')
                
                # e. NNI report

                try:
                    nni.report_final_result(acc)
                except Exception:
                    pass
                
                self._record_trial(i+1, sim_res, metrics, "success")
                
                # f. Early stop
                if acc >= target_acc:
                    print(f"Target accuracy {target_acc} reached. Early stopping.")
                    status = "early_stopped"
                    break
                    
            except Exception as e:
                print(f"Trial {i+1} failed during training: {e}")
                continue

        # Restore the absolute best checkpoint back to /tmp/best_candidate.pth for the exporter
        import shutil
        import os
        if os.path.exists('/tmp/global_best.pth'):
            shutil.copy('/tmp/global_best.pth', '/tmp/best_candidate.pth')

        # Step 4: Return SearchRun dict
        return {
            "run_id": self.run_id,
            "status": status,
            "hardware_id": self.config['hardware_id'],
            "domain": self.config['domain'],
            "trial_budget": budget,
            "trials_completed": self.trials_completed,
            "best_accuracy": self.best_accuracy,
            "best_latency_ms": self.best_latency,
            "best_model_size_kb": self.best_size,
            "best_candidate_id": self.best_candidate_id,
            "best_candidate_cfg": self.best_candidate_cfg,
            "llm_hints_used": [h['hint'] for h in self.hints]
        }


    def _record_trial(self, trial_n, sim_res, train_res, status):
        """Prints a one-line summary after each trial."""
        acc = train_res['val_accuracy'] if train_res else 0.0
        size = sim_res['estimated_model_size_kb']
        print(f"Trial {trial_n:02d} | acc={acc:.3f} | size={size:.1f}KB | status={status}")
