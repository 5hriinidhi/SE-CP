import os
import json
import anthropic
from nas.hardware_config import HardwareConfig

SYSTEM_PROMPT = """You are an expert in TinyML and embedded neural networks.
Given a task domain and chip constraints, return a JSON array of architecture hints.
Each hint: {"hint": "snake_case_key", "reason": "...", "priority": 1|2|3}
Priority 1 = critical constraint. Priority 2 = strong recommendation. 3 = suggestion.
ONLY return valid JSON. No markdown, no explanation outside the JSON."""

class LLMAdvisor:
    def __init__(self, model="claude-sonnet-4-5", provider="anthropic"):
        self.model = model
        self.provider = provider
        # API requires os.environ['ANTHROPIC_API_KEY'] (per user update in previous msg)
        self.client = anthropic.Anthropic(
            api_key=os.environ["ANTHROPIC_API_KEY"]
        )

    def get_hints(self, domain: str, hw: HardwareConfig, task_desc: str = "", max_hints: int = 8) -> list[dict]:
        user_msg = self._build_prompt(domain, hw, task_desc, max_hints)
        
        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                temperature=0.2, # for deterministic output per constraint
                system=SYSTEM_PROMPT,
                messages=[
                    {"role": "user", "content": user_msg}
                ],
            )
        except Exception as e:
            raise RuntimeError(f"API call failed: {str(e)}")
            
        raw = message.content[0].text.strip()
        
        # Strip markdown fences if Claude wraps JSON anyway
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json\n") or raw.startswith("json"):
                raw = raw.replace("json\n", "", 1).replace("json", "", 1)
        
        try:
            hints = json.loads(raw)
        except json.JSONDecodeError:
            # Retry with stricter prompt
            hints = self._retry_strict(domain, hw, task_desc, max_hints)
        
        print(f"[LLM] Got {len(hints)} hints from {self.model}.")
        print(f"[LLM] Input tokens: {message.usage.input_tokens} | Output: {message.usage.output_tokens}")
        
        # Verify schema exactly
        validated = []
        for hint in hints:
            if all(k in hint for k in ('hint', 'reason', 'priority')):
                try:
                    # Ensure priority is a valid integer (1, 2, or 3)
                    p = int(hint['priority'])
                    if p in (1, 2, 3):
                        hint['priority'] = p
                        validated.append(hint)
                except (ValueError, TypeError):
                    continue
        return validated

    def _build_prompt(self, domain, hw, task_desc, max_hints):
        return f"""
Domain: {domain}
Chip: {hw.chip_id} | Flash: {hw.flash_kb}KB | SRAM: {hw.sram_kb}KB | {hw.mhz}MHz
FPU: {hw.has_fpu} | SIMD: {hw.supports_simd}
Task: {task_desc or domain}
Return exactly {max_hints} architecture hints as a JSON array.
"""

    def _retry_strict(self, domain, hw, task_desc, max_hints):
        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                temperature=0.2,
                system="Return ONLY a JSON array. No text before or after. No markdown.",
                messages=[
                    {"role": "user", "content": self._build_prompt(domain, hw, task_desc, max_hints)}
                ],
            )
            raw = message.content[0].text.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json\n") or raw.startswith("json"):
                    raw = raw.replace("json\n", "", 1).replace("json", "", 1)
            return json.loads(raw)
        except Exception as e:
            raise RuntimeError(f"API call failed during retry: {str(e)}")
