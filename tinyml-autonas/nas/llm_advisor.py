import os
import json
import anthropic

SYSTEM_PROMPT = """You are an expert in TinyML and embedded neural networks.
Given a task domain and chip constraints, return a JSON array of architecture hints.
Each hint: {"hint": "snake_case_key", "reason": "...", "priority": 1|2|3}
Priority 1 = critical constraint. Priority 2 = strong recommendation. 3 = suggestion.
ONLY return valid JSON. No markdown, no explanation outside the JSON."""

class LLMAdvisor:
    def __init__(self, model="claude-sonnet-4-5", provider="anthropic"):
        self.model = model
        self.provider = provider
        self.client = anthropic.Anthropic(
            api_key=os.environ["ANTHROPIC_API_KEY"]
        )

    def get_hints(self, domain, hw, task_desc="", max_hints=8) -> list[dict]:
        user_msg = self._build_prompt(domain, hw, task_desc, max_hints)
        
        message = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": user_msg}
            ],
        )
        
        raw = message.content[0].text.strip()
        
        # Strip markdown fences if Claude wraps JSON anyway
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        
        try:
            hints = json.loads(raw)
        except json.JSONDecodeError:
            # Retry with stricter prompt
            hints = self._retry_strict(domain, hw, task_desc, max_hints)
        
        print(f"[LLM] Got {len(hints)} hints from {self.model}.")
        print(f"[LLM] Input tokens: {message.usage.input_tokens} | Output: {message.usage.output_tokens}")
        return hints

    def _build_prompt(self, domain, hw, task_desc, max_hints):
        return f"""
Domain: {domain}
Chip: {hw.chip_id} | Flash: {hw.flash_kb}KB | SRAM: {hw.sram_kb}KB | {hw.mhz}MHz
FPU: {hw.has_fpu} | SIMD: {hw.supports_simd}
Task: {task_desc or domain}
Return exactly {max_hints} architecture hints as a JSON array.
"""

    def _retry_strict(self, domain, hw, task_desc, max_hints):
        message = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system="Return ONLY a JSON array. No text before or after. No markdown.",
            messages=[
                {"role": "user", "content": self._build_prompt(domain, hw, task_desc, max_hints)}
            ],
        )
        return json.loads(message.content[0].text.strip())
