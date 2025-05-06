# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.0
#   kernelspec:
#     display_name: Python [conda env:base] *
#     language: python
#     name: conda-base-py
# ---

# %%
from openai import OpenAI
import time

class LegalNemotronAnswerer:
    def __init__(self, api_key, model="nvidia/llama-3.1-nemotron-70b-instruct"):
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://integrate.api.nvidia.com/v1"
        )
        self.model = model

    def generate(self, prompt, debug=False, retries=3, wait_time=10):
        """..."""
        messages = [{"role": "user", "content": prompt}]
        
        for attempt in range(retries):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.3,
                    top_p=1.0,
                    max_tokens=1024,
                    stream=False
                )
                response_text = completion.choices[0].message.content.strip()

                if debug:
                    print("🧠 Raw Response:\n", response_text)

                # Parse output
                answer = ""
                reasoning = ""
                if "Answer:" in response_text:
                    answer_part = response_text.split("Answer:", 1)[-1]
                    if "Reasoning:" in answer_part:
                        answer, reasoning = answer_part.split("Reasoning:", 1)
                    else:
                        answer = answer_part
                else:
                    answer = response_text

                return {
                    "answer": answer.strip(),
                    "reasoning": reasoning.strip() if reasoning else "(No explicit reasoning found)"
                }

            except Exception as e:
                print(f"⚠️ Attempt {attempt+1} failed: {e}")
                if attempt < retries - 1:
                    print(f"⏳ Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise RuntimeError("❌ All retries failed for NVIDIA LLM")


