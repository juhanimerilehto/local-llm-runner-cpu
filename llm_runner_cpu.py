import argparse
from pathlib import Path
from huggingface_hub import hf_hub_download
from ctransformers import AutoModelForCausalLM
import psutil
import os

class LLMRunner:
    def __init__(
        self,
        model_url: str,
        model_file: str = "model.gguf",
        model_type: str = "mistral",
        context_length: int = 4096,
        batch_size: int = 8,
        threads: int = 4,  # Default optimized for 6-core CPUs
        temperature: float = 0.7,
        max_new_tokens: int = 4096
    ):
        self.model_url = model_url
        self.model_file = model_file
        self.model_type = model_type
        self.context_length = context_length
        self.batch_size = batch_size
        self.threads = threads or min(psutil.cpu_count(logical=False) - 2, 4)  # Leave 2 cores for system
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.model = None
        
        self.system_prompt = """<|im_start|>system
You are a helpful AI assistant. You aim to be direct, truthful, and factual in your responses.
Always respond in a clear and coherent manner. If you don't know something, say so.
Avoid roleplaying or generating fictional scenarios.
<|im_end|>
"""

    def download_model(self) -> str:
        return hf_hub_download(
            repo_id=self.model_url,
            filename=self.model_file,
            resume_download=True
        )

    def load_model(self):
        model_path = self.download_model()
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            model_type=self.model_type,
            context_length=self.context_length,
            batch_size=self.batch_size,
            threads=self.threads
        )

    def generate(self, prompt: str) -> str:
        if not self.model:
            self.load_model()
            
        formatted_prompt = f"{self.system_prompt}<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        return self.model(
            formatted_prompt,
            temperature=self.temperature,
            max_new_tokens=self.max_new_tokens,
            threads=self.threads  # Ensure thread usage during generation
        )

def main():
    parser = argparse.ArgumentParser(description='Run local LLM models (CPU optimized)')
    parser.add_argument('--model_url', type=str, default="TheBloke/OpenHermes-2.5-Mistral-7B-GGUF",
                      help='HuggingFace model URL')
    parser.add_argument('--model_file', type=str, default="openhermes-2.5-mistral-7b.Q4_K_M.gguf",
                      help='Model filename')
    parser.add_argument('--model_type', type=str, default="mistral",
                      help='Model architecture type')
    parser.add_argument('--threads', type=int, default=None,
                      help='Number of CPU threads to use')
    parser.add_argument('--temperature', type=float, default=0.7,
                      help='Generation temperature')
    parser.add_argument('--batch_size', type=int, default=8,
                      help='Batch size for processing')
    
    args = parser.parse_args()
    
    # Set environment variable to force CPU usage
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    
    llm = LLMRunner(
        model_url=args.model_url,
        model_file=args.model_file,
        model_type=args.model_type,
        threads=args.threads,
        temperature=args.temperature,
        batch_size=args.batch_size
    )

    print(f"\nCPU-optimized model initialized using {llm.threads} threads")
    print("Enter your prompts (type 'quit' to exit):")
    print("Note: First run will download the model (~4.37GB) if not already present.")
    
    while True:
        try:
            prompt = input("\nPrompt> ")
            if prompt.lower() in ['quit', 'exit']:
                break
            response = llm.generate(prompt)
            print("\nResponse:", response.strip())
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()