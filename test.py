from llama_cpp import Llama
from rich.console import Console
import datetime
from rich.prompt import Prompt
console = Console(width=110) #initialize RICH console
console.clear() #clear screen

# Set gpu_layers to the number of layers to offload to GPU. Set to 0 if no GPU a
llm = Llama(
model_path="model/starling-lm-7b-alpha.Q4_K_M.gguf", # Download the model fil
n_ctx=8192, # this is the context window of StarlingLM
#n_threads=2, # The number of CPU threads to use...
)


console.clear()  # clear screen
console.print("[bold italic bright_red]Model [bold bright_yellow]starling-lm-7b[/]")
user = Prompt.ask("[bold green1 reverse]User: ")
prompt = f"GPT4 User: {user} GPT4 Assistant: "


# Simple inference example
generated_text = ""
start = datetime.datetime.now()
output = llm(prompt, max_tokens=912, stop=["</s>"], # Example stop token - not
            echo=False, stream=True)

for chunk in output:
    console.print(f'[bold green1]{chunk["choices"][0]["text"]}', end="")
    generated_text += chunk["choices"][0]["text"]
delta = datetime.datetime.now() - start

for chunk in output:
    console.print(f'[bold green1]{chunk["choices"][0]["text"]}', end="")
    generated_text += chunk["choices"][0]["text"]
