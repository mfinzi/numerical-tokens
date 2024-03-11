from finetune_together import finetune
import fire
import os
import together
import pandas as pd

if __name__ == "__main__":
    sizes = ["7b", "13b", "70b"]
    results = {}
    for size in sizes:
        model_name = f"togethercomputer/llama-2-{size}"
        tid = finetune(model_name, 'qm9', aug=True, epochs=0.1)
        results[size] = tid
    data = pd.DataFrame(results, index=[0])
    data.to_csv("model_scaling_results.csv")

    
