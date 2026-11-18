from pathlib import Path
import json
import pandas as pd

location = Path(__file__).resolve().parent

output = location/"results"/"imdb"

row = []
for i in sorted(output.glob("**/accuracies.json")):
    run, tag, seed_folder = i.relative_to(output).parts[:3]
    accuracy_fold = list(json.loads(i.read_text()).values())
    row.append(
        {
            "run": run,
            "seed": int(seed_folder.split("-")[1]),
            "mean": sum(accuracy_fold) / len(accuracy_fold),
            "min": min(accuracy_fold),
            "max": max(accuracy_fold),
        }
    )

df = pd.DataFrame(row)
print(df.round(4).to_string(index=False))
print(df.groupby(["run"])["mean"].agg(["mean", "std", "count"]))