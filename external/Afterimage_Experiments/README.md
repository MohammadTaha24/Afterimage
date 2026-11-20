# Experiment code from `D:\Afterimage_Experiments`

These files were run from `D:\Afterimage_Experiments\` on the machine that ran the paper experiments, outside this repository. They are copied here byte for byte so the team can rebuild the datasets and train Qwen. Copied on 20 November 2026.

| File | What it does |
| --- | --- |
| `01_preprocessing.ipynb` | Builds the 512-pixel APTOS datasets for both splits (original and clean), CLAHE on and off |
| `02_training.ipynb` | Ran the main Qwen grid on the clean split (ranks 8/16/32, CLAHE on/off, three seeds). Its saved outputs are the record of those runs. |
| `code/qwen_pilot.py` | Qwen training and scoring worker |
| `code/prepare_aptos.py` | Dataset builder behind `01_preprocessing.ipynb` and `finetuning/RetiZero/data/build_retizero_datasets.py` |
| `code/notebook_runner.py` | Launches workers from notebooks and keeps their console output |
| `code/recover_qwen.py` | Resumes or rescores a stopped Qwen run |

## How to use them

Copy this folder's contents to `D:\Afterimage_Experiments\`, keeping the layout (`code\` stays a subfolder). The notebooks and the scripts in `finetuning/` refer to them by that absolute path. The other prerequisites (models, data, the recovered split) are listed in the main README under "Reproducing the experiments: prerequisites".

## Do not edit

Saved runs record the SHA-256 of these scripts and check it. The copies here match those records:

| File | SHA-256 |
| --- | --- |
| `code/qwen_pilot.py` | `01ea6d6541cc9b7cec21f187393a9c3eb3621cc1a028b9f34d86e0f80cbd3222` |
| `code/prepare_aptos.py` | `2027b22c53d08f84b9a46267a1c74a32712878ac2e7669eb14d90fa344856bd0` |
| `code/notebook_runner.py` | `aafedf26a0bebc52fea954f03d41413559b91af44737328004ef714611f276cb` |
| `code/recover_qwen.py` | `f46846f3d91b1ccbb8142875f579c29fc36f0fe0272da95c813094086cdc2e59` |

If a change is needed, add a new file instead of editing one of these.
