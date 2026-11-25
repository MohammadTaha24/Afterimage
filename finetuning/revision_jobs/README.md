# Revision GPU jobs: run guide

Everything left for the paper revision that needs a GPU is in one command. Your job is to run it. The code builds its inputs, checks them, runs a pilot, runs the queue, then rebuilds the analysis tables.

## What it runs

| Stage | What | GPU time |
|---|---|---|
| `messidor2` | Every model scored on the **verified Messidor-2 cohort**: all 1,744 gradable images with the adjudicated grades (Krause 2018), each preprocessed straight from the raw photo. Models: Qwen r16 at lr 2e-4 and 5e-4 (CLAHE on), Qwen r16 CLAHE-off at lr 5e-4 and at lr 2e-4 (the `final_02` adapters), Qwen zero-shot (on and off), RetiZero matched, RetiZero CLAHE and the saved app checkpoint. 21 jobs. | about 7 h |
| `timing` | Warm per-image timing, the greedy-generation check and GPU power for the 7 Qwen configurations with no timing record. Each job must reproduce its run's saved predictions. 7 jobs. | about 2 h |

Nothing is trained. The rank sweep at lr 2e-4 is already done: the lr 2e-4 training queue (`finetuning/Qwen3/paper_revision/orchestrator_02.py`) trained all 27 rank/alpha/CLAHE adapters into `final_02\train`.

New results go to `D:\Afterimage_Experiments\final_runs\final_02` and `D:\Afterimage_Experiments\retizero_runs\retizero_02`. Nothing in `runs\`, `final_01` or `retizero_01` is changed, and no file the lr 2e-4 training queue wrote in `final_02` is overwritten: this queue's pilot uses `final_02__revision_pilot_01`, and its energy file is `revision_energy_per_job.csv`.

## Before you start (one time)

1. Use the same PC, checkout and environment as the earlier runs:
   - the checkout at `D:\Afterimage_Reproduction`, next to `D:\Afterimage_Experiments`
   - `afterimage_env` (Python 3.11)

   The code finds the experiments folder from the checkout's location.
2. Get this branch into that checkout:

   ```bat
   cd /d D:\Afterimage_Reproduction
   git fetch origin
   git checkout revision-analysis-cpu
   ```

3. Have the Messidor-2 package ready. Any folder with this layout works:
   - `01_raw_images\` (1,748 images)
   - `02_adjudicated_grades\messidor_data.csv`
   - `03_eye_pairing\pairs.csv`

   On the owner's PC it is `D:\Afterimage_data_share\Messidor-2`. The `--messidor-package` examples below use that path.
4. Install scipy, which the analysis tables need:

   ```bat
   conda activate afterimage_env
   python -m pip install scipy
   ```

5. Close anything else using the GPU.

## Run

The notebook `revision_gpu_jobs.ipynb` in this folder runs the same steps, one per cell. It adds a preflight, a check of the Messidor-2 build, a final check that every job finished and no earlier result changed, and a zip of the results.

Or, from a terminal:

```bat
conda activate afterimage_env
cd /d D:\Afterimage_Reproduction
python -B finetuning\revision_jobs\revision_gpu_jobs.py all --messidor-package D:\Afterimage_data_share\Messidor-2
```

The command runs these steps in order and stops at the first one that fails:

1. **check**: lists every input as `ok` or `MISSING`, then the planned jobs. Fix anything `MISSING` and run again.
2. **build-messidor**: builds the cohort in `D:\Afterimage_Experiments\datasets\messidor2_verified` (about 5 minutes, CPU). It compares the rebuilt images with the archived 512 px export; on our PC all 1,057 shared images were pixel-identical.
3. **pilot**: a few images of each kind of job (about 15 minutes). It must end with `Pilot passed`.
4. **run**: the full queue. It keeps the PC awake, records a failed job and carries on, and logs GPU power every 5 s.
5. **summarize**: prints the status table, writes `final_02\revision_energy_per_job.csv`, then rebuilds the tables in `finetuning\paper_tables\output\revision_analysis\`.

**If it stops or the PC restarts, run the same command again.** Finished jobs are skipped; only unfinished ones run. Add `--skip-pilot` once the pilot has passed.

Steps you can run on their own:

```bat
python -B finetuning\revision_jobs\revision_gpu_jobs.py status
python -B finetuning\revision_jobs\revision_gpu_jobs.py check
python -B finetuning\revision_jobs\revision_gpu_jobs.py summarize --messidor-package D:\Afterimage_data_share\Messidor-2
```

`status` lists every job with its state and the time left. `check` checks the inputs only. `summarize` rebuilds the tables without the GPU.

## When it has finished

1. Check that `status` shows every job `done`. A `waiting` job names the file it is missing.
2. Commit the rebuilt tables and push. They hold only metrics and image IDs, no images:

   ```bat
   git add finetuning/paper_tables/output/revision_analysis
   git commit -m "Add verified Messidor-2 and timing results from the revision GPU queue"
   git push
   ```

3. Zip these folders and send them back. They hold predictions, logs and power records, but no model weights:
   - `D:\Afterimage_Experiments\final_runs\final_02\scores`
   - `D:\Afterimage_Experiments\retizero_runs\retizero_02\scores`
   - `final_02\revision_queue_state.json`, `final_02\revision_energy_per_job.csv` and `final_02\power_log_*.csv`

## If something goes wrong

| Message | What to do |
|---|---|
| `MISSING` in check | Do what the arrow says, then run again. |
| `Another job holds the GPU lock` | Another Qwen or RetiZero job is running. Wait for it to finish, or close its notebook. |
| `Pilot FAILED` | Send the console output and `D:\Afterimage_Experiments\final_runs\final_02__revision_pilot_01\`. |
| A job `failed:` in the queue summary | The queue carries on. Rerun the command to retry; send the job's `attempts\*\score.log` if it fails again. |
| CUDA out of memory | Close other GPU programs and rerun. The jobs are the same size as the earlier runs. |

## Code

- `revision_gpu_jobs.py`: the driver. Jobs are built by the existing queues (`finetuning/Qwen3/paper_revision/final_runs.py` and `finetuning/RetiZero/paper_revision/retizero_runs.py`) and run through their launchers. Each job therefore keeps its log, hardware specs and 5-second utilisation record, and takes the shared GPU lock. No hash-pinned file is changed.
- `build_messidor2_cohort.py`: the cohort builder. It reuses the APTOS builders' preprocessing (`prepare_aptos.py`), and records the hash of every input and output image.
- `test_revision_gpu_jobs.py`: CPU-only tests of the queue, the energy calculation and the builder. Run with `python -B finetuning\revision_jobs\test_revision_gpu_jobs.py`.
