# Evaluation

You can reproduce evaluation once you generated predictions for a given `cs`using  :

```bash
python3 scripts/evaluation.py \
							--base_dir /path/to/data/folder # Path to the datafolder (DataSplit contains relative paths)
							--data_file DAVIS_D16Split\ # Data file
							--model_dir SavedModels/YourModel # Model folder
                            -cs -1 # Cut size of the prediction
                            --select_mask optimax_all # Mask selection
							--save_vis # Save visualisation
```

Evaluations are done with predicted mask resized to the ground truth size.

<u>Select Mask :</u>

In order to evaluate motion segmentation on VOS datasets we need to select segments as foreground or background and thus a `select_mask` strategy as we detailed in the paper and supplementary materials.

- `linear_assignment`:  Select optimally (using ground-truth) one predicted mask as foreground mask ( used in the paper for 2 masks evaluation )
- `optimax_all`: Select optimally (using ground-truth) one / several predicted masks as foreground mask (used in the paper for 4 masks evaluation)
- `except_biggest`: Select the biggest predicted mask as the background, the rest as foreground



<u>Official Evaluations :</u>

1. Evaluation on **FBMS** ans **SegTrack-v2** are done using the script above.

2. Evaluation on **Davis 16** is done with the [official script](https://github.com/fperazzi/davis) on the binary masks generated using the script above.

3. For evaluation on  **Davis 17-motion** (multi masks) we first convert the segmentation to colormap using :

   ```bash
   python3 scripts/proba_to_color.py --data_file DAVIS17_D17Split\
   								  --base_dir /path/to/data/folder #Path to the datafolder (DataSplit contains relative paths)
                                     --duplicate_last\
                                     --model_dir SavedModels/YourModel
                                     -cs -1
   ```

   Then use the official code with the method and labels as described in [OCLR github](https://github.com/Jyxarthur/OCLR_model)

4. Evaluation on FBMS-Multimask (Diva) with bIoU :

**Step 1 :** 

```bash
python3 scripts/proba_to_color.py --data_file FBMSclean_FBMSSplit\
								  --base_dir /path/to/data/folder #Path to the datafolder (DataSplit contains relative paths)
                                  --duplicate_last\
                                  --model_dir SavedModels/YourModel
                                  -cs -1
```

**Step 2 :** 

Call steps in script ` scripts/BIoU.py`

