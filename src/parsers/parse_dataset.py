### Ecosystem Imports ###
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
import pathlib
import shutil

### External Imports ###
import numpy as np
import torch as tc
import pandas as pd
import SimpleITK as sitk
from sklearn.model_selection import KFold

### Internal Imports ###
from paths import pc_paths as p


def parse_aortaseg():
    input_path = p.raw_data_path
    output_data_path = p.parsed_data_path / "Original"
    if not os.path.exists(output_data_path):
        os.makedirs(output_data_path)
    output_csv_path = p.csv_path / "dataset.csv"

    images_path = input_path / "images"
    gts_path = input_path / "masks"

    all_ids = list(range(1, 14)) + list(range(15, 37)) + list(range(38, 43)) + list(range(44, 54))
    print(f"Number of cases: {len(all_ids)}")
    ids = ["%03d" % (id,) for id in all_ids]

    dataframe = []
    for id in ids:
        image_path = images_path / f"subject{id}_CTA.mha"
        gt_path = gts_path / f"subject{id}_label.mha"
        print()
        print(f"Image path: {image_path}")
        print(f"GT Path: {gt_path}")

        image = sitk.ReadImage(image_path)
        gt_multiclass = sitk.ReadImage(gt_path)
        gt_binary = gt_multiclass > 0

        image_save_path = output_data_path / f"{id}_image.mha"
        gt_binary_save_path = output_data_path / f"{id}_gt_binary.nrrd"
        gt_multiclass_save_path = output_data_path / f"{id}_gt_multiclass.nrrd"

        sitk.WriteImage(image, str(image_save_path))
        sitk.WriteImage(gt_binary, str(gt_binary_save_path), useCompression=True)
        sitk.WriteImage(gt_multiclass, str(gt_multiclass_save_path), useCompression=True)

        to_append = (f"{id}_image.mha", f"{id}_gt_binary.nrrd", f"{id}_gt_multiclass.nrrd")
        dataframe.append(to_append)

    dataframe = pd.DataFrame(dataframe, columns=['Input Path', 'Ground-Truth Binary Path', 'Ground-Truth Multiclass Path'])
    dataframe.to_csv(output_csv_path, index=False)


def split_dataframe(num_folds=5, seed=1234):
    input_csv_path = p.csv_path / "dataset.csv"
    output_splits_path = p.csv_path
    if not os.path.isdir(os.path.dirname(output_splits_path)):
        os.makedirs(os.path.dirname(output_splits_path))
    dataframe = pd.read_csv(input_csv_path)
    print(f"Dataset size: {len(dataframe)}")
    kf = KFold(n_splits=num_folds, shuffle=True)
    folds = kf.split(dataframe)
    for fold in range(num_folds):
        train_index, test_index = next(folds)
        current_training_dataframe = dataframe.loc[train_index]
        current_validation_dataframe = dataframe.loc[test_index]
        print(f"Fold {fold + 1} Training dataset size: {len(current_training_dataframe)}")
        print(f"Fold {fold + 1} Validation dataset size: {len(current_validation_dataframe)}")
        training_csv_path = output_splits_path / f"training_fold_{fold+1}.csv"
        validation_csv_path = output_splits_path / f"val_fold_{fold+1}.csv"
        current_training_dataframe.to_csv(training_csv_path, index=False)
        current_validation_dataframe.to_csv(validation_csv_path, index=False)

def run():
    # parse_aortaseg()
    # split_dataframe(num_folds=5, seed=1234)
    pass

if __name__ == "__main__":
    run()