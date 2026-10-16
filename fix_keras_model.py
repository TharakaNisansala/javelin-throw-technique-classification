"""
Fixes .keras model files that fail to load with:
TypeError: Unrecognized keyword arguments passed to Dense: {'quantization_config': None}

This happens when a model is saved with a Keras version that added a
`quantization_config` field to layer configs, but the version used to load
it doesn't accept that field. This script strips out all `quantization_config`
keys recursively from the model's config.json and repackages the .keras file.

Usage:
    python fix_keras_model.py "path\\to\\your_model.keras"

It will create a new file: your_model_fixed.keras in the same folder.
"""

import sys
import os
import json
import zipfile
import shutil


def strip_quantization_config(obj):
    """Recursively remove 'quantization_config' keys from nested dicts/lists."""
    if isinstance(obj, dict):
        obj.pop("quantization_config", None)
        for key, value in obj.items():
            obj[key] = strip_quantization_config(value)
        return obj
    elif isinstance(obj, list):
        return [strip_quantization_config(item) for item in obj]
    else:
        return obj


def fix_keras_file(input_path):
    if not os.path.isfile(input_path):
        print(f"File not found: {input_path}")
        sys.exit(1)

    base, ext = os.path.splitext(input_path)
    output_path = f"{base}_fixed{ext}"
    tmp_dir = base + "_tmp_extract"

    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir)

    print(f"Extracting {input_path} ...")
    with zipfile.ZipFile(input_path, "r") as zf:
        zf.extractall(tmp_dir)
        namelist = zf.namelist()

    config_path = os.path.join(tmp_dir, "config.json")
    if not os.path.isfile(config_path):
        print("config.json not found inside the .keras archive. Aborting.")
        shutil.rmtree(tmp_dir)
        sys.exit(1)

    print("Cleaning quantization_config keys from config.json ...")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    config = strip_quantization_config(config)

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f)

    print(f"Repackaging into {output_path} ...")
    if os.path.exists(output_path):
        os.remove(output_path)

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in namelist:
            full_path = os.path.join(tmp_dir, name)
            zf.write(full_path, arcname=name)

    shutil.rmtree(tmp_dir)
    print("Done.")
    print(f"Fixed model saved as: {output_path}")
    print("Now load it with: keras.models.load_model(r'%s', compile=False)" % output_path)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python fix_keras_model.py <path_to_model.keras>")
        sys.exit(1)
    fix_keras_file(sys.argv[1])
