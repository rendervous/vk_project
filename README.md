# vk

A minimal Vulkan Python package with a PyTorch-like import flow:

```python
import vk

device = vk.create_device(device_index=0, enable_validation_layers=False)
```

The native extension is `_vk` and is loaded by `vk/__init__.py`.

## Build and install (Windows PowerShell)

```powershell
Set-Location "D:\projects\vk_project"
python -m pip install -U pip
python -m pip install -e .
```

## Run example

```powershell
Set-Location "D:\projects\vk_project"
python .\examples\create_device.py
```

## Tutorials

Each tutorial can be opened directly in Google Colab, or built into a
local Jupyter notebook via `build_notebooks.bat` (requires `jupytext`).

| Tutorial | Open in Colab |
| --- | --- |
| 01 - Vulkan Devices | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rendervous/vk_project/blob/master/tutorials/tutorial01_devices.ipynb) |
| 02 - Vectors and Matrices | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rendervous/vk_project/blob/master/tutorials/tutorial02_vectors_and_matrices.ipynb) |
| 03 - Basic Compute | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rendervous/vk_project/blob/master/tutorials/tutorial03_basic_compute.ipynb) |

The notebooks themselves (`tutorials/*.ipynb`) must be committed/pushed to
the repository for the Colab links to resolve.

