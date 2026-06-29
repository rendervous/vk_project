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

