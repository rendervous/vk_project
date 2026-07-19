"""Best-effort, zero-configuration Vulkan driver bootstrap for Debian/Ubuntu
-based environments (Google Colab in particular): if no Vulkan ICD is
registered yet and installing one doesn't require a password prompt, install
one automatically -- so `pip install vulky` alone is enough, no separate
`apt-get install libnvidia-gl-...` cell required. A no-op everywhere else
(non-Linux, no apt, already has a working ICD, or would need a password).
"""

import os
import re
import shutil
import subprocess
import sys


def _has_vulkan_icd() -> bool:
    icd_dir = "/usr/share/vulkan/icd.d"
    try:
        return os.path.isdir(icd_dir) and len(os.listdir(icd_dir)) > 0
    except OSError:
        return False


def _apt_runner():
    """The apt-get invocation prefix that needs no password, or None if
    installing anything would require an interactive password prompt."""
    if os.geteuid() == 0:
        return ["apt-get"]
    if shutil.which("sudo"):
        probe = subprocess.run(
            ["sudo", "-n", "true"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        if probe.returncode == 0:
            return ["sudo", "apt-get"]
    return None


def _nvidia_driver_version():
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return None
    try:
        output = subprocess.run(
            [nvidia_smi, "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return None
    match = re.match(r"(\d+)", output)
    return match.group(1) if match else None


def ensure_vulkan_driver() -> None:
    if sys.platform != "linux" or not shutil.which("apt-get") or _has_vulkan_icd():
        return

    apt = _apt_runner()
    if apt is None:
        return  # would need a password prompt -- leave it to the user

    # A GPU vendor's userspace Vulkan ICD must exactly match the kernel
    # driver already loaded on the machine (Colab preloads the kernel driver
    # into the VM, only the matching userspace package is missing); with no
    # detectable vendor, mesa-vulkan-drivers is a safe, freely redistributable
    # fallback covering AMD/Intel/software rendering.
    nvidia_version = _nvidia_driver_version()
    package = f"libnvidia-gl-{nvidia_version}" if nvidia_version else "mesa-vulkan-drivers"

    try:
        subprocess.run(apt + ["update", "-y"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(apt + ["install", "-y", package], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except (subprocess.SubprocessError, OSError):
        pass  # best-effort only; device creation will surface a clear error if this didn't work
