# %%
try:  # install all dependencies in colab
    import google.colab
    import subprocess
    subprocess.run(["sudo", "apt-get", "update", "-y"], check=True)
    subprocess.run(["sudo", "apt-get", "install", "-y", "libnvidia-gl-555", "vulkan-tools",
                     "glslang-tools", "vulkan-validationlayers-dev"], check=True)
    subprocess.run(["pip", "install", "vulky"], check=True)
except ImportError:
    print("Executing locally")

# %% [markdown]
# # Tutorial 01 - Vulkan Devices Tutorial
# This is the first tutorial on vk. It covers the one concept every other
# tutorial builds on: how a *device* (a logical handle to a physical GPU)
# is created, activated, switched, and released. Everything else in the
# library -- buffers, images, pipelines, commands -- is created against
# whichever device is currently active, so understanding this lifecycle
# first makes the rest of the API predictable.
# %%

import vk

# %% [markdown]
# Vulkan manages gpu execution and resources allocation through
# devices. In vulky, only one device per physical unit is created
# and activated if necessary. By default, if no device is activated
# it will be assumed using device(0).
#
# There is no explicit "init" call: the very first resource-creating
# function you call (``vk.buffer``, ``vk.image``, ...) transparently
# creates device 0 and makes it the active one. This is what "implicit
# context" means throughout vulky's API.
# %%

# creating a buffer of 100 floats with default device 0.
b0 = vk.buffer(100, element_type=vk.Type.FLOAT32)
print("Should be 0: ", b0.device_index)  # prints 0

# %% [markdown]
# If several GPUs are present, we can create concurrently
# to devices for each GPU but everytime only one device is
# active.
# With vk.device_infos() user can retrieve a list with all physical
# devices available for vulkan. Each device can be activated with vk.device(index).
#
# Activating a device is cheap once it has been created: the expensive
# part (instance/queue/allocator setup) only happens the first time a
# given index is activated. After that, switching back and forth is
# just flipping which device subsequent calls target.
# %%

devices = vk.device_infos()
print(devices)

if (len(devices) > 1):
    vk.device(1) # device 1 is created and set as active
    b1 = vk.buffer(100, element_type=vk.Type.FLOAT32)
    print("Should be 1: ", b1.device_index)  # prints 1
    b0.load(b1)  # copying buffer memory between devices
    del b1
else:
    print("Could not test multi-gpu block.")

# %% [markdown]
# Copying between devices is always possible. If devices belongs
# to physical units that are not connected, the copy will be done
# passing through the host (CPU) memory.
#
# This makes cross-device code portable: the same ``buffer.load(other)``
# call works whether the two devices are two GPUs, one GPU and a
# software renderer, or the same GPU with two logical devices -- vulky
# picks the fastest available path (peer-to-peer copy, or a staged
# host round-trip) transparently.
# %%

vk.device(0)  # activates back the device 0
print("Should be 0: ", vk.device_index())
# ...

# %% [markdown]
# For temporarily set a device and restore the previous active device
# user can use the device within a python context.
#
# This is the recommended pattern whenever a helper function needs to
# do a little work on a specific device without permanently changing
# what the caller sees as "the active device" -- much like temporarily
# ``chdir``-ing and restoring the previous working directory.
# %%

if (len(devices) > 1):
    with vk.device(1):
        # ...  Here the device active is 1
        print("Should be 1: ", vk.device_index())
else:
    print("Could not test multi-gpu block.")

# ... Here the device active is back 0
print("Should be 0: ", vk.device_index())

# %% [markdown]
# In this point there are two devices alive in this application.
# Even if there is no resource or command referencing to them.
# Use vk.relax() to remove all devices that are not referenced by any resource or command.
#
# ``vk.relax()`` is a good habit after a block of code that only ever
# needed a device transiently (e.g. a benchmarking loop over every
# available GPU): it keeps memory and driver resources bounded without
# requiring you to track every device index you touched.
# %%

del b0
vk.relax()  # releases all devices and destroy them if there is not other reference to them.

vk.device(0)  # creates again a device with the cost of all initialization.
print("Should be 0: ", vk.device_index())

# %% [markdown]
# Although it should not be necessary, explicitly destroying all device can be done with
# vk.dispose(). Notice, not further use of resources created from this device can be made.
#
# ``vk.dispose()`` is the "hard reset": unlike ``vk.relax()`` it tears
# down every device unconditionally, regardless of whether resources
# still reference them. Use it at the very end of a script or notebook
# session, not as a routine cleanup step mid-program.
# %%

vk.dispose()  # destroys all existing resources associated to current device and disables the usage of any pending resource.
