from setuptools import setup, find_packages

setup(
    name="vk",                # Replace with your library name
    version="0.0.5",                       # Semantic versioning
    author="Ludwic Leonard",
    author_email="rendervous.project@gmail.com",
    description="Vulkan API facade for python.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/rendervous/vk_project",
    packages=find_packages(where="src"),  # Look for packages in "src"
    package_dir={"": "src"},  # Map the root package directory to "src"
    # packages=find_packages(),             # Automatically find packages in your project
    install_requires=[
        "numpy>=1.21.0",                  # List dependencies here
        "cffi",
        "torch",
        "torchvision",
        "cuda-python",
        "PyOpenGL",
        "glfw",
        "imgui"
    ],
    extras_requires={
        "gui": [
            "pywin32; platform_system == 'Windows'",
            "pygobject; platform_system == 'Linux'",
            "pyobjc; platform_system == 'Darwin'",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',              # Specify supported Python versions
)