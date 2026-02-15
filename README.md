# Extending Splatography for VFX using Optix-Triangle-Tracer

The current tools for VFX on Gaussian Splatting are all 3D-only - No dynamic! (can you even believe it!)

This repo is a playground for developing some VFX/Geometry editing tools for dynamic Gaussian Splatting.

I am using the [ViVo dataset](https://vivo-bvicr.github.io/) for doing this on multi-view human-entertainment videos. You can modify `scene/dataset_readers.py` to implement you own custom data or use existing data-loading methods.



# Visualization/Editing

- Using Nvidia's Optix (for RTX machines; CUDA based): a Gaussian->Triangle algorithm + fast Ray-Triangle intersection
- View foreground/background segmented + modify various internal features


# Model/Code Information

I have re-implemented the [Splatography](https://azzarelli.github.io/splatographypage/index.html), a dynamic GS pipeline that my collegues and I designed for filmmaking application on 6DoF dynamic scenes in sparse camera settings. Please see the original repo if you're interested in how it works.

## Installation

1. Follow the installation instructions from [Splatography](https://azzarelli.github.io/splatographypage/index.html) to install the conda environment
2. Install/Build Optix 7 headers (I followed Optix 7 install instruction from [here](https://github.com/mortacious/python-optix))
3. Build the local `submodules/python-optix` lib

## Why Splatography and not 4D-GS/STG/...?

- It does targeted reconstruction based on a set of masks that are defined for timestamp t=0 (i.e. it biases training to better reconstruct the target foreground assets)
- Uses [MipSplatting](https://niujinshuchong.github.io/mip-splatting/) for antialiasing (a big problem in sparse-view capture scenarios)
- The foreground/background separation makes modifying the target regions a lot simpler

## Model Structure

### Primitive Classes
`GaussianModel` is the base class with save, loading, time-based deformation (`self.deform(time)`), etc. functionalities

`ForegroundGaussians` is the foreground primitives class

`BackgroundGaussians` is the background primitives class

### Runners
`gui.py` is the main file for running traniing/viewing and implements the `GUI` class for handling Scene, Gaussian, training initialization and running.

`gui_utils/base.py` is the base class for traniing/viewing and implements the [DearPyGui](https://github.com/hoffstadt/DearPyGui) rendering and editing functionality. 