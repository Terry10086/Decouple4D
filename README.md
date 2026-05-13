# Decouple4D: Static-Dynamic 3D Gaussian Decoupling from Monocular Video via Spatial-Temporal Consistency
The code for "Decouple4D: Static-Dynamic 3D Gaussian Decoupling from Monocular Video via Spatial-Temporal Consistency".

# Note
This is a preliminary draft. The code is currently not well organized and will be further updated.

# Environment Setup
We provide both pip and conda environment configurations for reproducibility.

```bash
conda env create -f environment.yml
conda activate sa4d
```
or

```bash
pip install -r requirements.txt
```

The environment file may contain some unnecessary packages. Alternatively, a simpler way is to install the dependencies of [4DGS](https://github.com/hustvl/4DGaussians), [GroupingGS](https://github.com/lkeab/gaussian-grouping), [DEVA](https://github.com/hkchengrex/Tracking-Anything-with-DEVA) (maybe not require), and [SAM](https://github.com/facebookresearch/segment-anything) separately. Note that the Python version should be no lower than 3.8. We use Python 3.8.20 in our implementation.


# Data
Take monocular video from [HyperNeRF](https://github.com/google/hypernerf) as an example. First, create a directory named data and place the dataset inside it. The expected directory structure is as follows:

```bash
./decouple4D/data/hypernerf/chickchicken/
├── images
└── sparse
```
Run 4DGS and SAM using the following command, or alternatively you may execute them in your own way.
```bash
# data preparation  
python train4dgs.py
python sam_all.py
```

# Training
```bash
# Please set the 'scene_name' hyperparameter in all the following file to your target scene
# SSGC
python train_separate.py --expname 'output_path'
# ISMR
python render_residual.py --model_path 'output_path'
# DMR  
python train_separate.py --object_masks
```

# Rendering
```bash
python render_separate.py --model_path 'output_path'
```

# Usage of other file
**Scene editing**: composite.py

**Render binary mask**: render_binary.py

**Select and track GS at each frame**: render_dynamic_mask.py




