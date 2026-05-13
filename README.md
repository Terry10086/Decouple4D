# Decouple4D: Static-Dynamic 3D Gaussian Decoupling from Monocular Video via Spatial-Temporal Consistency
The code for "Decouple4D: Static-Dynamic 3D Gaussian Decoupling from Monocular Video via Spatial-Temporal Consistency".

# Abstract 
Instance-level static-dynamic segmentation is a crucial task for comprehensive 3D scene understanding.  Existing methods typically identify dynamic regions by detecting violations of view consistency; however, they often struggle with periodic, subtle, or partially moving objects that maintain view-consistent appearances across most frames. 
In contrast, we explicitly model object motion and classify each Gaussian primitive according to its motion pattern and spatial position, using a novel spatial-temporal consistency constraint. The motion and positional information capture the dynamic behavior of individual Gaussians over time and the spatial relationships among Gaussians, respectively, while the spatial-temporal consistency constraint ensures reliable classification by enforcing that each Gaussian’s label remains consistent across multi-view projections and over time. For partially moving objects, we further associate dynamically significant Gaussians with instance segmentation masks, organizing dynamic Gaussians at the instance level rather than the pixel level. These masks, in turn, improve the training of both classifier and scene reconstruction, forming a reciprocal refinement loop. Experiments show that Decouple4D achieves state-of-the-art static-dynamic decoupling, and the explicit modeling of static and dynamic components enables applications such as distractor-free scene reconstruction and scene editing.

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
Scene editing: composite.py

Render binary mask: render_binary.py

Select and track GS at each frame: render_dynamic_mask.py




