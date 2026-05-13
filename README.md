# Decouple4D: Static-Dynamic 3D Gaussian Decoupling from Monocular Video via Spatial-Temporal Consistency
The code for "Decouple4D: Static-Dynamic 3D Gaussian Decoupling from Monocular Video via Spatial-Temporal Consistency".

# Abstract 
Instance-level static-dynamic segmentation is a crucial task for comprehensive 3D scene understanding.  Existing methods typically identify dynamic regions by detecting violations of view consistency; however, they often struggle with periodic, subtle, or partially moving objects that maintain view-consistent appearances across most frames. 
In contrast, we explicitly model object motion and classify each Gaussian primitive according to its motion pattern and spatial position, using a novel spatial-temporal consistency constraint. The motion and positional information capture the dynamic behavior of individual Gaussians over time and the spatial relationships among Gaussians, respectively, while the spatial-temporal consistency constraint ensures reliable classification by enforcing that each Gaussian’s label remains consistent across multi-view projections and over time. For partially moving objects, we further associate dynamically significant Gaussians with instance segmentation masks, organizing dynamic Gaussians at the instance level rather than the pixel level. These masks, in turn, improve the training of both classifier and scene reconstruction, forming a reciprocal refinement loop. Experiments show that Decouple4D achieves state-of-the-art static-dynamic decoupling, and the explicit modeling of static and dynamic components enables applications such as distractor-free scene reconstruction and scene editing.

# Install
**Simple version:** The installation procedure follows the same implementation as SuperNormal. Please refer to the link: [SuperNormal](https://github.com/CyberAgentAILab/SuperNormal)

**Complex version:**

```bash
conda create -n sn python=3.8
conda activate sn
pip install -r requirements.txt
```
Both work, but the requirements.txt file contains some unnecessary packages.

# Data
Take monocular video from [HyperNeRF]() for example.


# Training
```bash
python exp_runner.py --case $OBJ_NAME --conf $CONF_NAME
# For example
python exp_runner.py --case angle --conf ./confs/exp1.conf
```

# Usage of other file
Scene editing: composite.py
Render binary mask: render_binary.py
Select and track GS at each frame: render_dynamic_mask.py


