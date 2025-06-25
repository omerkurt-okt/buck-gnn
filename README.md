# Graph Neural Networks as Surrogate Models for Structural Analysis: A Study on Buckling Behavior

This repository contains the implementation of a Graph Neural Network (GNN) framework for predicting buckling behavior of thin-walled structures with and without 1D stiffeners, as presented in the Master's thesis by Ömer Kurt at Middle East Technical University.

## Overview

This research develops a novel approach to structural analysis using Graph Neural Networks as surrogate models, specifically focusing on predicting buckling behavior of thin-walled structures. The framework addresses computational challenges in traditional finite element analysis by providing an efficient machine learning-based alternative that maintains engineering-grade accuracy while achieving computational speeds approximately two orders of magnitude faster than conventional methods.

## Key Features

- **Comprehensive Data Generation Pipeline**: Automated generation of diverse structural geometries using Bezier curves
- **Advanced Graph Representation**: Novel approach incorporating super nodes and virtual edges for enhanced information flow
- **Rotational/Translational Invariance**: PCA-based coordinate transformation ensuring generalization across different orientations
- **Dual Architecture Support**: Implementation of both GraphSAGE and CustomGNN architectures
- **Multi-scale Analysis**: Support for both non-stiffened and stiffened structures
- **Efficient Buckling Prediction**: Accurate prediction of critical buckling eigenvalues

## Main Contributions

1. **Data Generation Framework**
   - Bezier curve-based shape generation system creating diverse yet physically meaningful geometries
   - Systematic load case generation with comprehensive coverage of realistic loading scenarios
   - Advanced dataset balancing methodology addressing inherent biases in structural response distributions

2. **Architectural Innovations**
   - Novel coordinate transformation approach based on principal component analysis
   - Enhanced information flow mechanisms through virtual edges and super node architecture
   - Specialized pooling strategies optimized for global property prediction

3. **Practical Applications**
   - Demonstrated effectiveness in buckling behavior prediction
   - Significant computational efficiency gains (100× faster than traditional FEA)
   - Framework for rapid stiffener layout optimization

## Performance Results

### Non-Stiffened Structures
- Validation MAPE: 5.5%
- Test MAPE: 6.73%
- Processing speed: ~160 structures/second (GPU)

### Stiffened Structures
- Validation MAPE: 12.5%
- Test MAPE: 17.64%
- Maintains practical accuracy for preliminary design applications

## Technical Details

### Dataset Generation
- Shape dimensions: 700-1000mm
- Material: Aluminum alloy (E=76 GPa, ν=0.3)
- Analysis types: Linear static and linear buckling
- Dataset sizes: 40,000 (non-stiffened) and 80,000 (stiffened) cases

### Model Architecture
- Base architecture: GraphSAGE with 6 layers
- Hidden dimension: 512
- Enhanced features: Super node for global information aggregation
- Pooling: Mean pooling for buckling prediction

### Training Infrastructure
- GPUs: NVIDIA Tesla V100 and P100 (16GB VRAM)
- Batch size: 16 (memory-constrained)
- Training time: ~10 hours for base dataset

## Requirements

   - Python 3.8+
   - PyTorch 1.10+
   - PyTorch Geometric
   - NumPy, SciPy, scikit-learn
   - MSC Nastran (for FEA validation)

## Limitations and Future Work

   - Currently limited to 2D thin-walled structures
   - Linear analysis only (no post-buckling behavior)
   - Isotropic materials only
   - Future extensions could include:
      -3 D structures and complex assemblies
      - Non-linear analysis capabilities
      - Composite materials

## Citation

If you use this work in your research, please cite:
@mastersthesis{kurt2025graph,
  title={Graph Neural Networks as Surrogate Models for Structural Analysis: A Study on Buckling Behavior},
  author={Kurt, {\"O}mer},
  year={2025},
  school={Middle East Technical University (Turkey)}
}

The full thesis is available at: https://open.metu.edu.tr/handle/11511/113521

## Acknowledgments
This research was conducted at Middle East Technical University and Turkish Aerospace Industries under the supervision of Prof. Dr. Ulaş Yaman. Computational resources were provided by TRUBA (Turkish National e-Infrastructure).
