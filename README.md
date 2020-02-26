# BinaryTomo

<img src="/extras/BT.gif" width="200">

This MATLAB toolbox solves the reconstruction of binary images from their tomographic projections. The challenge with this reconstruction problem is that the number of tomographic projections are much smaller than the size of the image. By exploiting the binary nature of the image, it is possible to solve the problem. This framework is based on the convex programming approach and can scale up fairly easily for large-scale tomographic problems.

## Problem description  
The least-squares formulation of a binary tomography problem is

![equation](/extras/primal.jpg)

where **A** is a tomography matrix of size *m* times *n*, **b** is the tomographic data of size *m* times *1*, and **x** is a binary image that has grey levels 0 and 1. This problem is NP-hard to solve. We propose to solve the following convex program instead:

![equation](/extras/dual.jpg)

This convex program is a Lagrangian dual of the main problem. The binary image is retrieved from a dual solution using

![equation](/extras/relation.jpg)



## Authors
* Ajinkya Kadu ([a.a.kadu@uu.nl](mailto:a.a.kadu@uu.nl))  
* Tristan van Leeuwen  
Mathematical Institute, Utrecht University, The Netherlands

## License
You can distribute the software as you wish.

## Dependencies
This framework has been tested on Matlab 2019a.


## Usage  
The examples scripts are  
1. **test_tomo** : classic discrete tomography problem with no regularization.
2. **test_tomo_gen** : discrete tomography problem with option for grey values to be other than -1 and 1.
3. **test_tomo_cvx** : small-scale discrete tomography problem with dual problem solved using CVX toolbox.
4. **test_tomo_TV** : Total-variation regularized discrete tomography problem
5. **test_tomo_TVmin** : Minimum total-variation discrete solution to noisy tomography problem

![image](/results/rat.png)

## Citation  
If you use this code, please use the following citation
```
@ARTICLE{8637779,
author={A. {Kadu} and T. {van Leeuwen}},
journal={IEEE Transactions on Computational Imaging},
title={A Convex Formulation for Binary Tomography},
year={2020},
volume={6},
number={},
pages={1-11},
keywords={Tomography;Mathematical model;Optimization;Iterative methods;Noise measurement;Phantoms;Binary tomography;inverse problems;duality;LASSO},
doi={10.1109/TCI.2019.2898333},
ISSN={2573-0436},
month={},
}
```
A preprint of the article can be found [here](https://arxiv.org/abs/1807.09196)

## Reporting Bugs
In case you experience any problems, please contact [Ajinkya Kadu](mailto:a.a.kadu@uu.nl)
