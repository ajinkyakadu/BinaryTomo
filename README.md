# BinaryTomo

Reconstruction of Binary Images from their tomographic projections

The least-squares formulation of a binary tomography problem is

![equation](/extras/primal.jpg)

where **A** is a tomography matrix of size *m* times *n*, **b** is the tomographic data of size *m* times *1*, and **x** is a binary image that has grey levels 0 and 1. This problem is NP-hard to solve. We propose to solve the following convex program instead:

![equation]()

This convex program is a Lagrangian dual of the main problem. The primal solution is retrieved from a dual solution using

![equation](http://www.sciweavers.org/tex2img.php?eq=%5Cmathbf%7Bx%7D%20%3D%20%5Cmathrm%7Bsign%7D%28%5Cmathbf%7Bp%7D%29&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0)



## Authors
* Ajinkya Kadu ([a.a.kadu@uu.nl](mailto:a.a.kadu@uu.nl))  
Mathematical Institute, Utrecht University, The Netherlands

## License
You can distribute the software as you wish.

## Dependencies
This framework has been tested on Matlab 2019a.


## Usage  
The examples scripts are  
1. **test_tomo** : classic discrete tomography problem with no regularization.
2. **test_tomo_cvx** : small-scale discrete tomography problem with dual problem solved using CVX toolbox.
3. **test_tomo_TV** : Total-variation regularized discrete tomography problem
4. **test_tomo_TVmin** : Minimum total-variation discrete solution to noisy tomography problem

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
