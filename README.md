# statistical-loss
This repository is an extension of Caffe for the paper "Statistical Loss and Analysis for Deep Learning in Hyperspectral Image Classification" (TNNLS)

## Installation
1. Install prerequisites for `Caffe` (see: [Caffe installation instructions](http://caffe.berkeleyvision.org/installation.html))

2. Add the header and source file in the Caffe project.

3. Modify
`#include <complex.h>`
`#define lapack_complex_float    float _Complex`
as
`#include<complex>`
`#define lapack_complex_float    std::complex<float>`
in the dependency package `NugetPackages/OpenBLAS.0.2.14.1/lib/native/include/lapacke.h`

4. Compile Caffe framework.

## Citing this work
If you find this work helpful for your research, please consider citing:

    @article{gong2020,
        Author = {Zhiqiang Gong and Ping Zhong and Weidong Hu},
        Title = {Statistical Loss and Analysis for Deep Learning in Hyperspectral Image Classification},
        Booktitle = {IEEE Transactions on Neural Networks and Learning Systems},
        Year = {2020}
    }
