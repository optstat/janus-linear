Set of linear solvers accepting dual tensors and able to be run in data parallel fashion.
Solvers are provided so that both dual and regular tensor version can be run.  All versions are designed to be run in a data parallel pattern utilizing any device that is designed for that purpose such as GPU, TPU or vector architectures.

Solvers with regular tensor data parallelism are given the extension Te and version with dual extensions are given the extension TeD
Solvers available currently are
   1.  LU decomposition
   2.  QR decomposition
   3.  GMRES

This set of solvers are dependent on the project janus-tensor-dual which contains a libtorch dependent version of dual numbers. 

Prerequisites to building this project:
I.   Libtorch together with any pre-requisites for running on your hardware (e.g. CUDA if you have a GPU card or TPU drivers if on the cloud.  Check with the pytorch/libtorch documentation for the hardware supported).
II.  Google test if you want to run the tests.
III. janus-tensor-dual is the library needed to run the dual version of the linear solvers and its installation is required.

TODO:
GMRES is not yet implemented in dual number form.  This will be implemented for high dimensional examples.
