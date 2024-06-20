Set of linear solvers accepting dual tensors and able to be run in data parallel fashion.
Solvers are provided so that both dual and regular tensor version can be run.  All versions are designed to be run in a data parallel manner.
Solvers with regular tensor data parallelism are given the extension Te and version with dual extensions are given the extension TeD
Solvers available currently are
   1.  LU decomposition
   2.  QR decomposition
   3.  GMRES

This set of solvers are dependent on the project janus-dual which contains a libtorch dependent version of dual numbers. 
