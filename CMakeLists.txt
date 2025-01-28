cmake_minimum_required(VERSION 3.10)
project(JanusLinearProject LANGUAGES CXX)

# ------------------------------------------------------------------------------
# 1) Basic Project Settings
# ------------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS 1 CACHE INTERNAL "Suppress developer warnings")

# ------------------------------------------------------------------------------
# 2) Dependencies: Torch, CUDA, Threads (pthreads)
# ------------------------------------------------------------------------------
find_package(Torch REQUIRED)
find_package(CUDA REQUIRED)
find_package(Threads REQUIRED)

# ------------------------------------------------------------------------------
# 3) Headers
# ------------------------------------------------------------------------------
set(HEADER_SOURCE_DIR "${CMAKE_SOURCE_DIR}/src/cpp")         # Where your .hpp files live
set(HEADER_INSTALL_DIR "/usr/local/include/janus")           # Where to copy for system-wide install

# ------------------------------------------------------------------------------
# 4) cuTENSOR (optional)
# ------------------------------------------------------------------------------
find_path(CUTENSOR_INCLUDE_DIR cutensor.h
    HINTS /usr/include /usr/local/cuda/targets/x86_64-linux/include
)
find_library(CUTENSOR_LIBRARY libcutensor.so
    HINTS /usr/lib/x86_64-linux-gnu /usr/local/cuda/targets/x86_64-linux/lib
)
# You can link these in your tests/cpp if you need cuTENSOR.

# ------------------------------------------------------------------------------
# 5) We need ExternalProject for building PETSc twice
# ------------------------------------------------------------------------------
include(ExternalProject)

# ------------------------------------------------------------------------------
# (A) Local PETSc Build (no sudo needed). Installs into build/petsc_install.
# ------------------------------------------------------------------------------
set(PETSC_LOCAL_DIR "${CMAKE_BINARY_DIR}/petsc_install")

#Add Petsc include directory
include_directories(${PETSC_LOCAL_DIR}/include)
ExternalProject_Add(PetscLocalProject
    GIT_REPOSITORY "https://gitlab.com/petsc/petsc.git"
    GIT_TAG "main"

    CONFIGURE_COMMAND
      <SOURCE_DIR>/configure
        --prefix=/opt/petsc
        --with-cc=gcc                # <-- Use non-MPI compilers
        --with-cxx=g++               #     instead of mpicc/mpicxx
        --with-fc=0
        --with-cuda=1
        --with-cudac=/usr/local/cuda/bin/nvcc
        --with-mpi=0                 # <-- Explicitly disable MPI
        --with-debugging=0
        --download-f2cblaslapack=1
        --COPTFLAGS="-O3"
        --CXXOPTFLAGS="-O3"
    BUILD_COMMAND make all
    INSTALL_COMMAND sudo make install
    BUILD_IN_SOURCE 1
    EXCLUDE_FROM_ALL TRUE  # So it won't build unless explicitly requested
)



# ------------------------------------------------------------------------------
# (B) System-Wide PETSc Build (needs sudo), installs into /opt/petsc
# ------------------------------------------------------------------------------
ExternalProject_Add(PetscSystemProject
    GIT_REPOSITORY "https://gitlab.com/petsc/petsc.git"
    GIT_TAG "main"

    CONFIGURE_COMMAND
      <SOURCE_DIR>/configure
        --prefix=/opt/petsc
        --with-cc=gcc                # <-- Use non-MPI compilers
        --with-cxx=g++               #     instead of mpicc/mpicxx
        --with-fc=0
        --with-cuda=1
        --with-cudac=/usr/local/cuda/bin/nvcc
        --with-mpi=0                 # <-- Explicitly disable MPI
        --with-debugging=0
        --download-f2cblaslapack=1
        --COPTFLAGS="-O3"
        --CXXOPTFLAGS="-O3"
    BUILD_COMMAND make all
    INSTALL_COMMAND sudo make install
    BUILD_IN_SOURCE 1
    EXCLUDE_FROM_ALL TRUE  # So it won't build unless explicitly requested
)


# We'll trigger this system-wide build only during "sudo make install" 
# in our custom target below.

# ------------------------------------------------------------------------------
# 6) Override the default "install" target for system-wide PETSc + headers
# ------------------------------------------------------------------------------
# We'll define our own "install" target that depends on building PETScSystemProject
# and then copies the Janus headers to /usr/local/include/janus.
# This means `sudo make install` will do the system-wide PETSc plus header copy.

add_custom_target(install_petsc
    COMMENT "System-wide install of PETSc and Janus headers"
)

# Ensure our "install" target depends on building & installing the system PETSc:
add_dependencies(install_petsc PetscSystemProject)

# Next, we copy the Janus headers into /usr/local/include/janus.
# We'll do it with a custom command that also depends on "PetscSystemProject-install" 
# so that PETSc is installed first.

add_custom_command(
    TARGET install_petsc
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${HEADER_INSTALL_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${HEADER_SOURCE_DIR} ${HEADER_INSTALL_DIR}
    COMMENT "Copying Janus headers to ${HEADER_INSTALL_DIR}"
)



# 2) Optionally retrieve the install directory
ExternalProject_Get_Property(PetscSystemProject INSTALL_DIR)

# 3) Pass that info along (if needed in subdirectories)
set(PETSC_INSTALL_DIR "${INSTALL_DIR}" CACHE INTERNAL "Petsc install dir")
# This ensures when you do `make -j`, we build PETSc locally (PetscLocalProject).
# We can define a custom target to show a message:
add_custom_target(PetscLocalMessage ALL
    DEPENDS PetscLocalProject
    COMMAND ${CMAKE_COMMAND} -E echo "Local PETSc built/installed at: ${PETSC_LOCAL_DIR}"
)

# Now when you run `sudo make install`, it will:
#  1. Build PETSc system-wide (PetscSystemProject)
#  2. Copy Janus headers into /usr/local/include/janus

# ------------------------------------------------------------------------------
# 7) Add subdirectory for tests
# ------------------------------------------------------------------------------
add_subdirectory(tests/cpp)

# ------------------------------------------------------------------------------
# That's it!
# ------------------------------------------------------------------------------