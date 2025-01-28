# Find libtorch
find_package(Torch REQUIRED)
find_package(CUDAToolkit REQUIRED)





# Specify the executable
set(SOURCE_FILE linear_solver_test.cpp)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
add_executable(lineartest ${SOURCE_FILE})

set(SOURCE_FILE linear_cuda_solver_test.cu)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
add_executable(linearcudatest ${SOURCE_FILE})

# Include GTest
find_package(GTest REQUIRED)
include_directories(src/cpp ${GTEST_INCLUDE_DIRS})


# Include MPI
find_package(MPI REQUIRED)
include_directories(${MPI_CXX_INCLUDE_PATH})

# Add PETSc paths
set(PETSC_DIR /opt/petsc)
set(PETSC_INCLUDE_DIR ${PETSC_DIR}/include)
set(PETSC_LIB_DIR ${PETSC_DIR}/lib)

# Include PETSc headers
include_directories(${PETSC_INCLUDE_DIR})

# Link PETSc libraries
set(PETSC_LIBRARIES
    ${PETSC_LIB_DIR}/libpetsc.so
    ${MPI_CXX_LIBRARIES}  # MPI dependencies
    -lblas -llapack        # Add BLAS/LAPACK explicitly if not auto-detected
)

# Link GTest
target_link_libraries(lineartest ${GTEST_LIBRARIES} ${GTEST_MAIN_LIBRARIES} ${PETSC_LIBRARIES})
target_link_libraries(linearcudatest 
  PRIVATE 
      ${GTEST_LIBRARIES} 
      ${GTEST_MAIN_LIBRARIES} 
      "${TORCH_LIBRARIES}"
        # Link modern CMake CUDA targets:
        CUDA::cudart
        CUDA::cublas
        CUDA::cusolver
)

# Link libtorch
target_link_libraries(lineartest "${TORCH_LIBRARIES}")
set_property(TARGET lineartest PROPERTY CXX_STANDARD 17)
add_test(NAME LinearTest COMMAND ${CMAKE_BINARY_DIR}/bin/lineartest)

# Link cuda
set_property(TARGET linearcudatest PROPERTY CXX_STANDARD 17)
add_test(NAME LinearCudaTest COMMAND ${CMAKE_BINARY_DIR}/bin/linearcudatest)

# Ensure the test executable is built when running 'make test'
add_custom_target(run_tests
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target lineartest
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
)

# Ensure the test executable is built when running 'make test'
add_custom_target(run_cuda_tests
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target linearcudatest
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
)


# Add an alias for the test target to ensure it works with `make test`
add_custom_target(test
    DEPENDS run_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
    #Run the ./bin/dualtest executable
    COMMAND ${CMAKE_BINARY_DIR}/bin/lineartest
    #COMMAND ${CMAKE_BINARY_DIR}/bin/linearcudatest
)



