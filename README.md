# optimizing-matrix-transpose-using-CUDA

Optimizing huge matrix transpose using CUDA techniques. To do this I use one of algortihms proposed by NVIDIA Corporation. 
For comparsion there are used different approaches: naive transpose, transposition with shared memory and transposition withot conflicts in
banks. Transposition with shared memory and without conflicts in banks manage the problem in more efficient way, it takes less time to
transpose the matrix.
CPU and GPU architectures are compared. For huge matrix like 10k x 10k elements GPU seems to be for a few thousands faster.

To run the project you need to have CUDA installed and use Visual Studio 2015 compiler(v140) if you want to run this in Visual Studio.
