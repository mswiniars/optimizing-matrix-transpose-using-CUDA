
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <ctime>
#include <cstdlib>
#include <chrono>

using namespace std;
using namespace std::chrono;


#define TILE_DIM 32
#define BLOCK_ROWS 8

__global__ void transposeNaive(int *vector, int *transposed, int size)
{
	int column = threadIdx.x + blockDim.x *blockIdx.x;
	int row = threadIdx.y + blockDim.x *blockIdx.y;

	if (row < size && column < size)
		transposed[row + column * size] = vector[column + row * size];
}

__global__ void transposeWithSharedMemory(int *vector, int *transposed, int size)
{
	__shared__ int tile[TILE_DIM][TILE_DIM];
	int column = threadIdx.x + TILE_DIM * blockIdx.x;
	int row = threadIdx.y + TILE_DIM * blockIdx.y;

	if (row < size && column < size)
		tile[threadIdx.x][threadIdx.y] = vector[column + size * row];

	__syncthreads();

	if (row < size && column < size)
		transposed[row + column * size] = tile[threadIdx.x][threadIdx.y];
}

__global__ void transposeNaiveNvidia(const int *idata, int *odata, int size)
{
	int column = blockIdx.x * TILE_DIM + threadIdx.x;
	int row = blockIdx.y * TILE_DIM + threadIdx.y;

	for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
		odata[column*size + (row + j)] = idata[(row + j)*size + column];
}

__global__ void transposeWithSharedMemoryNvidia(const int *idata, int *odata, int size)
{
	__shared__ float tile[TILE_DIM][TILE_DIM];

	int column = blockIdx.x * TILE_DIM + threadIdx.x;
	int row = blockIdx.y * TILE_DIM + threadIdx.y;

	for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
		tile[threadIdx.y + j][threadIdx.x] = idata[(row + j)*size + column];

	__syncthreads();

	column = blockIdx.y * TILE_DIM + threadIdx.x;
	row = blockIdx.x * TILE_DIM + threadIdx.y;

	for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
		odata[(row + j)*size + column] = tile[threadIdx.x][threadIdx.y + j];
}
__global__ void transposeWithNoBankConflicts(const int *idata, int *odata, int size)
{
	__shared__ float tile[TILE_DIM][TILE_DIM + 1];

	int column = blockIdx.x * TILE_DIM + threadIdx.x;
	int row = blockIdx.y * TILE_DIM + threadIdx.y;

	for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
		tile[threadIdx.y + j][threadIdx.x] = idata[(row + j)*size + column];

	__syncthreads();

	column = blockIdx.y * TILE_DIM + threadIdx.x;
	row = blockIdx.x * TILE_DIM + threadIdx.y;

	for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
		odata[(row + j)*size + column] = tile[threadIdx.x][threadIdx.y + j];
}

void transpositionCPU(int *vector, int *transposed, int size)
{
	for (int i = 0; i < size; i++)
		for (int j = 0; j < size; j++)
			transposed[i + j * size] = vector[j + i * size];
}

void show_matrix(int *vector, int size)
{
	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
			cout << vector[i + j * size];
		cout << endl;
	}
}
bool check_the_transposition(int *byCPU, int *byGPU, int size)
{
	for (int i = 0; i < size*size; i++)
		if (byCPU[i] != byGPU[i])
			return false;

	return true;
}

int main()
{
	int matrix_size;
	float naiveTime, sharedTime, naiveTimeNvidia, sharedTimeNvidia, noBankConflictsTime;
	cudaEvent_t naiveStart, naiveStop, sharedStart, sharedStop, naiveStartNvidia, naiveStopNvidia, sharedStartNvidia, sharedStopNvidia;
	cudaEvent_t noBankConflictsStart, noBankConflictsStop;
	while (true)
	{
		cout << "How big matrix do you want to transpose: ";
		cin >> matrix_size;

		int *dev_vector;
		int *dev_transposed;
		size_t memory_size = matrix_size * matrix_size * sizeof(int);

		//Dynamic allocation for CPU
		int *vector_of_matrix = new int[matrix_size*matrix_size];
		int *transposed_vector = new int[matrix_size*matrix_size];
		int *transposed_vectorCPU = new int[matrix_size*matrix_size];

		cudaSetDevice(0);
		cudaMalloc((int**)&dev_vector, memory_size);
		cudaMalloc((int**)&dev_transposed, memory_size);

		//Filling matrix
		srand(time(NULL));
		high_resolution_clock::time_point t3 = high_resolution_clock::now();
		for (int i = 0; i < matrix_size; i++)
			for (int j = 0; j < matrix_size; j++)
				vector_of_matrix[i + j * matrix_size] = rand() % 9 + 1;
		high_resolution_clock::time_point t4 = high_resolution_clock::now();
		auto duration2 = duration_cast<microseconds>(t4 - t3).count();
		cout << "Time needed to fullfill the matrix with random numbers: " << (double)duration2 / 1000 << "ms." << endl;
		cudaMemcpy(dev_vector, vector_of_matrix, memory_size, cudaMemcpyHostToDevice);

		//Calculate time for CPU
		high_resolution_clock::time_point t1 = high_resolution_clock::now();
		transpositionCPU(vector_of_matrix, transposed_vectorCPU, matrix_size);
		high_resolution_clock::time_point t2 = high_resolution_clock::now();

		auto duration = duration_cast<microseconds>(t2 - t1).count();
		double timeCPU = (double)duration / 1000;
		cout << "Time needed to transpose matrix by CPU: " << timeCPU << "ms." << endl;

		int threadsPerBlock = TILE_DIM;
		int blocksPerGrid = (matrix_size + threadsPerBlock - 1) / threadsPerBlock;
		int blocksPerGrid2 = matrix_size / TILE_DIM;
		// DIMENSIONS FOR KERNELS 
		dim3 gridSize = dim3(blocksPerGrid, blocksPerGrid, 1);
		dim3 gridSize2 = dim3(blocksPerGrid2, blocksPerGrid2, 1);
		dim3 blockSize = dim3(threadsPerBlock, threadsPerBlock, 1);
		dim3 blockSize2 = dim3(TILE_DIM, BLOCK_ROWS, 1);

		//CUDA TIMER
		cudaEventCreate(&naiveStart);
		cudaEventCreate(&naiveStop);
		cudaEventCreate(&sharedStart);
		cudaEventCreate(&sharedStop);
		cudaEventCreate(&naiveStartNvidia);
		cudaEventCreate(&naiveStopNvidia);
		cudaEventCreate(&sharedStartNvidia);
		cudaEventCreate(&sharedStopNvidia);
		cudaEventCreate(&noBankConflictsStart);
		cudaEventCreate(&noBankConflictsStop);

		//NAIVE TRANSPOSE KERNEL
		cudaEventRecord(naiveStart, 0);
		transposeNaive << < gridSize, blockSize >> > (dev_vector, dev_transposed, matrix_size);
		cudaEventRecord(naiveStop, 0);
		cudaEventSynchronize(naiveStop);
		cudaEventElapsedTime(&naiveTime, naiveStart, naiveStop);

		//SHARED MEMORY KERNEL
		cudaEventRecord(sharedStart, 0);
		transposeWithSharedMemory << < gridSize, blockSize >> > (dev_vector, dev_transposed, matrix_size);
		cudaEventRecord(sharedStop, 0);
		cudaEventSynchronize(sharedStop);
		cudaEventElapsedTime(&sharedTime, sharedStart, sharedStop);

		//NAIVE TRANSPOSE KERNEL NVIDIA
		cudaEventRecord(naiveStartNvidia, 0);
		transposeNaiveNvidia << < gridSize2, blockSize2 >> > (dev_vector, dev_transposed, matrix_size);
		cudaEventRecord(naiveStopNvidia, 0);
		cudaEventSynchronize(naiveStopNvidia);
		cudaEventElapsedTime(&naiveTimeNvidia, naiveStartNvidia, naiveStopNvidia);

		//SHARED MEMORY KERNEL NVIDIA
		cudaEventRecord(sharedStartNvidia, 0);
		transposeWithSharedMemoryNvidia << < gridSize2, blockSize2 >> > (dev_vector, dev_transposed, matrix_size);
		cudaEventRecord(sharedStopNvidia, 0);
		cudaEventSynchronize(sharedStopNvidia);
		cudaEventElapsedTime(&sharedTimeNvidia, sharedStartNvidia, sharedStopNvidia);

		//NO BANK CONFLICTS
		cudaEventRecord(noBankConflictsStart, 0);
		transposeWithNoBankConflicts << <gridSize2, blockSize2 >> > (dev_vector, dev_transposed, matrix_size);
		cudaEventRecord(noBankConflictsStop, 0);
		cudaEventSynchronize(noBankConflictsStop);
		cudaEventElapsedTime(&noBankConflictsTime, noBankConflictsStart, noBankConflictsStop);

		cudaMemcpy(transposed_vector, dev_transposed, memory_size, cudaMemcpyDeviceToHost);

		cout << "Time needed to transpose matrix by NAIVE KERNEL: " << naiveTime << "ms." << endl;
		cout << "Speedup by NAIVE KERNEL: " << timeCPU / (double)naiveTime << " times." << endl;

		cout << "Time needed to transpose matrix by kernel with SHARED MEMORY: " << sharedTime << "ms." << endl;
		cout << "Speedup by kernel with SHARED MEMORY: " << timeCPU / (double)sharedTime << " times." << endl;

		cout << "--------------------------------------------------------------------------" << endl;

		cout << "Time needed to transpose matrix by NAIVE KERNEL NVIDIA: " << naiveTimeNvidia << "ms." << endl;
		cout << "Speedup by NAIVE KERNEL NVIDIA: " << timeCPU / (double)naiveTimeNvidia << " times." << endl;
		if (check_the_transposition(transposed_vector, transposed_vectorCPU, matrix_size))
			cout << "Transposition by CPU == Transposition by GPU NAIVE" << endl << endl;
		else
			cout << "Transposition by CPU != Transpostion by GPU NAIVE" << endl << endl;

		cout << "Time needed to transpose matrix by kernel with SHARED MEMORY NVIDIA: " << sharedTimeNvidia << "ms." << endl;
		cout << "Speedup by kernel with SHARED MEMORY NVIDIA: " << timeCPU / (double)sharedTimeNvidia << " times." << endl;
		if (check_the_transposition(transposed_vector, transposed_vectorCPU, matrix_size))
			cout << "Transposition by CPU == Transposition by GPU SHARED" << endl << endl;
		else
			cout << "Transposition by CPU != Transpostion by GPU SHARED" << endl << endl;

		cout << "Time needed to transpose matrix by kernel with NO BANK CONFLICTS: " << noBankConflictsTime << "ms." << endl;
		cout << "Speedup by kernel with NO BANK CONFLICTS: " << timeCPU / (double)noBankConflictsTime << " times." << endl;

		if (check_the_transposition(transposed_vector, transposed_vectorCPU, matrix_size))
			cout << "Transposition by CPU == Transposition by GPU NO BANK CONFLICTS" << endl << endl;
		else
			cout << "Transposition by CPU != Transpostion by GPU NO BANK CONFLICTS" << endl << endl;

		char check;
		cout << "Would u like to check the transposition?(y/n): ";
		cin >> check;
		if (check == 'y')
		{
			cout << "Original matrix: " << endl;
			show_matrix(vector_of_matrix, matrix_size);
			cout << "Transposed matrix: " << endl;
			show_matrix(transposed_vector, matrix_size);
		}


		cudaEventDestroy(naiveStart);
		cudaEventDestroy(naiveStop);
		cudaEventDestroy(sharedStart);
		cudaEventDestroy(sharedStop);
		cudaEventDestroy(naiveStartNvidia);
		cudaEventDestroy(naiveStopNvidia);
		cudaEventDestroy(sharedStartNvidia);
		cudaEventDestroy(sharedStopNvidia);

		delete[] vector_of_matrix;
		delete[] transposed_vector;
		cudaFree(dev_vector);
		cudaFree(dev_transposed);
	}

	return 0;
}
