#ifndef COMPUTER_FUNCTIONS
#define COMPUTER_FUNCTIONS
#include"types.h"

void vectorMultiplication(
	FDATA_T *partWeightMatrix,
	FDATA_T *inputVector,
	int dim,
	FDATA_T &output
);

void matrixMultiplication(
	const FDATA_T *weightMatrix,
	FDATA_T *inputVector,
	int inputDim,
	int outputDim,
	FDATA_T *outputVector
);

#endif // !COMPUTER_FUNCTIONS
