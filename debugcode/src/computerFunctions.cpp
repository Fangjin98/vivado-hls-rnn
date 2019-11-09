#include"computerFunctions.h"

void vectorMultiplication(
	FDATA_T *partWeightMatrix,
	FDATA_T *inputVector,
	int dim,
	FDATA_T &output
) {
	FDATA_T tmp = 0;
	for (int i = 0; i < dim; i++) {
#pragma HLS PIPELINE
		tmp += (inputVector[i] * partWeightMatrix[i]);
	}
	output = tmp;
}
void matrixMultiplication(
	const FDATA_T *weightMatrix,
	FDATA_T * inputVector,
	int inputDim,
	int outputDim,
	FDATA_T * outputVector
) {
	FDATA_T weightMatrixReg[inputDim];

	for (int i = 0; i < outputDim; i++) {
		for (int j = 0; j < inputDim; j++) {
			weightMatrixReg[j] = weightMatrix[i*inputDim + j];
		}
#pragma HLS PIPELINE
		FDATA_T tmp;
		vectorMultiplication(
			weightMatrixReg,
			inputVector,
			inputDim,
			tmp
		);
		outputVector[i] = tmp;
	}
}