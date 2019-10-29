#include"fileOperator.h"
#include<fstream>
#include<iostream>


void writeArrayIntoFile(char *fileName, FDATA_T *array, int arraySize) {
	std::ofstream outfile;
	outfile.open(fileName, std::ios::out);
	
	for (int i = 0; i < arraySize; i++) {
		outfile << array[i] << std::endl;
	}
	
	outfile.close();
}
