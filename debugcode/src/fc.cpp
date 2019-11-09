#include "fc.h"
#include"constants.h"
#include"activations.h"


void fc_64(
	FDATA_T input_feature_map[FC_INPUT_SIZE1],
	FDATA_T output_feature_map[FC_OUTPUT_SIZE1]) {

	FDATA_T input_feature_map_reg[FC_INPUT_SIZE1];
	FDATA_T output_feature_map_reg=0;
	FDATA_T kernel_reg[FC_INPUT_SIZE1];

#pragma HLS ARRAY_PARTITION variable=input_feature_map_reg dim=1 cyclic factor=32

#pragma HLS ARRAY_PARTITION variable=kernel_reg dim=1 cyclic factor=32

fc_64_load_data_to_reg:
	for (LDATA_T i = 0; i < FC_INPUT_SIZE1; i++)
	{
#pragma HLS PIPELINE
		input_feature_map_reg[i] =	input_feature_map[i];
	}

	for (LDATA_T i = 0;i< FC_OUTPUT_SIZE1;i++) {
#pragma HLS DATAFLOW
		//load kernel
		LDATA_T kernel_offset = i * FC_INPUT_SIZE1;
		for (int j = 0; j < FC_INPUT_SIZE1; j++) {
#pragma HLS PIPELINE
			kernel_reg[j] = fc_kernel_1[kernel_offset+j];
		}
fc_64_compute_output:
		output_feature_map_reg=0;
		for (int j = 0; j < FC_INPUT_SIZE1; j++) {
			output_feature_map_reg += (kernel_reg[j] * input_feature_map_reg[j]);
		}
		output_feature_map[i] =output_feature_map_reg+fc_bias_1[i];
	}
}


void fc_16(
	FDATA_T input_feature_map[FC_INPUT_SIZE2],
	FDATA_T &output_feature_map) {

#pragma HLS INTERFACE ap_none port=input_feature_map

	FDATA_T input_feature_map_reg[FC_INPUT_SIZE2];
	FDATA_T output_feature_map_reg=0;
	FDATA_T kernel_reg[FC_INPUT_SIZE2];

	#pragma HLS ARRAY_PARTITION variable=input_feature_map_reg dim=1 cyclic factor=8

	#pragma HLS ARRAY_PARTITION variable=kernel_reg dim=1 cyclic factor=8

		//load input feature map
	
fc_16_load_data_to_reg:
for (int i = 0; i < FC_INPUT_SIZE2; i++)
	{
#pragma HLS PIPELINE
#pragma HLS UNROLL
		input_feature_map_reg[i] = input_feature_map[i];
		kernel_reg[i] = fc_kernel_2[i];
	}
	//compute
fc_16_compute_output:
	for (int j = 0; j <FC_INPUT_SIZE2; j++) {
#pragma HLS PIPELINE
		output_feature_map_reg += (kernel_reg[j] * input_feature_map_reg[j]);
	}

	output_feature_map = m_relu(output_feature_map_reg+fc_bias_2);
}
