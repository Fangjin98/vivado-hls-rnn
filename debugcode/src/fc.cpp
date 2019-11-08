#include "fc.h"
#include"constants.h"
//#pragma SDS data zero_copy(fc_kernel[0: FC_OUTPUT_SIZE * FC_INPUT_SIZE])
//#pragma SDS data zero_copy(fc_bias[0: FC_OUTPUT_SIZE])
//
//#pragma SDS data zero_copy( \
//    input_feature_map[0: BATCH_SIZE * RNN_STATE_SIZE])
//#pragma SDS data zero_copy(output_feature_map[0: BATCH_SIZE * FC_OUTPUT_SIZE])

void fc_64(
	FDATA_T input_feature_map[FC_INPUT_SIZE1],
	FDATA_T output_feature_map[FC_OUTPUT_SIZE1]) {

	FDATA_T input_feature_map_reg[FC_INPUT_SIZE1];
	FDATA_T output_feature_map_reg;
	FDATA_T kernel_reg[FC_INPUT_SIZE1];

//#pragma HLS ARRAY_PARTITION variable=input_feature_map_reg \
//    dim=2 cyclic factor=32
//#pragma HLS ARRAY_PARTITION variable=input_feature_map_reg \
//    dim=1 cyclic factor=2
//#pragma HLS ARRAY_PARTITION variable=output_feature_map_reg \
//    dim=1 cyclic factor=32
//#pragma HLS ARRAY_PARTITION variable=kernel_reg dim=1 cyclic factor=32

	//load input feature map
	for (LDATA_T i = 0; i < FC_INPUT_SIZE1; i++)
	{
#pragma HLS PIPELINE
		input_feature_map_reg[i] =
			input_feature_map[i];
	}

EACH_OUT_FM:
	for (LDATA_T output_feature_map_index = 0;
		output_feature_map_index < FC_OUTPUT_SIZE1;
		output_feature_map_index++) {
#pragma HLS DATAFLOW

		//load kernel
		LDATA_T kernel_offset = output_feature_map_index * FC_INPUT_SIZE1;
		for (int i = 0; i < FC_INPUT_SIZE1; i++) {
#pragma HLS PIPELINE
			kernel_reg[i] = fc_kernel_1[kernel_offset+i];
		}

		//compute
		FDATA_T tmp = 0;
		for (int j = 0; j < FC_INPUT_SIZE1; j++) {
			tmp += (kernel_reg[j] * input_feature_map_reg[j]);
		}
		output_feature_map_reg = tmp;

		output_feature_map_reg += fc_bias_1[output_feature_map_index];

		//relu
		/*output_feature_map[output_feature_map_index] =
			(output_feature_map_reg > 0) ? output_feature_map_reg : 0;*/

		output_feature_map[output_feature_map_index] =output_feature_map_reg;
	}
}


void fc_16(
	FDATA_T input_feature_map[FC_INPUT_SIZE2],
	FDATA_T &output_feature_map) {

	FDATA_T input_feature_map_reg[FC_INPUT_SIZE2];
	FDATA_T output_feature_map_reg;
	FDATA_T kernel_reg[FC_INPUT_SIZE2];

	//#pragma HLS ARRAY_PARTITION variable=input_feature_map_reg \
	//    dim=2 cyclic factor=32
	//#pragma HLS ARRAY_PARTITION variable=input_feature_map_reg \
	//    dim=1 cyclic factor=2
	//#pragma HLS ARRAY_PARTITION variable=output_feature_map_reg \
	//    dim=1 cyclic factor=32
	//#pragma HLS ARRAY_PARTITION variable=kernel_reg dim=1 cyclic factor=32

		//load input feature map
	
	for (int i = 0; i < FC_INPUT_SIZE2; i++)
	{
#pragma HLS PIPELINE
		input_feature_map_reg[i] = input_feature_map[i];
	}

EACH_OUT_FM:
#pragma HLS DATAFLOW

		//load kernel
	for (int i = 0; i < FC_INPUT_SIZE2; i++) {
#pragma HLS PIPELINE
		kernel_reg[i] = fc_kernel_2[i];
	}

	//compute
	FDATA_T tmp = 0;
	for (int j = 0; j <FC_INPUT_SIZE2; j++) {
		tmp += (kernel_reg[j] * input_feature_map_reg[j]);
	}
	output_feature_map_reg= tmp;

	output_feature_map_reg = fc_bias_2 + output_feature_map_reg;

	//relu
	output_feature_map =
		(output_feature_map_reg > 0) ? output_feature_map_reg : 0;
}
