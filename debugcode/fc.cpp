#include "fc.h"
#include"constants.h"

//#pragma SDS data zero_copy(fc_kernel[0: FC_OUTPUT_SIZE * FC_INPUT_SIZE])
//#pragma SDS data zero_copy(fc_bias[0: FC_OUTPUT_SIZE])
//
//#pragma SDS data zero_copy( \
//    input_feature_map[0: BATCH_SIZE * RNN_STATE_SIZE])
//#pragma SDS data zero_copy(output_feature_map[0: BATCH_SIZE * FC_OUTPUT_SIZE])

void fc_64_16(
	FDATA_T input_feature_map[FC_BATCH_SIZE1*FC_INPUT_SIZE1],
	FDATA_T output_feature_map[FC_BATCH_SIZE1*FC_OUTPUT_SIZE1]) {

	FDATA_T input_feature_map_reg[FC_BATCH_SIZE1*FC_INPUT_SIZE1];
	FDATA_T output_feature_map_reg[FC_BATCH_SIZE1];
	FDATA_T kernel_reg[FC_INPUT_SIZE1];

//#pragma HLS ARRAY_PARTITION variable=input_feature_map_reg \
//    dim=2 cyclic factor=32
//#pragma HLS ARRAY_PARTITION variable=input_feature_map_reg \
//    dim=1 cyclic factor=2
//#pragma HLS ARRAY_PARTITION variable=output_feature_map_reg \
//    dim=1 cyclic factor=32
//#pragma HLS ARRAY_PARTITION variable=kernel_reg dim=1 cyclic factor=32

	//load input feature map
	for (LDATA_T batch_iter = 0; batch_iter < FC_BATCH_SIZE1; batch_iter++) {
		for (LDATA_T i = 0; i < FC_INPUT_SIZE1; i++)
		{
#pragma HLS PIPELINE
			input_feature_map_reg[batch_iter*FC_INPUT_SIZE1 + i] =
				input_feature_map[batch_iter*FC_INPUT_SIZE1 + i];
		}
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

		//computer
		for (LDATA_T i = 0; i < FC_BATCH_SIZE1; i++) {
			FDATA_T tmp = 0;
			for (LDATA_T j = 0; j < FC_BATCH_SIZE1; j++) {
				tmp += (kernel_reg[j] * input_feature_map_reg[i*FC_INPUT_SIZE1 + j]);
			}
			output_feature_map_reg[i] = tmp;
		}

		LDATA_T output_feature_map_offset = output_feature_map_index * FC_BATCH_SIZE1;
		for (LDATA_T i = 0; i < FC_BATCH_SIZE1; i++) {
#pragma HLS PIPELINE
			output_feature_map[i+output_feature_map_offset] =
				fc_bias_1[output_feature_map_index] + output_feature_map_reg[i];
		}
	}
}


void fc_16_1(FDATA_T input_feature_map[FC_BATCH_SIZE1*FC_INPUT_SIZE1],
	FDATA_T output_feature_map[FC_BATCH_SIZE1*FC_OUTPUT_SIZE1]) {

	FDATA_T input_feature_map_reg[FC_BATCH_SIZE1*FC_INPUT_SIZE1];
	FDATA_T output_feature_map_reg[FC_BATCH_SIZE1];
	FDATA_T kernel_reg[FC_INPUT_SIZE1];

	//#pragma HLS ARRAY_PARTITION variable=input_feature_map_reg \
	//    dim=2 cyclic factor=32
	//#pragma HLS ARRAY_PARTITION variable=input_feature_map_reg \
	//    dim=1 cyclic factor=2
	//#pragma HLS ARRAY_PARTITION variable=output_feature_map_reg \
	//    dim=1 cyclic factor=32
	//#pragma HLS ARRAY_PARTITION variable=kernel_reg dim=1 cyclic factor=32

		//load input feature map
	for (LDATA_T batch_iter = 0; batch_iter < FC_BATCH_SIZE1; batch_iter++) {
		for (LDATA_T i = 0; i < FC_INPUT_SIZE1; i++)
		{
#pragma HLS PIPELINE
			input_feature_map_reg[batch_iter*FC_INPUT_SIZE1 + i] =
				input_feature_map[batch_iter*FC_INPUT_SIZE1 + i];
		}
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
			kernel_reg[i] = fc_kernel_1[kernel_offset + i];
		}

		//computer
		for (LDATA_T i = 0; i < FC_BATCH_SIZE1; i++) {
			FDATA_T tmp = 0;
			for (LDATA_T j = 0; j < FC_BATCH_SIZE1; j++) {
				tmp += (kernel_reg[j] * input_feature_map_reg[i*FC_INPUT_SIZE1 + j]);
			}
			output_feature_map_reg[i] = tmp;
		}

		LDATA_T output_feature_map_offset = output_feature_map_index * FC_BATCH_SIZE1;
		for (LDATA_T i = 0; i < FC_BATCH_SIZE1; i++) {
#pragma HLS PIPELINE
			output_feature_map[i + output_feature_map_offset] =
				fc_bias_1[output_feature_map_index] + output_feature_map_reg[i];
		}
	}
}
