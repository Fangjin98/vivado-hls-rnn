#include"top.h"
#include"fc.h"
#include"lstm.h"
#include"constants.h"

void top(
	FDATA_T input[LSTM_BATCH_SIZE1*LSTM_INPUT_SIZE1],
	FDATA_T &output
) {

	FDATA_T fc_output_feature_map_1[FC_OUTPUT_SIZE1];
	FDATA_T hidden_1[LSTM_BATCH_SIZE1*LSTM_OUTPUT_SIZE1];
	FDATA_T hidden_2[LSTM_OUTPUT_SIZE2];

	lstm_128(
		input,
		hidden_1
	);

	lstm_64(
		hidden_1,
		hidden_2
	);

	fc_64(
		hidden_2,
		fc_output_feature_map_1
	);

	fc_16(
		fc_output_feature_map_1,
		output
	);
}

void load_fc_input(
	FDATA_T lstm_array[LSTM_BATCH_SIZE2*LSTM_OUTPUT_SIZE2],
	FDATA_T fc_array[FC_INPUT_SIZE1]
) {
	int offset = LSTM_BATCH_SIZE2 - 1;
	for (int i = 0; i < LSTM_OUTPUT_SIZE2; i++) {
#pragma HLS DATAFLOW
		fc_array[i] = lstm_array[offset*LSTM_OUTPUT_SIZE2 + i];
	}
}
