#include"top.h"
#include"fc.h"
#include"lstm.h"
#include"constants.h"

void top(
	FDATA_T input[LSTM_BATCH_SIZE1*LSTM_INPUT_SIZE1],
	FDATA_T lstm_prev_hidden_1[LSTM_OUTPUT_SIZE1],
	FDATA_T lstm_prev_hidden_2[LSTM_OUTPUT_SIZE2],
	FDATA_T &output
) {
	lstm_128(
		input,
		lstm_prev_hidden_1,
		lstm_hidden_1
	);

	lstm_64(
		lstm_hidden_1,
		lstm_prev_hidden_2,
		lstm_hidden_2
	);

	FDATA_T fc_input_tmp[FC_INPUT_SIZE1];
	load_fc_input(lstm_hidden_2, fc_input_tmp);

	fc_64(
		fc_input_tmp,
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
		fc_array[i] = lstm_array[offset*LSTM_OUTPUT_SIZE2 + i];
	}
}