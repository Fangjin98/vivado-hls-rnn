#include"top.h"
#include"fc.h"
#include"lstm.h"
#include"constants.h"

void top(
	FDATA_T input[LSTM_BATCH_SIZE1*LSTM_INPUT_SIZE1],
	FDATA_T output[FC_BATCH_SIZE2*FC_OUTPUT_SIZE2]
) {
	lstm_128(
		input,
		lstm_prev_hidden_test_1,
		lstm_prev_memory_test_1,
		lstm_memory_test_1,
		lstm_hidden_test_1
	);

	lstm_64(
		lstm_hidden_test_1,
		lstm_prev_hidden_test_2,
		lstm_prev_memory_test_2,
		lstm_memory_test_2,
		lstm_hidden_test_2
	);

	fc_64(
		lstm_hidden_test_2,
		fc_output_feature_map_1
	);


	fc_16(
		fc_output_feature_map_1,
		output
	);
}
