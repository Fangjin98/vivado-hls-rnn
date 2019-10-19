#include<iostream>
#include"lstm.h"
#include"constants.h"

void generateTestData(){
    for(int i=0;i<LSTM_BATCH_SIZE1;i++){
        for(int j=0;j<LSTM_INPUT_SIZE1;j++){
            lstm_input_feature_map_1[i*LSTM_INPUT_SIZE1+j]=1;
			lstm_prev_hidden_test_1[i*LSTM_OUTPUT_SIZE1 + j] = 0;
			lstm_prev_memory_test_1[i*LSTM_OUTPUT_SIZE1 + j] = 0;
        }
    }

	int dim = LSTM_INPUT_SIZE1 + LSTM_OUTPUT_SIZE1;
	for (int i = 0; i < LSTM_OUTPUT_SIZE1; i++) {
		for (int j = 0; j < dim; j++) {
			lstm_weight_c_1[i*dim + j] = 2;
			lstm_weight_i_1[i*dim + j] = 2;
			lstm_weight_f_1[i*dim + j] = 2;
			lstm_weight_o_1[i*dim + j] = 2;
		}
	}

	for (int i = 0; i < LSTM_OUTPUT_SIZE1; i++) {
		lstm_bias_c_1[i] = 1;
		lstm_bias_f_1[i] = 1;
		lstm_bias_i_1[i] = 1;
		lstm_bias_o_1[i] = 1;
	}
}

int main(){
    generateTestData();

	/*fc_64_16(
		fc_input_feature_map_1[FC_BATCH_SIZE1*FC_INPUT_SIZE1]
		,fc_output_feature_map_1[FC_BATCH_SIZE1*FC_OUTPUT_SIZE1]);*/

	lstm_128(
		lstm_input_feature_map_1,
		lstm_prev_hidden_test_1,
		lstm_prev_memory_test_1,
		lstm_memory_test_1,
		lstm_hidden_test_1
	);
    for(int i=0;i<FC_OUTPUT_SIZE1;i++){
        for(int j=0;j<FC_BATCH_SIZE1;j++){
            std::cout<<fc_output_feature_map_1[i*FC_BATCH_SIZE1+j]<<" ";
        }
        std::cout<<std::endl;
    }

    return 0;
}