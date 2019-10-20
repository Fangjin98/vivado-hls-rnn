#include<iostream>
#include"constants.h"
#include"top.h"

void generateTestData();
void generateFcTestData();
void generateLstmTestData();

int main(){
    generateTestData();

	//build & inference
	top(lstm_input_feature_map_1, fc_output_feature_map_2);

    for(int i=0;i<FC_OUTPUT_SIZE2;i++){
        for(int j=0;j<FC_BATCH_SIZE2;j++){
            std::cout<<fc_output_feature_map_2[i*FC_BATCH_SIZE1+j]<<" ";
        }
        std::cout<<std::endl;
    }

    return 0;
}

void generateTestData() {
	generateFcTestData();
	generateLstmTestData();
}

void generateFcTestData() {
	for (int i = 0; i < FC_BATCH_SIZE1; i++) {
		for (int j = 0; j < FC_INPUT_SIZE1; j++) {
			fc_input_feature_map_1[i*FC_BATCH_SIZE1 + j] = 1;
		}
	}
	for (int i = 0; i < FC_OUTPUT_SIZE1; i++) {
		for (int j = 0; j < FC_INPUT_SIZE1; j++) {
			fc_kernel_1[i*FC_INPUT_SIZE1 + j] = 1;
		}
		fc_bias_1[i] = 1;
	}

	for (int i = 0; i < FC_OUTPUT_SIZE2; i++) {
		for (int j = 0; j < FC_INPUT_SIZE2; j++) {
			fc_kernel_2[i*FC_INPUT_SIZE2 + j] = 1;
		}
		fc_bias_2[i] = 1;
	}
}

void generateLstmTestData() {
	for (int i = 0; i < LSTM_BATCH_SIZE1; i++) {
		for (int j = 0; j < LSTM_INPUT_SIZE1; j++) {
			lstm_input_feature_map_1[i*LSTM_INPUT_SIZE1 + j] = 1;
		}
	}

	for (int i = 0; i < LSTM_BATCH_SIZE1; i++) {
		for (int j = 0; j < LSTM_OUTPUT_SIZE1; j++) {
			lstm_prev_hidden_test_1[i*LSTM_OUTPUT_SIZE1 + j] = 0;
			lstm_prev_memory_test_1[i*LSTM_OUTPUT_SIZE1 + j] = 0;
		}
	}

	for (int i = 0; i < LSTM_BATCH_SIZE2; i++) {
		for (int j = 0; j < LSTM_OUTPUT_SIZE2; j++) {
			lstm_prev_hidden_test_2[i*LSTM_OUTPUT_SIZE2 + j] = 0;
			lstm_prev_memory_test_2[i*LSTM_OUTPUT_SIZE2 + j] = 0;
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

	dim = LSTM_INPUT_SIZE2 + LSTM_OUTPUT_SIZE2;
	for (int i = 0; i < LSTM_OUTPUT_SIZE2; i++) {
		for (int j = 0; j < dim; j++) {
			lstm_weight_c_2[i*dim + j] = 2;
			lstm_weight_i_2[i*dim + j] = 2;
			lstm_weight_f_2[i*dim + j] = 2;
			lstm_weight_o_2[i*dim + j] = 2;
		}
	}

	for (int i = 0; i < LSTM_OUTPUT_SIZE2; i++) {
		lstm_bias_c_2[i] = 1;
		lstm_bias_f_2[i] = 1;
		lstm_bias_i_2[i] = 1;
		lstm_bias_o_2[i] = 1;
	}
}