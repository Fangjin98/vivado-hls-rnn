#include<iostream>
#include"constants.h"
#include"top.h"

void generateTestData();
//void generateFcTestData();
void generateLstmTestData();

int main(){
    //generateTestData();

	//build & inference
	for (int i = 0; i < TEST_DATA_SIZE; i++) {
		for (int j = 0; j < LSTM_BATCH_SIZE1; j++) {
			for (int k = 0; k < LSTM_INPUT_SIZE1; k++) {
				lstm_input_feature_map_1[j*LSTM_INPUT_SIZE1 + k] =
					test_data_input[(i*TEST_DATA_STEP + j)*TEST_DATA_DIM+k];
			}
		}
		if (i == 0) {
			for (int j = 0; j < LSTM_OUTPUT_SIZE1; j++) {
				lstm_prev_hidden_1[j] = 0;
			}
			for (int j = 0; j < LSTM_OUTPUT_SIZE2; j++) {
				lstm_prev_hidden_2[j] = 0;
			}
		}
		else {
			for (int j = 0; j < LSTM_OUTPUT_SIZE1; j++) {
				lstm_prev_hidden_1[j] = lstm_hidden_1[4 * LSTM_OUTPUT_SIZE1+j];
			}
			for (int j = 0; j < LSTM_OUTPUT_SIZE2; j++) {
				lstm_prev_hidden_2[j] = lstm_hidden_2[4 * LSTM_OUTPUT_SIZE2 + j];
			}
		}
		top(lstm_input_feature_map_1, 
			lstm_prev_hidden_1,
			lstm_prev_hidden_2,
			fc_output_feature_map_2);
		std::cout << fc_output_feature_map_2<<std::endl;
	}
    return 0;
}

void generateTestData() {
	//generateFcTestData();
	//generateLstmTestData();
}

void generateLstmTestData() {
	for (int i = 0; i < LSTM_BATCH_SIZE1; i++) {
		for (int j = 0; j < LSTM_INPUT_SIZE1; j++) {
			lstm_input_feature_map_1[i*LSTM_INPUT_SIZE1 + j] = 1;
		}
	}
}