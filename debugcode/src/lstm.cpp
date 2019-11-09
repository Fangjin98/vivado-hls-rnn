#include"lstm.h"
#include"constants.h"
#include"activations.h"
#include"computerFunctions.h"

void lstm_128(
	FDATA_T input_feature_map[LSTM_BATCH_SIZE1*LSTM_INPUT_SIZE1],
	FDATA_T hidden_state[LSTM_BATCH_SIZE1*LSTM_OUTPUT_SIZE1]
)
{
	FDATA_T inputFeatureMapReg[LSTM_INPUT_SIZE1];
	FDATA_T hiddenStateReg[LSTM_OUTPUT_SIZE1] = { 0 };
	FDATA_T kernelReg[LSTM_OUTPUT_SIZE1];
	FDATA_T recurrentKernelReg[LSTM_OUTPUT_SIZE1];
	FDATA_T it_state_reg[LSTM_OUTPUT_SIZE1];
	FDATA_T ft_state_reg[LSTM_OUTPUT_SIZE1];
	FDATA_T ot_state_reg[LSTM_OUTPUT_SIZE1];
	FDATA_T ct_state_reg[LSTM_OUTPUT_SIZE1];
	FDATA_T prev_memory[LSTM_OUTPUT_SIZE1] = { 0 };

	int computer_dim = LSTM_INPUT_SIZE1 + LSTM_OUTPUT_SIZE1;

	for (int batch_id = 0; batch_id < LSTM_BATCH_SIZE1; batch_id++) {
		
		for (int i = 0; i < LSTM_INPUT_SIZE1; i++) {
#pragma HLS PIPELINE
			inputFeatureMapReg[i] =
				input_feature_map[batch_id*LSTM_INPUT_SIZE1 + i];
		}

		matrixMultiplication(
			lstm_kernel_i_1,
			inputFeatureMapReg,
			LSTM_INPUT_SIZE1,
			LSTM_OUTPUT_SIZE1,
			kernelReg
		);
		matrixMultiplication(
			lstm_recurrent_kernel_i_1,
			hiddenStateReg,
			LSTM_OUTPUT_SIZE1,
			LSTM_OUTPUT_SIZE1,
			recurrentKernelReg
		);
		for (int i = 0; i < LSTM_OUTPUT_SIZE1; i++) {
#pragma HLS PIPELINE
			it_state_reg[i] = m_sigmoid(kernelReg[i]+recurrentKernelReg[i] + lstm_bias_i_1[i]);
		}
		
		matrixMultiplication(
			lstm_kernel_f_1,
			inputFeatureMapReg,
			LSTM_INPUT_SIZE1,
			LSTM_OUTPUT_SIZE1,
			kernelReg
		);
		matrixMultiplication(
			lstm_recurrent_kernel_f_1,
			hiddenStateReg,
			LSTM_OUTPUT_SIZE1,
			LSTM_OUTPUT_SIZE1,
			recurrentKernelReg
		);
		for (int i = 0; i < LSTM_OUTPUT_SIZE1; i++) {
#pragma HLS PIPELINE
			ft_state_reg[i] = m_sigmoid(kernelReg[i] + recurrentKernelReg[i] + lstm_bias_f_1[i]);
		}

		matrixMultiplication(
			lstm_kernel_o_1,
			inputFeatureMapReg,
			LSTM_INPUT_SIZE1,
			LSTM_OUTPUT_SIZE1,
			kernelReg
		);
		matrixMultiplication(
			lstm_recurrent_kernel_o_1,
			hiddenStateReg,
			LSTM_OUTPUT_SIZE1,
			LSTM_OUTPUT_SIZE1,
			recurrentKernelReg
		);
		for (int i = 0; i < LSTM_OUTPUT_SIZE1; i++) {
#pragma HLS PIPELINE
			ot_state_reg[i] = m_sigmoid(kernelReg[i] + recurrentKernelReg[i] + lstm_bias_o_1[i]);
		}

		matrixMultiplication(
			lstm_kernel_c_1,
			inputFeatureMapReg,
			LSTM_INPUT_SIZE1,
			LSTM_OUTPUT_SIZE1,
			kernelReg
		);
		matrixMultiplication(
			lstm_recurrent_kernel_c_1,
			hiddenStateReg,
			LSTM_OUTPUT_SIZE1,
			LSTM_OUTPUT_SIZE1,
			recurrentKernelReg
		);
		for (int i = 0; i < LSTM_OUTPUT_SIZE1; i++) {
#pragma HLS PIPELINE
			ct_state_reg[i] = m_tanh(kernelReg[i] + recurrentKernelReg[i] + lstm_bias_c_1[i]);
			ct_state_reg[i] =
				ft_state_reg[i] * prev_memory[i] + it_state_reg[i] * ct_state_reg[i];
			prev_memory[i] = ct_state_reg[i];

			hiddenStateReg[i] =
				ot_state_reg[i] * m_tanh(ct_state_reg[i]);
			hidden_state[batch_id*LSTM_OUTPUT_SIZE1 + i] = hiddenStateReg[i];
			
		}
	}
}
void lstm_64(
	FDATA_T input_feature_map[LSTM_BATCH_SIZE2*LSTM_INPUT_SIZE2],
	FDATA_T hidden_state[LSTM_OUTPUT_SIZE2])
{
	FDATA_T inputFeatureMapReg[LSTM_INPUT_SIZE2];
	FDATA_T hiddenStateReg[LSTM_OUTPUT_SIZE2] = { 0 };
	FDATA_T kernelReg[LSTM_OUTPUT_SIZE2];
	FDATA_T recurrentKernelReg[LSTM_OUTPUT_SIZE2];
	FDATA_T it_state_reg[LSTM_OUTPUT_SIZE2];
	FDATA_T ft_state_reg[LSTM_OUTPUT_SIZE2];
	FDATA_T ot_state_reg[LSTM_OUTPUT_SIZE2];
	FDATA_T ct_state_reg[LSTM_OUTPUT_SIZE2];
	FDATA_T prev_memory[LSTM_OUTPUT_SIZE2] = { 0 };

	int computer_dim = LSTM_INPUT_SIZE2 + LSTM_OUTPUT_SIZE2;

	for (int batch_id = 0; batch_id < LSTM_BATCH_SIZE2; batch_id++) {

		for (int i = 0; i < LSTM_INPUT_SIZE2; i++) {
			inputFeatureMapReg[i] =
				input_feature_map[batch_id*LSTM_INPUT_SIZE2 + i];
		}

		matrixMultiplication(
			lstm_kernel_i_2,
			inputFeatureMapReg,
			LSTM_INPUT_SIZE2,
			LSTM_OUTPUT_SIZE2,
			kernelReg
		);
		matrixMultiplication(
			lstm_recurrent_kernel_i_2,
			hiddenStateReg,
			LSTM_OUTPUT_SIZE2,
			LSTM_OUTPUT_SIZE2,
			recurrentKernelReg
		);
		for (int i = 0; i < LSTM_OUTPUT_SIZE2; i++) {
#pragma HLS PIPELINE
			it_state_reg[i] = m_sigmoid(kernelReg[i]+recurrentKernelReg[i] + lstm_bias_i_2[i]);
		}

		matrixMultiplication(
			lstm_kernel_f_2,
			inputFeatureMapReg,
			LSTM_INPUT_SIZE2,
			LSTM_OUTPUT_SIZE2,
			kernelReg
		);
		matrixMultiplication(
			lstm_recurrent_kernel_f_2,
			hiddenStateReg,
			LSTM_OUTPUT_SIZE2,
			LSTM_OUTPUT_SIZE2,
			recurrentKernelReg
		);
		for (int i = 0; i < LSTM_OUTPUT_SIZE2; i++) {
#pragma HLS PIPELINE
			ft_state_reg[i] = m_sigmoid(kernelReg[i]+recurrentKernelReg[i] + lstm_bias_f_2[i]);
		}

		matrixMultiplication(
			lstm_kernel_o_2,
			inputFeatureMapReg,
			LSTM_INPUT_SIZE2,
			LSTM_OUTPUT_SIZE2,
			kernelReg
		);
		matrixMultiplication(
			lstm_recurrent_kernel_o_2,
			hiddenStateReg,
			LSTM_OUTPUT_SIZE2,
			LSTM_OUTPUT_SIZE2,
			recurrentKernelReg
		);
		for (int i = 0; i < LSTM_OUTPUT_SIZE2; i++) {
#pragma HLS PIPELINE
			ot_state_reg[i] = m_sigmoid(kernelReg[i] + recurrentKernelReg[i] + lstm_bias_o_2[i]);
		}

		matrixMultiplication(
			lstm_kernel_c_2,
			inputFeatureMapReg,
			LSTM_INPUT_SIZE2,
			LSTM_OUTPUT_SIZE2,
			kernelReg
		);
		matrixMultiplication(
			lstm_recurrent_kernel_c_2,
			hiddenStateReg,
			LSTM_OUTPUT_SIZE2,
			LSTM_OUTPUT_SIZE2,
			recurrentKernelReg
		);
		for (int i = 0; i < LSTM_OUTPUT_SIZE2; i++) {
#pragma HLS PIPELINE
			ct_state_reg[i] = m_tanh(kernelReg[i] + recurrentKernelReg[i] + lstm_bias_c_2[i]);
			ct_state_reg[i] =
				ft_state_reg[i] * prev_memory[i] + it_state_reg[i] * ct_state_reg[i];

			hiddenStateReg[i] = ot_state_reg[i] * m_tanh(ct_state_reg[i]);
			prev_memory[i] = ct_state_reg[i];
		}
	}

	for (int i = 0; i < LSTM_OUTPUT_SIZE2; i++) {
		hidden_state[i] = hiddenStateReg[i];
	}
}