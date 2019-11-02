#include"lstm.h"
#include"floatConstants.h"
#include"activations.h"

void lstm_128(
	FDATA_T input_feature_map[LSTM_BATCH_SIZE1*LSTM_INPUT_SIZE1],
	FDATA_T prev_hidden_state[LSTM_OUTPUT_SIZE1],
	FDATA_T prev_memory_cell[LSTM_OUTPUT_SIZE1],
	FDATA_T hidden_state[LSTM_BATCH_SIZE1*LSTM_OUTPUT_SIZE1],
	FDATA_T memory_cell[LSTM_OUTPUT_SIZE1]
)
{
	FDATA_T input_reg[LSTM_INPUT_SIZE1];
	FDATA_T hidden_reg[LSTM_OUTPUT_SIZE1];
	FDATA_T computer_reg[LSTM_OUTPUT_SIZE1];
	FDATA_T output_state_reg[LSTM_OUTPUT_SIZE1];
	FDATA_T it_state_reg[LSTM_OUTPUT_SIZE1];
	FDATA_T ft_state_reg[LSTM_OUTPUT_SIZE1];
	FDATA_T ot_state_reg[LSTM_OUTPUT_SIZE1];
	FDATA_T ct_state_reg[LSTM_OUTPUT_SIZE1];

	int computer_dim = LSTM_INPUT_SIZE1 + LSTM_OUTPUT_SIZE1;

	for (int i = 0; i < LSTM_OUTPUT_SIZE1; i++) {
		prev_memory_cell[i] = 0;
	}
	for (int batch_id = 0; batch_id < LSTM_BATCH_SIZE1; batch_id++) {
		for (int i = 0; i < LSTM_INPUT_SIZE1; i++) {
			input_reg[i] =
				input_feature_map[batch_id*LSTM_INPUT_SIZE1 + i];
		}

		for (int i = 0; i < LSTM_OUTPUT_SIZE1; i++) {
			hidden_reg[i] = prev_hidden_state[i];
			
		}

		vectorProduct(
			lstm_kernel_i_1,
			input_reg,
			LSTM_OUTPUT_SIZE1, 
			LSTM_INPUT_SIZE1,
			computer_reg
		);
		vectorProduct(
			lstm_recurrent_kernel_i_1,
			hidden_reg,
			LSTM_OUTPUT_SIZE1,
			LSTM_OUTPUT_SIZE1,
			output_state_reg
		);
		for (int i = 0; i < LSTM_OUTPUT_SIZE1; i++) {
			it_state_reg[i] = m_sigmoid(computer_reg[i]+output_state_reg[i] + lstm_bias_i_1[i]);
		}
		
		vectorProduct(
			lstm_kernel_f_1,
			input_reg,
			LSTM_OUTPUT_SIZE1,
			LSTM_INPUT_SIZE1,
			computer_reg
		);
		vectorProduct(
			lstm_recurrent_kernel_f_1,
			hidden_reg,
			LSTM_OUTPUT_SIZE1,
			LSTM_OUTPUT_SIZE1,
			output_state_reg
		);
		for (int i = 0; i < LSTM_OUTPUT_SIZE1; i++) {
			ft_state_reg[i] = m_sigmoid(computer_reg[i] + output_state_reg[i] + lstm_bias_f_1[i]);
		}

		vectorProduct(
			lstm_kernel_o_1,
			input_reg,
			LSTM_OUTPUT_SIZE1,
			LSTM_INPUT_SIZE1,
			computer_reg
		);
		vectorProduct(
			lstm_recurrent_kernel_o_1,
			hidden_reg,
			LSTM_OUTPUT_SIZE1,
			LSTM_OUTPUT_SIZE1,
			output_state_reg
		);
		for (int i = 0; i < LSTM_OUTPUT_SIZE1; i++) {
			ot_state_reg[i] = m_sigmoid(computer_reg[i] + output_state_reg[i] + lstm_bias_o_1[i]);
		}

		vectorProduct(
			lstm_kernel_c_1,
			input_reg,
			LSTM_OUTPUT_SIZE1,
			LSTM_INPUT_SIZE1,
			computer_reg
		);
		vectorProduct(
			lstm_recurrent_kernel_c_1,
			hidden_reg,
			LSTM_OUTPUT_SIZE1,
			LSTM_OUTPUT_SIZE1,
			output_state_reg
		);
		for (int i = 0; i < LSTM_OUTPUT_SIZE1; i++) {
			ct_state_reg[i] = m_tanh(computer_reg[i] + output_state_reg[i] + lstm_bias_c_1[i]);
		}

		for (int i = 0; i < LSTM_OUTPUT_SIZE1; i++) {
			ct_state_reg[i] =
				ft_state_reg[i] * prev_memory_cell[i] + it_state_reg[i] * ct_state_reg[i];
		}

		for (int i = 0; i < LSTM_OUTPUT_SIZE1; i++) {
			hidden_state[batch_id*LSTM_OUTPUT_SIZE1 + i] =
				ot_state_reg[i] * m_tanh(ct_state_reg[i]);
		}
		for (int i = 0; i < LSTM_OUTPUT_SIZE1; i++) {
			memory_cell[i] = ct_state_reg[i];
			prev_memory_cell[i] = memory_cell[i];
			prev_hidden_state[i] = hidden_state[batch_id*LSTM_OUTPUT_SIZE1 + i];
		}
	}
}
void lstm_64(
	FDATA_T input_feature_map[LSTM_BATCH_SIZE2*LSTM_INPUT_SIZE2],
	FDATA_T prev_hidden_state[LSTM_OUTPUT_SIZE2],
	FDATA_T prev_memory_cell[LSTM_OUTPUT_SIZE2],
	FDATA_T hidden_state[LSTM_BATCH_SIZE2*LSTM_OUTPUT_SIZE2],
	FDATA_T memory_cell[LSTM_OUTPUT_SIZE2])
{
	FDATA_T input_reg[LSTM_INPUT_SIZE2];
	FDATA_T hidden_reg[LSTM_OUTPUT_SIZE2];
	FDATA_T computer_reg[LSTM_OUTPUT_SIZE2];
	FDATA_T output_state_reg[LSTM_OUTPUT_SIZE2];
	FDATA_T it_state_reg[LSTM_OUTPUT_SIZE2];
	FDATA_T ft_state_reg[LSTM_OUTPUT_SIZE2];
	FDATA_T ot_state_reg[LSTM_OUTPUT_SIZE2];
	FDATA_T ct_state_reg[LSTM_OUTPUT_SIZE2];

	//int computer_dim = LSTM_INPUT_SIZE2 + LSTM_OUTPUT_SIZE2;

	for (int i = 0; i < LSTM_OUTPUT_SIZE2; i++) {
		prev_memory_cell[i] = 0;
	}
	for (int batch_id = 0; batch_id < LSTM_BATCH_SIZE2; batch_id++) {
		for (int j = 0; j < LSTM_INPUT_SIZE2; j++) {
			input_reg[j] =
				input_feature_map[batch_id*LSTM_INPUT_SIZE2 + j];
		}
		for (int j = 0; j < LSTM_OUTPUT_SIZE2; j++) {
			hidden_reg[j] =	prev_hidden_state[j];
		}

		vectorProduct(
			lstm_kernel_i_2,
			input_reg,
			LSTM_OUTPUT_SIZE2,
			LSTM_INPUT_SIZE2,
			computer_reg
		);
		vectorProduct(
			lstm_recurrent_kernel_i_2,
			hidden_reg,
			LSTM_OUTPUT_SIZE2,
			LSTM_OUTPUT_SIZE2,
			output_state_reg
		);
		for (int i = 0; i < LSTM_OUTPUT_SIZE2; i++) {
			it_state_reg[i] = m_sigmoid(computer_reg[i]+output_state_reg[i] + lstm_bias_i_2[i]);
		}
		vectorProduct(
			lstm_kernel_f_2,
			input_reg,
			LSTM_OUTPUT_SIZE2,
			LSTM_INPUT_SIZE2,
			computer_reg
		);
		vectorProduct(
			lstm_recurrent_kernel_f_2,
			hidden_reg,
			LSTM_OUTPUT_SIZE2,
			LSTM_OUTPUT_SIZE2,
			output_state_reg
		);
		for (int i = 0; i < LSTM_OUTPUT_SIZE2; i++) {
			ft_state_reg[i] = m_sigmoid(computer_reg[i] + output_state_reg[i] + lstm_bias_f_2[i]);
		}

		vectorProduct(
			lstm_kernel_o_2,
			input_reg,
			LSTM_OUTPUT_SIZE2,
			LSTM_INPUT_SIZE2,
			computer_reg
		);
		vectorProduct(
			lstm_recurrent_kernel_o_2,
			hidden_reg,
			LSTM_OUTPUT_SIZE2,
			LSTM_OUTPUT_SIZE2,
			output_state_reg
		);
		for (int i = 0; i < LSTM_OUTPUT_SIZE2; i++) {
			ot_state_reg[i] = m_sigmoid(computer_reg[i] + output_state_reg[i] + lstm_bias_o_2[i]);
		}

		vectorProduct(
			lstm_kernel_c_2,
			input_reg,
			LSTM_OUTPUT_SIZE2,
			LSTM_INPUT_SIZE2,
			computer_reg
		);
		vectorProduct(
			lstm_recurrent_kernel_c_2,
			hidden_reg,
			LSTM_OUTPUT_SIZE2,
			LSTM_OUTPUT_SIZE2,
			output_state_reg
		);
		for (int i = 0; i < LSTM_OUTPUT_SIZE2; i++) {
			ct_state_reg[i] = m_tanh(computer_reg[i] + output_state_reg[i] + lstm_bias_c_2[i]);
		}

		for (int i = 0; i < LSTM_OUTPUT_SIZE2; i++) {
			ct_state_reg[i] =
				ft_state_reg[i] * prev_memory_cell[i] + it_state_reg[i] * ct_state_reg[i];
		}

		for (int i = 0; i < LSTM_OUTPUT_SIZE2; i++) {
			hidden_state[batch_id*LSTM_OUTPUT_SIZE2 + i] =
				ot_state_reg[i] * m_tanh(ct_state_reg[i]);
		}
		for (int i = 0; i < LSTM_OUTPUT_SIZE2; i++) {
			memory_cell[i] = ct_state_reg[i];
			prev_memory_cell[i] = memory_cell[i];
			prev_hidden_state[i] = hidden_state[batch_id*LSTM_OUTPUT_SIZE2 + i];
		}
	}
}

void vectorProduct(
	const FDATA_T *weight_matrix,
	FDATA_T * input,
	int dim1,
	int dim2,
	FDATA_T * output)
{
	for (int i = 0; i < dim1; i++) {
		FDATA_T tmp = 0;
		for (int j = 0; j < dim2; j++) {
			tmp += (input[j] * weight_matrix[i*dim2 + j]);
		}
		output[i] = tmp;
	}
}