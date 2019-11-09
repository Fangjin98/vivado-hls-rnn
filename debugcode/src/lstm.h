#ifndef LSTM_H_
#define LSTM_H_
#include"types.h"
void lstm_128(
	FDATA_T *input_feature_map,
	FDATA_T *hidden_state
);

void lstm_64(
	FDATA_T *input_feature_map,
	FDATA_T *hidden_state);

#endif
