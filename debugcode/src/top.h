#ifndef TOP_H_
#define TOP_H_
#include"types.h"
//void top(
//	FDATA_T *input,
//	FDATA_T *hidden_1,
//	FDATA_T *hidden_2,
//	FDATA_T &output
//);

void top(
	FDATA_T *input,
	FDATA_T &output
);
void load_fc_input(
	FDATA_T *lstm_array,
	FDATA_T *fc_array);
#endif
