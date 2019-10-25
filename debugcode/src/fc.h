#ifndef FC_H_
#define FC_H_
#include"types.h"
void fc_64(
	FDATA_T *input_feature_map,
	FDATA_T *output_feature_map);

void fc_16(
	FDATA_T *input_feature_map,
	FDATA_T &output_feature_map);

#endif

