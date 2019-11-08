#include"activations.h"
#include<math.h>

FDATA_T m_sigmoid(FDATA_T input) {
	return 1 / (1 + exp((double)-input));
}

FDATA_T m_tanh(FDATA_T input) {
	return tanh((double)input);
}