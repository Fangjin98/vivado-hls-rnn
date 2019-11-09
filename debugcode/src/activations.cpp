#include"activations.h"
#include<math.h>

FDATA_T m_sigmoid(FDATA_T input) {
	//return 1 / (1 + exp((double)-input));

	if(input>2.5) return 1;
	else if(input<-2.5) return 0;
	else {return (0.2*(float) input+0.5);}
}

FDATA_T m_tanh(FDATA_T input) {
	return tanh((double)input);
}

FDATA_T m_relu(FDATA_T input) {
	if (input > 0) return input;
	return 0;
}
