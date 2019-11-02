#include"activations.h"
#include<math.h>


FDATA_T m_sigmoid(FDATA_T input) {

	FDATA_T tmp = input;
	// standard sigmoid
	//return 1 / (1 + exp(-tmp));

	//keras hard_sigmoid
	if(tmp<-2.5) return 0;
	else if(-2.5<=tmp<=2.5){ return (0.2*tmp + 0.5);}
	else return 1;
}

FDATA_T m_tanh(FDATA_T input) {
	FDATA_T tmp = input;

	return ((exp(tmp) - exp(-tmp)) / (exp(tmp) + exp(-tmp)));
	//return tanh((double) input);
}

FDATA_T m_relu(FDATA_T input){
	return (input > 0) ? input : 0;
	/*if(input>0) return input;
	else return 0;*/
}
