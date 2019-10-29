#include"activations.h"
#include"types.h"
#include<cmath>


FDATA_T m_sigmoid(FDATA_T input) {
	// standard sigmoid
	//return 1 / (1 + exp((double)-input));

	//keras hard_sigmoid
	double tmp=(double) input;

	if(tmp<-2.5) return 0;
	else if(-2.5<=tmp<=2.5){ return (0.2*tmp + 0.5);}
	else return 1;
}

FDATA_T m_tanh(FDATA_T input) {
	return tanh( (double) input);
}

FDATA_T m_relu(FDATA_T input){
	if(input>0) return input;
	return 0;
}
