#include"Utils.h"

void Mem2Stream(FDATA_T* in,stream<FDATA_T> & out,const int size)
{
	for (int i = 0; i < size; i++)
	{
#pragma HLS PIPELINE II=1
		FDATA_T e = in[i];
		out.write(e);
	}
}