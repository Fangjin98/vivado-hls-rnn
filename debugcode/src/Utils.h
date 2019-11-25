#ifndef UTILS
#define UTILS
#include"types.h"
#include"hls_stream.h"
void Mem2Stream(FDATA_T * in, stream<FDATA_T> & out,const int size);
}
#endif