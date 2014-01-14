// Copyright © THALES 2010 All rights reserved
//
// THE PROGRAM IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED INCLUDING,
// WITHOUT LIMITATION, ANY WARRANTIES ON ITS, NON-INFRINGEMENT, MERCHANTABILITY, SECURED, INNOVATIVE OR RELEVANT NATURE, FITNESS 
// FOR A PARTICULAR PURPOSE OR COMPATIBILITY WITH ANY EQUIPMENT OR SOFTWARE.

#if defined(__cplusplus) ||defined(_cplusplus)
extern "C" {
#endif
#include "appli_headers.h"
#if defined(__cplusplus) ||defined(_cplusplus)
}
#endif

s_para_vol param_vol;
s_para_vol *ptr_vol = &param_vol;

s_para_vol_out param_vol_out;
s_para_vol_out *ptr_vol_out = &param_vol_out;

s_para_global param_global;

//s_para_global *ptr_global = &param_global;
//int tab_index[5];
//Cplfloat ValSteer[16][64];

u_para_private param_private;
void *ptr_vol_void = &param_vol;
void *ptr_vol_out_void = &param_vol_out;
int size_vol = sizeof(s_para_vol);
int size_vol_out = sizeof(s_para_vol_out);

/**
 * .ApplicationModel.sel - mode: init rate: *
 **/
void trigger_1(int tab_index[5]) {
  //MOTIF
	tab_index[0] = 20;
	tab_index[1] = 200;
	tab_index[2] = 700;
	tab_index[3] = 1000;
	tab_index[4] = 1200;
}


