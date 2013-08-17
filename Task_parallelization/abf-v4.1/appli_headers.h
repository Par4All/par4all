// Copyright © THALES 2010 All rights reserved
//
// THE PROGRAM IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED INCLUDING,
// WITHOUT LIMITATION, ANY WARRANTIES ON ITS, NON-INFRINGEMENT, MERCHANTABILITY, SECURED, INNOVATIVE OR RELEVANT NATURE, FITNESS 
// FOR A PARTICULAR PURPOSE OR COMPATIBILITY WITH ANY EQUIPMENT OR SOFTWARE.

#ifndef __appli_headers__
#define __appli_headers__

#define ATL_REC_TIME_BURST_1 0.003
#define ATL_FE 10000000
#define ATL_CD 0
#define ATL_nb_cd 2000.0
#define ATL_RSB -20
#define ATL_NUM_TRACKS 64
#define ATL_NUM_BEAMS 16
#define ATL_FORM_FACTOR 0.1
#define ATL_COMPL_FORM_FACTOR 0.9
#define ATL_NUM_PULSE_BURST_1 7
#define ATL_REC_TIME_BURST_3 2.0E-4
#define ATL_NUM_PULSE_BURST_3 19
#define ATL_REC_TIME_BURST_2 1.0E-3
#define ATL_NUM_PULSE_BURST_2 11
#define ATL_nb_pls_pow2 32

#include "GPUradar.h"

typedef struct {
} s_para_vol;

typedef struct {
} s_para_vol_out;

typedef struct {
	int tab_index[5];
	Cplfloat ValSteer[16][64];
} s_para_global;

typedef union {
	struct {
	} a; /* fin struct seg */
} u_para_private; /* fin union private */

/**
 * .ApplicationModel.sel - mode: init rate: *
 **/
void trigger_20100826173740406();

/**
 * .ApplicationModel.CTR - mode: init rate: *
 **/
void trigger_20100826135651984();

#endif
