/*
 * $Id$
 *
 * Stub functions for old passes that have been moved out.
 */

#include <stdio.h>
#include "genC.h"
#include "misc.h"

#define STUB(p)								     \
bool p(string module) 							     \
{									     \
    pips_user_warning("pass on %s no more implemented or linked\n", module); \
    return FALSE;							     \
}

/* 
 * PAF-related stuff, moved out 16/04/1998 by FC.
 * No problem to move it back in! Just port to new newgen;-)
 */
STUB(array_dfg)
STUB(print_array_dfg)
STUB(scheduling)
STUB(print_bdt)
STUB(prgm_mapping)
STUB(print_plc)
STUB(reindexing)
STUB(print_parallelizedCMF_code)
STUB(print_parallelizedCRAFT_code)
STUB(static_controlize)
STUB(print_code_static_control)
