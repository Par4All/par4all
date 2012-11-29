/* Copyright 2007, 2008 Alain Muller, Frederique Silber-Chaussumier

This file is part of STEP.

The program is distributed under the terms of the GNU General Public
License.
*/

#ifdef HAVE_CONFIG_H
    #include "pips_config.h"
#endif

/*

Module STEP

regroupement de la majorité des includes

*/

#include "step.h" // include step-local.h

#include "pipsdbm.h" // for db_get_memory_resource, ...
#include "properties.h" // for get_string_property, ...

#include "misc.h" // for pips_debug, pips_assert, ...
#include "ri-util.h"
#include "text-util.h" // for text_to_string
#include "effects-util.h" // for action_equal_p, statement_mapping, ...
#include "effects-generic.h" // needed by effects-convex.h for descriptor
#include "effects-simple.h" // for effect_to_string
#include "effects-convex.h" // for region

#include "step_common.h" // for STEP_PARALLEL, ...
#include "STEP_name.h" // for  STEP_PROD_NAME, STEP_MAX_NB_REQUEST_NAME, ...


/* debug macro
 */
extern int the_current_debug_level;
#define GEN_DEBUG(D, W, P) ifdebug(D) { pips_debug(D, "%s:\n", W); P;}
#define STEP_DEBUG_STATEMENT(D, W, S) GEN_DEBUG(D, W, print_statement(S))
#define STEP_DEBUG_DIRECTIVE(D, W, DRT) GEN_DEBUG(D, W, step_print_directive(DRT))

#define STEP_DEFAULT_RT_H "step/c"

/* STEP sentinelle */
#define STEP_SENTINELLE "STEP "

 // definit les valeurs possible pour la property STEP_DEFAULT_TRANSFORMATION utilisee en l'abscence de clause STEP
#define STEP_DEFAULT_TRANSFORMATION_OMP_TXT "OMP"
#define STEP_DEFAULT_TRANSFORMATION_HYBRID_TXT "HYBRID"
#define STEP_DEFAULT_TRANSFORMATION_MPI_TXT "MPI"

#define STEP_TRANSFORMATION_OMP    1
#define STEP_TRANSFORMATION_MPI    2
#define STEP_TRANSFORMATION_HYBRID 3
#define STEP_TRANSFORMATION_SEQ    4
