#include <stdio.h>

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "constants.h"

extern char *strndup();

/* a few simple functions to build or access internal representation */

/*value MakeValueLitteral()*/
/*{*/
/*    return(make_value(is_value_constant, */
/*		      make_constant(is_constant_litteral, NIL)));*/
/*}*/

/* FI: these functions were moved in ri-util/type.c and ri-util/entity.c */
