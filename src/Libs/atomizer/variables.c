/* -- variables.c
 *
 * package atomizer :  Alexis Platonoff, juillet 91
 * --
 *
 * This file contains functions that creates the new temporary entities and
 * the new auxiliary entities, and functions that compute the "basic" of an
 * expression.
 */

#include <stdio.h>
#include <string.h>

#include "genC.h"
#include "ri.h"
#include "graph.h"
#include "dg.h"

#include "ri-util.h"
/* #include "constants.h" */
#include "misc.h"

/* AP: I removed this include because it no longer exists
#include "loop_normalize.h" */

#include "atomizer.h"

/* FI: I moved these procedures in ri-util/variable.c and ri-util/type.c */
