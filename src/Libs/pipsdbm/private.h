/*
 * $Id$
 *
 * stuff to include, local to pipsdbm.
 * SHOULD NOT include ri-util.h.
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>

typedef void * db_void;

#include "linear.h"

#include "resources.h"

#include "genC.h"
#include "ri.h"

#include "misc.h"
#include "properties.h"

#include "ri-util.h" /* needed for statement_mapping */

#include "database.h" /* for obsolete functions... */
#include "pipsdbm.h"
