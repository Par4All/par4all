/*
 * $Id$
 *
 * Functions which should/could be in linear...
 */

#include <stdio.h>

#include "linear.h"
#include "genC.h"
#include "ri.h"
#include "ri-util.h"
#include "misc.h"

/* debug messages before calling the version in polyedre.
 */
Psysteme cute_convex_union(Psysteme s1, Psysteme s2)
{
  Psysteme s;

  ifdebug(9) {
    pips_debug(9, "IN\nS1 and S2:\n");
    sc_fprint(stderr, s1, entity_local_name);
    sc_fprint(stderr, s2, entity_local_name);
  }

  s = sc_cute_convex_hull(s1, s2);

  ifdebug(9) {
    pips_debug(9, "S =\n");
    sc_fprint(stderr, s, entity_local_name);
    pips_debug(9, "OUT\n");
  }

  return s;
}
