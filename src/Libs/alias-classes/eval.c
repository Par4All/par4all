#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "ri-util.h"
#include "misc.h"
#include "text-util.h"
#include "newgen_set.h"
#include "points_to_private.h"
#include "properties.h"
#include "preprocessor.h"

#include "effects-generic.h"
#include "effects-simple.h"
#include "effects-convex.h"
#include "alias-classes.h"





/*
  A subcase of eval_cell_with_points_to(). It takes a reference and a
  list of points-to and try to evaluate the reference. Each time we
  dereference we test if it's a constant one by exploiting the
  points-to information.

  we dereference and eval until either found a constant one so return a
  list of reference to wich points r or return anywhere:anywhere

  it is not clear if the decision to replace a dereferencement by a
  reference to anywhere should be taken here or not. I would rather
  postpone it as it depends on write effects unknown from this function.
*/
reference eval_reference_with_points_to(reference r, list pts_to)
{
/* iterer sur le path p[0][1][2][0] et tester chaque fois si on peut
 * dereferencer le pointeur*/
	return make_reference(entity_all_locations(),NIL);
}

/*
  input : cell c
          list of points_to ptl

  output : well, I do not know why it is a list... I allocate a new
  cell nc, which may be equal to the input c

  goal: see if cell c can be shortened by replacing its indirections
  by their values when they are defined in ptl. For instance, p[0][0]
  and (p,q,EXACT) is reduced to q[0]. And if (q,i,EXACT) is also
  available, the reference is reduced to i. The reduced cell is less likely
  to be invalidated by a write effect. The function is called "eval"
  because its goal is to build an as constant as possible reference or
  gap.

  For the time being this function is never called...

  It should be called by effect to see if a memory access path can be
  transformed into a constant, and by the points-to analysis to see if
  a sink or a source can be preserved in spite of write effects. This
  function should be called before points_to_filter_effects() to
  reduce the number of anywhere locations generated.
*/
cell eval_cell_with_points_to(cell c, list ptl)
{
  reference nr = reference_undefined; // new reference
  cell nc = cell_undefined;

  if(cell_reference_p(c)) {
    nr = eval_reference_with_points_to(cell_reference(c), ptl);
  }
  else if(cell_preference_p(c)) {
    reference r = preference_reference(cell_preference(c));
    nr = eval_reference_with_points_to(r, ptl);
  }
  else { /* Should be the gap case */
    pips_user_error("eval_preference_with_points_to() not implemented yet\n");
  }

  if(!reference_undefined_p(nr)) {
    nc = make_cell_reference(nr);
  }
  return nc;
}



