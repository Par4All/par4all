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

*/
list eval_reference_with_points_to(reference r, list pts_to)
{
/* iterer sur le path p[0][1][2][0] et tester chaque fois si on peut
 * dereferencer le pointeur*/
	return entity_all_locations();
}

/*
  input : cell




*/
list eval_cell_with_points_to(cell c, list pts_to)
{
  if(cell_reference_p(c))
	return eval_reference_with_points_to(cell_reference(c), pts_to);
  else
  {
	pips_user_error("eval_preference_with_points_to() not implemented yet\n");
	return NIL;
  }
}



