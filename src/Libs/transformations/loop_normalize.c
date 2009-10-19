/*

  $Id$

  Copyright 1989-2009 MINES ParisTech

  This file is part of PIPS.

  PIPS is free software: you can redistribute it and/or modify it
  under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  any later version.

  PIPS is distributed in the hope that it will be useful, but WITHOUT ANY
  WARRANTY; without even the implied warranty of MERCHANTABILITY or
  FITNESS FOR A PARTICULAR PURPOSE.

  See the GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with PIPS.  If not, see <http://www.gnu.org/licenses/>.

*/
/* Name     :   loop_normalize.c
 * Package  : 	loop_normalize
 * Author   :   Arnauld LESERVOT & Alexis PLATONOFF
 * Date     :	27 04 93
 * Modified :   moved to Lib/transformations, AP, sep 95
 *      strongly modernized by SG
 * Documents:	"Implementation du Data Flow Graph dans Pips"
 * Comments :
 *
 * Functions of normalization of DO loops. Normalization consists in changing
 * the loop index so as to have something like:
 *      DO I = 0, UPPER, 1
 *
 * If the old DO loops was:
 *      DO I = lower, upper, incre
 * then : UPPER = (upper - lower + incre)/incre - 1
 *
 * The normalization is done only if "incre" is a constant number.
 * The normalization produces two statements. One assignment of the old
 * loop index (in the exemple I) to its value function of the new index;
 * the formula is: I = incre*NLC + lower
 * and one assignment of the old index at the end of the loop for its final
 * value: I = incre * MAX(UPPER+1, 0) + lower
 *
 * So, for exemple:
 *      DO I = 2, 10, 4
 *        INST
 *      ENDDO
 * is normalized in:
 *      DO I = 0, 2, 1
 *        I = 4*I + 2
 *        INST
 *      ENDDO
 *      I = 14
 *
 * Or:
 *      DO I = 2, 1, 4
 *        INST
 *      ENDDO
 * is normalized in:
 *      DO I = 0, -1, 1
 *        I = 4*I + 2
 *        INST
 *      ENDDO
 *      I = 2
 *
 * SG: normalized loop used to have a new loop counter.
 * It made code less reeadable, so I supressed this
 *
 * If a loop has a label, it is removed. For example:
 *      DO 10 I = 1, 10, 1
 *        INST
 * 10   CONTINUE
 *
 * is modified in:
 *      DO i = 1, 10, 1
 *        INST
 *      ENDDO
 */

/* Ansi includes	*/
#include <stdio.h>
#include <string.h>

/* Newgen includes	*/
#include "genC.h"

#include "boolean.h"
#include "arithmetique.h"
#include "vecteur.h"
#include "contrainte.h"
#include "ray_dte.h"
#include "sommet.h"
#include "sg.h"
#include "sc.h"
#include "polyedre.h"
#include "matrix.h"

/* Pips includes	*/
#include "linear.h"
#include "ri.h"

#include "database.h"
#include "ri-util.h"
#include "constants.h"
#include "misc.h"
#include "control.h"
#include "text-util.h"
#include "pipsdbm.h"
#include "resources.h"
#include "transformations.h"
#include "properties.h"


/**
 * statement normalization
 * normalize a statement if it's a loop
 *
 * @param s statement to normalize
 *
 */
static
void loop_normalize_statement(statement s) {
  if( statement_loop_p(s) ) {
    // OK, it's a loop, go on
    loop l = statement_loop(s);
    pips_debug(4, "begin loop\n");
    if (! constant_step_loop_p(l))
      // Cannot normalize loops with non-constant increment
      return;

    /* Do not normalize normal loops (loops with a 1-increment), except if
       we ask for: */
    if (normal_loop_p(l) && !get_bool_property("LOOP_NORMALIZE_ONE_INCREMENT"))
      return;

    // Get the new lower bound of the loop:
    int new_lb = get_int_property("LOOP_NORMALIZE_LOWER_BOUND");

    entity index = loop_index(l);
    range lr = loop_range(l);
    // Initial loop range: rl:ru:ri
    expression rl = range_lower(lr);
    expression ru = range_upper(lr);
    expression ri = range_increment(lr);

    // Number of iteration: nub = ((ru-rl)+ri)/ri
    /* Note that in the following, make_op_exp do make some partial eval
       for integer values! */
    expression nub = make_op_exp(DIVIDE_OPERATOR_NAME,
				 make_op_exp(PLUS_OPERATOR_NAME,
					     make_op_exp(MINUS_OPERATOR_NAME,
							 ru,rl),
					     ri),
				 copy_expression(ri));
    expression nub2 = copy_expression(nub);

    expression nlc_exp = entity_to_expression(index);
    // New range new_lb:(nub2+(new_lb-1)):1
    range_lower(lr) = make_integer_constant_expression(new_lb);
    range_upper(lr) =
      make_op_exp(PLUS_OPERATOR_NAME,
		  nub2,
		  make_op_exp(MINUS_OPERATOR_NAME,
			      make_integer_constant_expression(new_lb),
			      make_integer_constant_expression(1)));
    range_increment(lr) = make_integer_constant_expression(1);

    // Base change: (index - new_lb)*ri  + rl
    expression new_index_exp =
      make_op_exp(PLUS_OPERATOR_NAME,
		  copy_expression(rl),
		  make_op_exp(MULTIPLY_OPERATOR_NAME,
			      make_op_exp(MINUS_OPERATOR_NAME,
					  nlc_exp,
					  make_integer_constant_expression(new_lb)),
			      copy_expression(ri)));


    /* Commit the changes */

    /* Replace all references to index in loop_body(l) by new_index_exp */
    replace_entity_by_expression(loop_body(l),index,new_index_exp);

    if (!get_bool_property("LOOP_NORMALIZE_SKIP_INDEX_SIDE_EFFECT")) {
      /* We want to compute the real side effect of the loop on its index:
      its final value after the loop. */
      expression nub3 = copy_expression(nub);

      /* Compute the final value of the loop index to have the correct side
	 effect of the loop on the original index: */
      expression exp_max = expression_undefined, exp_plus = expression_undefined;
      if ( expression_constant_p(nub3)) {
	int upper = expression_to_int(nub3);
	if (upper > 0)
	  exp_max = make_integer_constant_expression(upper);
      }
      else {
	entity max_ent = FindEntity(TOP_LEVEL_MODULE_NAME,MAX_OPERATOR_NAME);
	exp_max = make_max_exp(max_ent, copy_expression(nub),
			       make_integer_constant_expression(0));
      }
      if (expression_undefined_p(exp_max))
	exp_plus = copy_expression(rl);
      else
	exp_plus = make_op_exp(PLUS_OPERATOR_NAME,
			       make_op_exp(MULTIPLY_OPERATOR_NAME,
					   copy_expression(ri),
					   exp_max),
			       copy_expression(rl));

      expression index_exp = entity_to_expression(index);
      /* Add after the loop the initialization of the loop index to the
	 final value: */
      statement end_stmt = make_assign_statement(copy_expression(index_exp),
						 exp_plus);
      append_a_statement(s, end_stmt);
    }
    pips_debug( 4, "end LOOP\n");
  }
}


/** Apply the loop normalization upon a module

    Try to normalize each loop into a loop with a 1-step increment

    @param mod_name name of the normalized module

    @return true
*/
bool loop_normalize(char *mod_name)
{
    /* prelude */
    debug_on("LOOP_NORMALIZE_DEBUG_LEVEL");
    pips_debug(1, "\n\n *** LOOP_NORMALIZE for %s\n", mod_name);

    set_current_module_entity(module_name_to_entity(mod_name));
    set_current_module_statement((statement) db_get_memory_resource(DBR_CODE, mod_name, TRUE));


    /* Compute the loops normalization of the module. */
    gen_recurse(get_current_module_statement(),statement_domain,gen_true,loop_normalize_statement);

    /* commit changes */
    module_reorder(get_current_module_statement()); ///< we may have had statements
    DB_PUT_MEMORY_RESOURCE(DBR_CODE, mod_name, get_current_module_statement());

    /* postlude */
    pips_debug(1, "\n\n *** LOOP_NORMALIZE done\n");
    debug_off();
    reset_current_module_entity();
    reset_current_module_statement();

    return true;
}

