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


/**
 * statement normalization
 * normalize a statement if it's a loop
 *
 * @param s statement to normalize
 *
 */
static
void loop_normalize_statement(statement s)
{
    if( statement_loop_p(s) )
    {
        loop l = statement_loop(s);
        pips_debug(4, "begin loop\n");
        if (constant_step_loop_p(l) && ! normal_loop_p(l))
        {
            entity index = loop_index( l );
            range lr = loop_range( l );
            expression rl = range_lower( lr );
            expression ru = range_upper( lr );
            expression ri = range_increment( lr );

            expression nub =   make_op_exp(DIVIDE_OPERATOR_NAME,
                    make_op_exp(PLUS_OPERATOR_NAME,
                        make_op_exp(MINUS_OPERATOR_NAME,ru,rl),
                        ri),
                    copy_expression(ri));
            expression nub2 = copy_expression(nub);

            expression nlc_exp = entity_to_expression(index);
            range_lower( lr ) = make_integer_constant_expression( 1 );
            range_upper( lr ) = nub2;
            range_increment( lr ) = make_integer_constant_expression( 1 );

            expression new_index_exp = make_op_exp(PLUS_OPERATOR_NAME,
                    make_op_exp(MINUS_OPERATOR_NAME,
                        make_op_exp(MULTIPLY_OPERATOR_NAME,
                            copy_expression(ri),
                            nlc_exp),
                        copy_expression(ri)),
                    copy_expression(rl));


            expression nub3 = copy_expression( nub );
            expression exp_max = expression_undefined, exp_plus = expression_undefined;
            if ( expression_constant_p( nub3 )) {
                int upper = expression_to_int( nub3 );
                if ( upper > 0 )
                    exp_max = make_integer_constant_expression( upper );
            }
            else {
                entity max_ent = FindEntity(TOP_LEVEL_MODULE_NAME,MAX_OPERATOR_NAME);
                exp_max = make_max_exp(max_ent, copy_expression( nub ),
                        make_integer_constant_expression( 0 ));
            }
            if ( expression_undefined_p(exp_max) )
                exp_plus = copy_expression( rl );
            else
                exp_plus = make_op_exp(PLUS_OPERATOR_NAME,
                        make_op_exp( MULTIPLY_OPERATOR_NAME,
                            copy_expression( ri ),
                            exp_max),
                        copy_expression( rl ));
            expression index_exp = entity_to_expression( index );
            statement end_stmt = make_assign_statement( copy_expression(index_exp), exp_plus );

            /* commit the changes */

            /* #0 replace all references to index in loop_body(l) by new_index_exp */
            replace_entity_by_expression(loop_body(l),index,new_index_exp);

            /* #1 copy s into a new statement */
            statement new_statement = make_empty_statement();
            statement_label(new_statement)=statement_label(s);
            statement_comments(new_statement)=statement_comments(s);
            statement_declarations(new_statement)=statement_declarations(s);
            statement_extensions(new_statement)=statement_extensions(s);
            statement_decls_text(new_statement)=statement_decls_text(s);
            statement_instruction(new_statement)=statement_instruction(s);
            /* #2 reset the non relevant fields of new_statement */
            statement_label(s)=entity_empty_label();
            statement_number(s)=STATEMENT_NUMBER_UNDEFINED;
            statement_ordering(s)=STATEMENT_ORDERING_UNDEFINED;
            statement_comments(s)=empty_comments;
            statement_declarations(s)=NIL;
            statement_extensions(s)=empty_extensions();
            statement_decls_text(s)=NULL;
            /* #3 make s a block instead of a loop */
            statement_instruction(s) = make_instruction_block(make_statement_list(new_statement,end_stmt));

            debug( 4, __FUNCTION__, "end LOOP\n");
        }
    }
}


/** 
 * Apply the loop normalization upon a module
 * 
 * @param mod_name name of the normalized module
 * 
 * @return true
 */
bool loop_normalize(char *mod_name)
{
    /* prelude */
    debug_on("LOOP_NORMALIZE_DEBUG_LEVEL");
    if (get_debug_level() > 1) user_log("\n\n *** LOOP_NORMALIZE for %s\n", mod_name);

    set_current_module_entity(module_name_to_entity(mod_name));
    set_current_module_statement((statement) db_get_memory_resource(DBR_CODE, mod_name, TRUE));


    /* Compute the loops normalization of the module. */
    gen_recurse(get_current_module_statement(),statement_domain,gen_true,loop_normalize_statement);

    /* commit changes */
    module_reorder(get_current_module_statement()); ///< we may have had statements
    DB_PUT_MEMORY_RESOURCE(DBR_CODE, mod_name, get_current_module_statement());

    /* postlude */
    if (get_debug_level() > 1) user_log("\n\n *** LOOP_NORMALIZE done\n");
    debug_off();
    reset_current_module_entity();
    reset_current_module_statement();

    return true;
}

