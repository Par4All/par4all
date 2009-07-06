/*
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

/** 
 * @file terapixify.c
 * apply transformations required to generate terapix microcode
 * @author Serge Guelton <serge.guelton@enst-bretagne.fr>
 * @date 2009-07-01
 */

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "ri-util.h"
#include "text.h"
#include "pipsdbm.h"
#include "resources.h"
#include "properties.h"
#include "misc.h"
#include "control.h"
#include "effects-generic.h"
#include "preprocessor.h"
#include "text-util.h"
#include "transformations.h"
#include "parser_private.h"
#include "syntax.h"
#include "c_syntax.h"

/** 
 * create a statement eligible for outlining into a kernel
 * #1 find the loop flagged with loop_label
 * #2 make sure we know thebound of the loop
 * #2' flag the kernel with a pragma
 * #3 call index set splitting on this loop to get a first loop with range_count multiple of KERNEL_NBNODES
 * #4 perform strip mining on this loop to make the kernel appear
 * #5 supress the generated loop and replace its index by the appropriate call to KERNEL_ID()
 * 
 * @param s statement where the kernel can be found
 * @param loop_label label of the loop to be turned into a kernel
 * 
 * @return true as long as the kernel is not found
 */
static
bool do_kernelize(statement s, entity loop_label) 
{
    if( same_entity_p(statement_label(s),loop_label) ||
            (statement_loop_p(s) && same_entity_p(loop_label(statement_loop(s)),loop_label)))
    {
        if( !instruction_loop_p(statement_instruction(s)) )
            pips_user_error("you choosed a label of a non-doloop statement");

        loop l = instruction_loop(statement_instruction(s));

        /* gather and check parameters */
        int nb_nodes = get_int_property("KERNEL_NBNODES");
        while(!nb_nodes)
        {
            string ur = user_request("number of nodes for your kernel?\n");
            nb_nodes=atoi(ur);
        }

        /* verify the loop is parallel */
        if( execution_sequential_p(loop_execution(l)) )
            pips_user_error("you tried to kernelize a sequential loop\n");

        /* perform index set splitting to get the good loop range */
        int count;
        if(!range_count(loop_range(l),&count))
            pips_user_error("unable to count the number of iterations in given loop\n");
        int increment_val;
        expression_integer_value(range_increment(loop_range(l)),&increment_val);
        int split_index = increment_val*(count - count%nb_nodes) ;
        entity split_index_entity = 
            make_C_or_Fortran_constant_entity(
                    itoa(split_index),
                    is_basic_int,
                    DEFAULT_INTEGER_TYPE_SIZE,
                    fortran_module_p(get_current_module_entity())
                    );
        index_set_split_loop(s,split_index_entity);
        /* now s is a block with two loops, we are interested in the first one */

        s= STATEMENT(CAR(statement_block(s)));

        /* we can strip mine the loop */
        loop_strip_mine(s,nb_nodes,-1);
        l = statement_loop(s);

        /* it's safe to skip the second level loop, because of the index set splitting */
        statement replaced_loop = loop_body(l);
        instruction erased_instruction = statement_instruction(replaced_loop);
        entity outermost_loop_index = loop_index(instruction_loop(erased_instruction));

        entity kernel_id = FindEntity(TOP_LEVEL_MODULE_NAME,"KERNEL_ID");
        if(entity_undefined_p(kernel_id))
            pips_user_error("you should define a 'KERNEL_ID' function\n");

        instruction assign = make_assign_instruction(make_expression_from_entity(outermost_loop_index),
                MakeBinaryCall(entity_intrinsic(c_module_p(get_current_module_entity())?PLUS_C_OPERATOR_NAME:PLUS_OPERATOR_NAME),
                    MakeNullaryCall(kernel_id),make_expression_from_entity(loop_index(l))
                )
                );
        statement_instruction(replaced_loop) = 
            make_instruction_block(
                    CONS(STATEMENT,make_stmt_of_instr(assign),
                        CONS(STATEMENT,loop_body(instruction_loop(erased_instruction)),NIL)));

        /* as the newgen free is recursive, we use a trick to prevent the recursion */
        loop_body(instruction_loop(erased_instruction)) = make_continue_statement(entity_empty_label());
        free_instruction(erased_instruction);

        /* flag the remaining loop to be proceed next*/
        extensions se = statement_extensions(s);
        extensions_extension(se)=CONS(EXTENSION,make_extension(make_pragma_string(strdup(OUTLINE_PRAGMA))),extensions_extension(se));

        /* job done */
        gen_recurse_stop(NULL);

    }
    return true;
}


/** 
 * turn a loop flagged with LOOP_LABEL into a kernel (GPU, terapix ...)
 * 
 * @param module_name name of the module
 * 
 * @return true
 */
bool kernelize(char * module_name)
{
    /* prelude */
    set_current_module_entity(module_name_to_entity( module_name ));
    set_current_module_statement((statement) db_get_memory_resource(DBR_CODE, module_name, TRUE) );

    /* rereive loop label */
    string loop_label_name = get_string_property_or_ask("LOOP_LABEL","label of the loop to turn into a kernel ?\n");
    entity loop_label_entity = find_label_entity(module_name,loop_label_name);
    if( entity_undefined_p(loop_label_entity) )
        pips_user_error("label '%s' not found in module '%s' \n",loop_label_name,module_name);


    /* run terapixify */
    gen_context_recurse(get_current_module_statement(),loop_label_entity,statement_domain,do_kernelize,gen_null);

    /* validate */
    module_reorder(get_current_module_statement());
    DB_PUT_MEMORY_RESOURCE(DBR_CODE, module_name,get_current_module_statement());

    /*postlude*/
    reset_current_module_entity();
    reset_current_module_statement();
    return true;
}


/** 
 * have a look to the pipsmake-rc description
 * basically call kernelize then outlining
 * 
 * @param module_name name of the module
 * 
 * @return true
 */
bool terapixify(char * module_name)
{
    /* everything is done by pispmake */
    return true;
}

