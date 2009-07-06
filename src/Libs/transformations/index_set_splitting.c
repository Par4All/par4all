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
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <errno.h>



#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "text.h"
#include "database.h"

#include "misc.h"
#include "ri-util.h"
#include "text-util.h"

#include "boolean.h"

#include "pipsdbm.h"
#include "resources.h"
#include "control.h"

#include "arithmetique.h"
#include "properties.h"
#include "preprocessor.h"

#include "transformations.h"

static statement loop_statement = statement_undefined;

static bool
find_loop_from_label(statement s, entity label)
{
    instruction inst = statement_instruction (s);
    if(instruction_loop_p(inst)) {
	  entity do_lab_ent = loop_label(instruction_loop(inst));
	  entity stmt_lab_ent = statement_label(s);
	  if (gen_eq(label, do_lab_ent) || gen_eq(label, stmt_lab_ent) )
      {
          loop_statement=s;
          return false;
      }
    }
    return true;
} 

void
index_set_split_loop(statement original_loop, entity new_loop_bound)
{
    pips_assert("index_set_split_loop called on a loop", statement_loop_p(original_loop));
    if(same_entity_p(new_loop_bound,loop_index(statement_loop(original_loop))))
    {
        pips_user_error("please set INDEX_SET_SPLITTING_BOUND property to an entity that is not the loop bound\n");
    }

    /* split the loop */
    statement first_loop_statement = copy_statement(original_loop);


    clone_context cc = make_clone_context(
            get_current_module_entity(),
            get_current_module_entity(),
            get_current_module_statement() );

    statement second_loop_statement = clone_statement(original_loop,cc);

    free_clone_context(cc);

    /* clone statement may have had declarations to the statement
     * has a consequence the return statement may not be a for loop, but a sequence
     */
    statement the_second_loop_statement = second_loop_statement;
    if( ! statement_loop_p(the_second_loop_statement) )
    {
        FOREACH(STATEMENT,s,
                sequence_statements(instruction_sequence(statement_instruction(the_second_loop_statement))))
            the_second_loop_statement=s;
        pips_assert("last statement of cloned sequence is a loop",statement_loop_p(the_second_loop_statement));
    }

    /* fix the bound */
    bool index_set_split_before_bound = get_bool_property("INDEX_SET_SPLITTING_SPLIT_BEFORE_BOUND");
    expression increment = range_increment(loop_range(statement_loop(the_second_loop_statement)));
    expression new_loop_bound_expression = make_expression_from_entity(new_loop_bound);
    expression new_loop_bound_expression_with_xcrement = 
        make_expression(
                make_syntax_call(
                    make_call(
                        CreateIntrinsic(
                            index_set_split_before_bound?
                            MINUS_OPERATOR_NAME:
                            PLUS_OPERATOR_NAME),
                        CONS(
                            EXPRESSION,
                            make_expression_from_entity(new_loop_bound),
                            CONS(
                                EXPRESSION,
                                copy_expression(increment),
                                NIL
                                )
                            )
                        )
                    ),
                normalized_undefined
                );

    expression fst_loop_upper = index_set_split_before_bound ?
        new_loop_bound_expression_with_xcrement:
        new_loop_bound_expression;

    expression snd_loop_lower = index_set_split_before_bound ?
        new_loop_bound_expression:
        new_loop_bound_expression_with_xcrement;


    range_upper(loop_range(statement_loop(first_loop_statement)))=
        fst_loop_upper;
    range_lower(loop_range(statement_loop(the_second_loop_statement)))=
        snd_loop_lower;

    /* put loops together */
    instruction new_instruction = make_instruction_sequence(
            make_sequence(CONS(STATEMENT,first_loop_statement, CONS(STATEMENT,second_loop_statement,NIL)))
            );
    statement_label(original_loop)=entity_empty_label();
    statement_number(original_loop)=STATEMENT_NUMBER_UNDEFINED;
    statement_ordering(original_loop)=STATEMENT_ORDERING_UNDEFINED;
    statement_comments(original_loop)=empty_comments;
    statement_instruction(original_loop)= new_instruction;
    statement_declarations(original_loop)=NIL;
    statement_decls_text(original_loop)=NULL;

}

bool index_set_splitting(char* module_name)
{
    /* prelude */
    set_current_module_entity(module_name_to_entity( module_name ));
    set_current_module_statement((statement) db_get_memory_resource(DBR_CODE, module_name, TRUE) );

    /* get the loop */
    string loop_label = get_string_property("LOOP_LABEL");
    entity loop_label_entity = entity_undefined;
    if( string_undefined_p( loop_label ) || 
            entity_undefined_p((loop_label_entity=find_label_entity(module_name, loop_label))) )
        pips_user_error("please set LOOP_LABEL property to a valid label\n");

    loop_statement=statement_undefined;
    gen_context_recurse(get_current_module_statement(), loop_label_entity, statement_domain, find_loop_from_label, gen_null);
    if(statement_undefined_p(loop_statement))
        pips_user_error("no statement with label %s found\n",loop_label);

    /* get the bound */
    string loop_bound = get_string_property("INDEX_SET_SPLITTING_BOUND");
    entity loop_bound_entity = entity_undefined;
    if( string_undefined_p( loop_bound ) )
        pips_user_error("please set INDEX_SET_SPLITTING_BOUND property to a known entity\n");
    else {
        loop_bound_entity = FindEntity(module_name,loop_bound);
        if(entity_undefined_p(loop_bound_entity)) // maybe its a constant
        {
            int integer ;
            if( sscanf(loop_bound,"%d",&integer)==1 )
            {
                loop_bound_entity = 
                    make_C_or_Fortran_constant_entity(
                            loop_bound,
                            is_basic_int,
                            DEFAULT_INTEGER_TYPE_SIZE,
                            fortran_module_p(get_current_module_entity())
                    );
            }
            else
            {
                pips_user_error("please set INDEX_SET_SPLITTING_BOUND property to a known entity\n");
            }
        }
    }


    /* perform substitution */
    if(statement_loop_p(loop_statement))
        index_set_split_loop(loop_statement,loop_bound_entity);

    /* validate */
    module_reorder(get_current_module_statement());
    DB_PUT_MEMORY_RESOURCE(DBR_CODE, module_name, get_current_module_statement());

    /*postlude*/
    reset_current_module_entity();
    reset_current_module_statement();
    return true;
}


