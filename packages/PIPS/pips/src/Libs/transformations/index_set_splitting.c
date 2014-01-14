/*

  $Id$

  Copyright 1989-2014 MINES ParisTech

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
#ifdef HAVE_CONFIG_H
    #include "pips_config.h"
#endif
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <errno.h>



#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "effects.h"
#include "text.h"
#include "database.h"

#include "misc.h"
#include "ri-util.h"
#include "effects-util.h"
#include "text-util.h"

#include "boolean.h"

#include "pipsdbm.h"
#include "resources.h"
#include "control.h"

#include "arithmetique.h"
#include "properties.h"

#include "transformations.h"

struct flfl {
    entity label;
    statement found;
};

static bool
find_loop_from_label_walker(statement s, struct flfl *p)
{
    instruction inst = statement_instruction (s);
    if(instruction_loop_p(inst)) {
	  entity do_lab_ent = loop_label(instruction_loop(inst));
	  entity stmt_lab_ent = statement_label(s);
	  if (gen_eq(p->label, do_lab_ent) || gen_eq(p->label, stmt_lab_ent) )
      {
          p->found=s;
          gen_recurse_stop(NULL);
      }
    }
    return true;
} 

statement
find_loop_from_label(statement s, entity label)
{
    struct flfl p = { label, statement_undefined };
    gen_context_recurse(s,&p,statement_domain,find_loop_from_label_walker,gen_null);
    return p.found;
} 

static void
index_set_split_loop(statement original_loop, entity new_loop_bound)
{
    pips_assert("index_set_split_loop called on a loop", statement_loop_p(original_loop));
    if(same_entity_p(new_loop_bound,loop_index(statement_loop(original_loop))))
    {
        pips_user_error("please set INDEX_SET_SPLITTING_BOUND property to an entity that is not the loop bound\n");
    }

    loop l = statement_loop(original_loop);
    /* split the loop */
    statement first_loop_statement = copy_statement(original_loop);


    clone_context cc = make_clone_context(
            get_current_module_entity(),
            get_current_module_entity(),
            NIL,
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

    intptr_t vincrement;
    if (!expression_integer_value(increment, &vincrement)) {
        pips_user_warning("unable to guess loop increment sign, assuming positive\n");
        vincrement = 1;
    }


    expression new_loop_bound_expression =
            binary_intrinsic_expression(vincrement > 0 ? MIN_OPERATOR_NAME : MAX_OPERATOR_NAME,
                                        entity_to_expression(new_loop_bound),
                                        copy_expression(range_upper(loop_range(l)))
                                        );
    expression new_loop_bound_expression_with_xcrement = 

       binary_intrinsic_expression(index_set_split_before_bound? MINUS_OPERATOR_NAME: PLUS_OPERATOR_NAME,
                        copy_expression(new_loop_bound_expression),
                        copy_expression(increment));

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
            make_sequence(make_statement_list(first_loop_statement,second_loop_statement))
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
    set_current_module_statement((statement) db_get_memory_resource(DBR_CODE, module_name, true) );

    /* get the loop */
    const char* loop_label = get_string_property("LOOP_LABEL");
    entity loop_label_entity = entity_undefined;
    if( string_undefined_p( loop_label ) || 
            entity_undefined_p((loop_label_entity=find_label_entity(module_name, loop_label))) )
        pips_user_error("please set LOOP_LABEL property to a valid label\n");

    statement loop_statement=find_loop_from_label(get_current_module_statement(),loop_label_entity);
    if(statement_undefined_p(loop_statement))
        pips_user_error("no statement with label %s found\n",loop_label);

    /* get the bound */
    const char* loop_bound = get_string_property("INDEX_SET_SPLITTING_BOUND");
    entity loop_bound_entity = entity_undefined;
    if( string_undefined_p( loop_bound ) )
        pips_user_error("please set INDEX_SET_SPLITTING_BOUND property to a known entity\n");
    else {
        loop_bound_entity = FindEntityFromUserName(module_name,loop_bound);
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

static bool try_loop_fusion(statement s0, statement s1) {
    loop l0 = statement_loop(s0),
         l1 = statement_loop(s1);
    statement base=s1;

    if(range_equal_p(loop_range(l0),loop_range(l1))) {
        hash_table renaming = hash_table_make(set_pointer,HASH_DEFAULT_SIZE);
        do {
            if(!same_entity_p(loop_index(l0), loop_index(l1)))
                hash_put(renaming,loop_index(l1),loop_index(l0));
            s0=loop_body(l0);
            s1=loop_body(l1);
        } while(statement_loop_p(s0)&&statement_loop_p(s1) && 
                range_equal_p(loop_range(l0=statement_loop(s0)),loop_range(l1=statement_loop(s1))) );
        replace_entities(base,renaming);
        hash_table_free(renaming);
        insert_statement(s0,s1,false);
        loop_body(l1)=statement_undefined;
        return true;
    }
    return false;
}

static void do_loop_fusion_walker(sequence seq, entity lbl) {
    for(list iter = sequence_statements(seq);!ENDP(iter);POP(iter)) {
        statement st = STATEMENT(CAR(iter));
        if(same_entity_p(statement_label(st),lbl)) {
            /* look for loop to merge */
            statement next = statement_undefined;
            if(!ENDP(CDR(iter))) {
                next = STATEMENT(CAR(CDR(iter)));
                /* verify it is a good candidate */
                if(statement_loop_p(next)) {
                    if(try_loop_fusion(st,next))
                        update_statement_instruction(next,make_continue_instruction());
                    else pips_user_error("loop fusion failed\n");
                }
                else pips_user_error("loop fusion failed\n");
            }
            else pips_user_error("loop fusion failed\n");
            gen_recurse_stop(0);
        }
    }
}

static void do_loop_fusion(entity lbl) {
    gen_context_recurse(get_current_module_statement(),lbl,sequence_domain,gen_true,do_loop_fusion_walker);
}

/* loop_fusion */
bool force_loop_fusion(char * module_name) {
    /* prelude */
    set_current_module_entity(module_name_to_entity( module_name ));
    set_current_module_statement((statement) db_get_memory_resource(DBR_CODE, module_name, true) );

    /* get the loop */
    const char* loop_label = get_string_property("LOOP_LABEL");
    entity loop_label_entity = entity_undefined;
    if( string_undefined_p( loop_label ) || 
            entity_undefined_p((loop_label_entity=find_label_entity(module_name, loop_label))) ) {
        pips_user_error("please set LOOP_LABEL property to a valid label, not '%s'\n",loop_label);
    }
    /* do the job */
    do_loop_fusion(loop_label_entity);

    /* validate */
    module_reorder(get_current_module_statement());
    DB_PUT_MEMORY_RESOURCE(DBR_CODE, module_name, get_current_module_statement());

    /*postlude*/
    reset_current_module_entity();
    reset_current_module_statement();
    return true;
}

