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

#include <stdlib.h>
#include <stdio.h>

#include "genC.h"

#include "linear.h"
#include "ri.h"
#include "effects.h"
#include "database.h"

#include "ri-util.h"
#include "effects-util.h"
#include "pipsdbm.h"
#include "control.h"
#include "callgraph.h"

#include "misc.h"

#include "resources.h"
#include "properties.h"
#include "expressions.h"

static expression pattern = expression_undefined;
static const char* pattern_module_name = string_undefined;

/* retrieves the expression used as a pattern based on
 * the one - statement function given in a property
 * and sets static variables accordingly
 */
static
bool set_pattern()
{
    pattern_module_name = get_string_property("EXPRESSION_SUBSTITUTION_PATTERN");
    if( ! string_undefined_p(pattern_module_name) )
    {

        statement s = (statement) db_get_memory_resource(DBR_CODE, pattern_module_name, true);
        instruction i = statement_instruction(s);
        if( instruction_block_p(i))
        {
            list l = instruction_block(i);
            if(!ENDP(l)) i = statement_instruction(STATEMENT(CAR(l)));
        }
        if( return_instruction_p(i) )
            pattern = EXPRESSION(CAR(call_arguments(instruction_call(i))));
        else {
            pips_user_warning("%s used as pattern for expression substitution, but is not a module with a single return !\n", pattern_module_name);
            return false;
        }
        convert_to_standard_operators(pattern);

    }
    else
    {
        pips_user_warning("EXPRESSION_SUBSTITUTION_PATTERN undefined, but needed to perform expression substitution\n");
        return false;
    }
	return true;
}

static void substitute_expression_walker(expression e, expression * pi )
{
    if(same_expression_p(e,pi[0])) 
        update_expression_syntax(e,copy_syntax(expression_syntax(pi[1])));
}

void substitute_expression(void *in, expression pattern, expression into)
{
    expression arge [] = { pattern, into };
    gen_context_recurse(in,&arge[0],expression_domain,gen_true, substitute_expression_walker);
}

static 
bool replace_expression_similar_to_pattern(expression e)
{
    hash_table symbols; // contains the symbols gathered during the matching
    // match e against pattern and stocks symbols in hash_table
    if(expression_similar_get_context_p(e,pattern,&symbols)) 
    {
        entity pattern_entity = FindEntity(TOP_LEVEL_MODULE_NAME,pattern_module_name);
        expression_normalized(e) = normalized_undefined;

        /* recover pattern's arguments */
        list iter = NIL;
        FOREACH(ENTITY,e,code_declarations(value_code(entity_initial(pattern_entity))))
            if(entity_formal_p(e))
                iter=CONS(ENTITY,e,iter);
        iter=gen_nreverse(iter);

        /* fill the arguments */
        list args = NIL;
        FOREACH(ENTITY, arg, iter)
        {
            expression arge = hash_get(symbols,entity_name(arg));
            pips_assert("created map consistant with dynamic decl",arge != HASH_UNDEFINED_VALUE);
            args=CONS(EXPRESSION,copy_expression(arge),args);

        }

        free_syntax(expression_syntax(e));

        /* fix the expression field*/
        expression_syntax(e)=make_syntax_call(
                make_call(pattern_entity,gen_nreverse(args))
        );
        hash_table_free(symbols);
        return false;
    }
    return true;
}
static
bool replace_instruction_similar_to_pattern(instruction i)
{
    if(instruction_call_p(i))
    {
        expression exp = call_to_expression(instruction_call(i));
        if( !replace_expression_similar_to_pattern(exp) ) /* replacement successfull */
        {
            instruction_call(i)=expression_call(exp);
            expression_syntax(exp)=syntax_undefined;
            free_expression(exp);
            return false;
        }
    }
    return true;
}

/* simple pass that performs substitution of expression by module call
 */
bool expression_substitution(const char* module_name)
{
    /* prelude */
    set_current_module_entity(module_name_to_entity( module_name ));
    set_current_module_statement((statement) db_get_memory_resource(DBR_CODE, module_name, true) );

    /* search pattern*/
    bool pattern_set_p = set_pattern();
    if( pattern_set_p )
    {
        convert_to_standard_operators(get_current_module_statement());
        gen_multi_recurse(
            get_current_module_statement(),
            reference_domain, gen_false, gen_null,
            range_domain, gen_false, gen_null,
            expression_domain, replace_expression_similar_to_pattern, gen_null,
            instruction_domain, replace_instruction_similar_to_pattern, gen_null,
            0
        );
    }

    /* validate */
    module_reorder(get_current_module_statement());
    DB_PUT_MEMORY_RESOURCE(DBR_CODE, module_name, get_current_module_statement());
    DB_PUT_MEMORY_RESOURCE(DBR_CALLEES, module_name, compute_callees(get_current_module_statement()));

    /*postlude*/
    reset_current_module_entity();
    reset_current_module_statement();
    return pattern_set_p;
}
