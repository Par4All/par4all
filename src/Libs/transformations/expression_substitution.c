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

#include <stdlib.h>
#include <stdio.h>

#include "genC.h"

#include "linear.h"
#include "ri.h"
#include "database.h"

#include "ri-util.h"
#include "pipsdbm.h"

#include "misc.h"

#include "resources.h"
#include "properties.h"

static expression pattern = expression_undefined;
static string pattern_module_name = string_undefined;

static
bool set_pattern()
{
    pattern_module_name = get_string_property("EXPRESSION_SUBSTITUTION_PATTERN");
    if( ! string_undefined_p(pattern_module_name) )
    {

        statement s = (statement) db_get_memory_resource(DBR_CODE, pattern_module_name, TRUE);
        instruction i = statement_instruction(s);
        if( instruction_return_p(i) )
        {
            pattern = instruction_return(i);
        }
        else if( return_instruction_p(i) )
        {
            pattern = EXPRESSION(CAR(call_arguments(instruction_call(i))));
        }
        else {
            pips_user_warning("%s used as pattern for expression substitution, but is not a module with a single return !\n", pattern_module_name);
            return false;
        }

    }
    else
    {
        pips_user_warning("EXPRESSION_SUBSTITUTION_PATTERN undefined, but needed to perform expression substitution\n");
        return false;
    }
}

static 
void expression_similar_to_pattern(expression e)
{
    hash_table symbols;
    if(expression_similar_get_context_p(e,pattern,&symbols))
    {
        entity pattern_entity = global_name_to_entity(TOP_LEVEL_MODULE_NAME,pattern_module_name);
        expression_normalized(e) = normalized_undefined;

        /* recover pattern's arguments */
        list iter = code_declarations(value_code(entity_initial(pattern_entity)));
        while( !ENTITY_NAME_P( ENTITY(CAR(iter)), DYNAMIC_AREA_LOCAL_NAME ) ) POP(iter);
            POP(iter);/*pop the dynamic area label*/
        if( !ENDP(iter) && ENTITY_NAME_P( ENTITY(CAR(iter)), entity_user_name(pattern_entity) ) )
            POP(iter); /* pop the first flag if needed */

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
    }
}

bool expression_substitution(string module_name)
{
    /* prelude */
    set_current_module_entity(module_name_to_entity( module_name ));
    set_current_module_statement((statement) db_get_memory_resource(DBR_CODE, module_name, TRUE) );

    /* search pattern*/
    bool pattern_set_p = set_pattern();
    if( pattern_set_p )
    {
        gen_recurse(
            get_current_module_statement(),
            expression_domain,
            gen_true,
            &expression_similar_to_pattern
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
