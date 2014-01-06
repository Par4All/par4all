
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
/* An atomizer that uses the one made by Fabien Coelho for HPFC,
 * and is in fact just a hacked version of the one made by Ronan
 * Keryell...
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "effects.h"

#include "dg.h"

typedef dg_arc_label arc_label;
typedef dg_vertex_label vertex_label;

#include "graph.h"
#include "ri-util.h"
#include "effects-util.h"
#include "text-util.h"
#include "database.h"
#include "misc.h"
#include "pipsdbm.h"
#include "resources.h"
#include "control.h"
#include "arithmetique.h"

#include "effects-generic.h"
#include "effects-simple.h"
#include "atomizer.h"


typedef struct {
    list written_references;
    list inserted_statements;
} atomizer_param;

static void atomize_call(expression ,atomizer_param* );
static void do_atomize_call(expression parent,atomizer_param* p,list expressions)
{
    bool safe=true;
    /* stop if call directly involves a written reference */
    FOREACH(EXPRESSION,arg,expressions)
    {
        if(expression_reference_p(arg)&&
                !entity_field_p(reference_variable(expression_reference(arg))))
        {
            list effects = proper_effects_of_expression(arg);
            FOREACH(REFERENCE,rw,p->written_references) {
                FOREACH(EFFECT,eff,effects)
                    if(references_may_conflict_p(rw,effect_any_reference(eff)))
                    { safe=false; break; }
                if(!safe) break;
            }
            gen_full_free_list(effects);
            if(!safe) break;
        }
    }
    /* go on and atomize, that is
     * - create a variable to store call result
     * - recurse on call arguments
     */
    if( safe )
    {
            basic bofe=basic_of_expression(parent);
            entity result = make_new_scalar_variable(get_current_module_entity(),bofe);
            AddEntityToCurrentModule(result);
            statement ass = make_assign_statement(
                    entity_to_expression(result),
                    make_expression(expression_syntax(parent),normalized_undefined));
            p->inserted_statements=CONS(STATEMENT,ass,p->inserted_statements);
            expression_syntax(parent)=syntax_undefined;
            update_expression_syntax(parent,make_syntax_reference(make_reference(result,NIL)));
    }
}

static void atomize_call(expression parent,atomizer_param* p)
{
    if(expression_call_p(parent))
    {
        call c = expression_call(parent);
        do_atomize_call(parent,p,call_arguments(c));
    }
    else if(expression_reference_p(parent))
    {
        reference ref = expression_reference(parent);
        if(!ENDP(reference_indices(ref)))
            do_atomize_call(parent,p,reference_indices(ref));
    }
}


static bool atomize_call_filter(expression parent,atomizer_param* p)
{
    if(expression_call_p(parent))
    {
        call c= expression_call(parent);
        entity op = call_function(c);
        /* do not visit rhs of . and -> */
        if(ENTITY_POINT_TO_P(op) || ENTITY_FIELD_P(op)) gen_recurse_stop(binary_call_rhs(c));
        if(call_constant_p(c)) return false;
    }
    return true;
}

static void atomize_all(void *v,atomizer_param* p)
{
    gen_context_multi_recurse(v,p,
            expression_domain,atomize_call_filter,atomize_call,
            0);
}

/* This function is called for all statements in the code
*/
static void atomize_statement(statement stat)
{
    instruction i = statement_instruction(stat);
    /* SG: we could atomize condition in test, loops etc too */
    if(instruction_call_p(i) || instruction_expression_p(i))
    {
        atomizer_param p = { NIL, NIL };
        list weffects = effects_write_effects(load_cumulated_rw_effects_list(stat));
        FOREACH(EFFECT,weff,weffects)
            p.written_references=CONS(REFERENCE,effect_any_reference(weff),p.written_references);
        /* this may cause a useless atomization if i is an expression */
        atomize_all(i,&p);
        if(!ENDP(p.inserted_statements))
            insert_statement(stat,make_block_statement(gen_nreverse(p.inserted_statements)),true);
    }

}

bool new_atomizer(char * mod_name)
{
    /* get the resources */
    set_current_module_entity(module_name_to_entity(mod_name));
    set_current_module_statement((statement)db_get_memory_resource(DBR_CODE, mod_name,true));
    set_cumulated_rw_effects((statement_effects)db_get_memory_resource(DBR_CUMULATED_EFFECTS, mod_name, true));
    debug_on("NEW_ATOMIZER_DEBUG_LEVEL");


    /* Now do the job */
    gen_recurse(get_current_module_statement(), statement_domain, gen_true, atomize_statement);

    /* Reorder the module, because new statements have been added */  
    clean_up_sequences(get_current_module_statement());
    module_reorder(get_current_module_statement());
    DB_PUT_MEMORY_RESOURCE(DBR_CODE, mod_name, get_current_module_statement());

    /* update/release resources */
    reset_cumulated_rw_effects();
    reset_current_module_statement();
    reset_current_module_entity();
    debug_off();

    return true;
}
