/*
  Copyright 1989-2010 MINES ParisTech

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
 * @file insert_statement.c
 * check if a statement can be inserted without too much side effect
 * @author Serge Guelton <serge.guelton@enst-bretagne.fr>
 * @date 2010-06-01
 */
#ifdef HAVE_CONFIG_H
    #include "pips_config.h"
#endif
#include <ctype.h>


#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "ri-util.h"
#include "text.h"
#include "pipsdbm.h"
#include "resources.h"
#include "properties.h"
#include "misc.h"
#include "conversion.h"
#include "control.h"
#include "effects-generic.h"
#include "effects-simple.h"
#include "effects-convex.h"
#include "text-util.h"
#include "parser_private.h"
#include "accel-util.h"
/* generate a pcontrainte corresponding to dimensions, with a preset list of phis
 * ex: int a[10][n]; and (phi0,phi1)
 * will result in 0<=phi0<=9 and 0<=phi1<=n-1
 */
static Pcontrainte dimensions_to_psysteme(list dims,list phis)
{
    pips_assert("as many dimensions as phis",gen_length(dims) == gen_length(phis));
    Pcontrainte pc = NULL;
    FOREACH(DIMENSION,dim,dims)
    {
        entity phi = ENTITY(CAR(phis));
        expression lower = dimension_lower(dim);
        expression upper = dimension_upper(dim);
        NORMALIZE_EXPRESSION(lower);
        NORMALIZE_EXPRESSION(upper);

        if(normalized_linear_p(expression_normalized(lower)) && normalized_linear_p(expression_normalized(upper)))
        {

            Pvecteur vlower = vect_dup(normalized_linear(expression_normalized(lower)));
            Pvecteur vupper = vect_dup(normalized_linear(expression_normalized(upper)));
            vect_add_elem(&vlower,phi,1);
            vect_add_elem(&vupper,phi,-1);
            vect_chg_sgn(vlower);
            vect_chg_sgn(vupper);
            pc=contrainte_append(pc,contrainte_make(vlower));
            pc=contrainte_append(pc,contrainte_make(vupper));
        }
        else {
            contrainte_rm(pc);
            return NULL;
        }
        POP(phis);
    }
    return pc;
}
typedef struct {
    entity to;
    list found;
} entity_list_pair;

static void find_calls_to_function_walker(call c, entity_list_pair *p)
{
    if(same_entity_p(p->to,call_function(c)))
        p->found=CONS(CALL,c,p->found);
}

static void find_calls_to_function_walker_in_declaration(statement s, entity_list_pair *p)
{
    FOREACH(ENTITY,e,statement_declarations(s))
        if(value_expression_p(entity_initial(e)))
            gen_context_recurse(s,&p,call_domain,gen_true,find_calls_to_function_walker);
}

/* returns a list of call to @p to found in @p in*/
static list find_calls_to_function(statement in,entity to) {
    entity_list_pair p = { to , NIL };
    gen_context_multi_recurse(in,&p,call_domain,gen_true,find_calls_to_function_walker,
            statement_domain,gen_true,find_calls_to_function_walker_in_declaration,0);
    return p.found;
}


/* tries hard to propagate entity dimension change */
static void statement_insertion_fix_access_in_callers(const char * module_name, entity new_formal)
{
    callees callers = (callees)db_get_memory_resource(DBR_CALLERS, module_name,true);
    intptr_t new_formal_offset=formal_offset(storage_formal(entity_storage(new_formal)));
    size_t new_formal_dimensions=gen_length(variable_dimensions(type_variable(ultimate_type(entity_type(new_formal)))));
    FOREACH(STRING,caller_name,callees_callees(callers)) {
        statement caller_statement = (statement)db_get_memory_resource(DBR_CODE,caller_name,true);
        list calls = find_calls_to_function(caller_statement,module_name_to_entity(module_name));
        FOREACH(CALL,c,calls) {
            expression nth = EXPRESSION(gen_nth(new_formal_offset-1,call_arguments(c)));
            if(expression_reference_p(nth))
            {
                reference r = expression_reference(nth);
                size_t nb_indices = gen_length(reference_indices(r));
                size_t nb_dimensions = gen_length(variable_dimensions(type_variable(ultimate_type(entity_type(reference_variable(r))))));
                if(nb_dimensions - nb_indices == new_formal_dimensions)
                {
                    list *iter = &variable_dimensions(type_variable(ultimate_type(entity_type(reference_variable(r)))));
                    for(size_t i=nb_indices;i;i--) POP(*iter);
                    gen_full_free_list(*iter);
                    *iter=gen_full_copy_list(variable_dimensions(type_variable(ultimate_type(entity_type(new_formal)))));
                    statement_insertion_fix_access_in_callers(caller_name,reference_variable(r));
                }
                else pips_internal_error("unhandled case");
            }
            else pips_internal_error("unhandled case");
        }
        DB_PUT_MEMORY_RESOURCE(DBR_CODE,caller_name,caller_statement);
        gen_free_list(calls);
    }
}

/* fixes statement declaration depending on region access */
static void statement_insertion_fix_access(list regions)
{
    FOREACH(REGION,reg,regions)
    {
        reference r = region_any_reference(reg);
        entity e = reference_variable(r);
        if(formal_parameter_p(e)) {
            pips_user_warning("cannot change formal parameter with this version\n"
                    "try using inlining if possible\n");
            break;
        }
        list phis = expressions_to_entities(reference_indices(r));
        Pcontrainte dims_sc = dimensions_to_psysteme(variable_dimensions(type_variable(entity_type(e))), phis);
        Psysteme access_syst = region_system(reg);

        volatile Psysteme stmp ,sr;
        CATCH(overflow_error)
        {
            pips_debug(1, "overflow error\n");
            return ;
        }
        TRY
        {
            stmp = sc_make(contrainte_new(),CONTRAINTE_UNDEFINED);
            for(Pcontrainte iter=dims_sc;!CONTRAINTE_UNDEFINED_P(iter);iter=contrainte_succ(iter)) {
                Pcontrainte toadd = contrainte_make(vect_dup(contrainte_vecteur(iter)));
                sc_add_inegalite(stmp,toadd);// there should not be any basis issue, they share the same ... */
            }
            sr = sc_cute_convex_hull(access_syst, stmp);
            sc_rm(stmp);
            contrainte_rm(dims_sc);
            sc_nredund(&sr);
            UNCATCH(overflow_error);
        }
        /* if we reach this point, we are ready for backward translation from vecteur to dimensions :) */
        list new_dimensions = NIL;
        FOREACH(ENTITY,phi,phis)
        {
            Pcontrainte lower,upper;
            constraints_for_bounds(phi, &sc_inegalites(sr), &lower, &upper);
            if( !CONTRAINTE_UNDEFINED_P(lower) && !CONTRAINTE_UNDEFINED_P(upper))
            {
                expression elower = constraints_to_loop_bound(lower,phi,true,entity_intrinsic(DIVIDE_OPERATOR_NAME));
                expression eupper = constraints_to_loop_bound(upper,phi,false,entity_intrinsic(DIVIDE_OPERATOR_NAME));
                new_dimensions=CONS(DIMENSION, make_dimension(elower,eupper),new_dimensions);
            }
            else {
                pips_user_warning("failed to translate region\n");
            }
        }
        new_dimensions=gen_nreverse(new_dimensions);
        gen_full_free_list(variable_dimensions(type_variable(entity_type(e))));
        variable_dimensions(type_variable(entity_type(e)))=new_dimensions;
        gen_free_list(phis);
        /* formal entites are a special case: the actual parameter declaration must be changed too*/

        /* currently disabled, see you later, aligator */
        if(formal_parameter_p(e))
            statement_insertion_fix_access_in_callers(get_current_module_name(),e);
    }
}

static bool do_statement_insertion(statement s)
{
    /* first find a statement with the relevant pragma */
    string inserted_pragma = get_string_property("STATEMENT_INSERTION_PRAGMA");
    if(empty_string_p(inserted_pragma)) {
        pips_user_warning("STATEMENT_INSERTION_PRAGMA property should not be empty\n");
    }
    else {
        list flagged_statements = find_statements_with_pragma(s,inserted_pragma);
        if(ENDP(flagged_statements)) {
            pips_user_warning("no statement with pragma '%s' found\n",inserted_pragma);
        }
        else {
            FOREACH(STATEMENT,flagged_statement,flagged_statements) {
                list out_regions = effects_effects(load_proper_rw_effects(flagged_statement));
                if(ENDP(out_regions))
                {
                    /* make sure all data access are ok by building the convex union of data access and data declarations */
                    list regions = load_cumulated_rw_effects_list(flagged_statement);
                    statement_insertion_fix_access(regions);
                    /* update pragma */
                    FOREACH(EXTENSION,ext,extensions_extension(statement_extensions(flagged_statement)))
                    {
                        if(pragma_string_p(extension_pragma(ext)))
                        {
                            string str_pragma = pragma_string(extension_pragma(ext));
                            if(strstr( str_pragma , inserted_pragma ))
                            {
                                free(str_pragma);
                                pragma_string(extension_pragma(ext))=strdup(get_string_property("STATEMENT_INSERTION_SUCCESS_PRAGMA"));
                            }
                        }
                    }
                    return true;
                }
                else
                {
                    pips_user_warning("inserted statment has out effects\n");
                    /* update pragma */
                    FOREACH(EXTENSION,ext,extensions_extension(statement_extensions(flagged_statement)))
                    {
                        if(pragma_string_p(extension_pragma(ext)))
                        {
                            string str_pragma = pragma_string(extension_pragma(ext));
                            if(strstr( str_pragma , inserted_pragma ))
                            {
                                free(str_pragma);
                                pragma_string(extension_pragma(ext))=strdup(get_string_property("STATEMENT_INSERTION_FAILURE_PRAGMA"));
                            }
                        }
                    }
                }

            }
        }

    }
    return false;
}

bool statement_insertion(const char *module_name)
{
    /* init */
    set_current_module_entity(module_name_to_entity( module_name ));
    set_current_module_statement((statement) db_get_memory_resource(DBR_CODE, module_name, true) );
    set_cumulated_rw_effects((statement_effects)db_get_memory_resource(DBR_REGIONS, module_name, true));
    set_proper_rw_effects((statement_effects)db_get_memory_resource(DBR_OUT_REGIONS, module_name, true));
    debug_on("STATEMENT_INSERTION_DEBUG_LEVEL");

    /* do */
    if(do_statement_insertion(get_current_module_statement()))
    {
        /* validate */
        module_reorder(get_current_module_statement());
        DB_PUT_MEMORY_RESOURCE(DBR_CODE, module_name,get_current_module_statement());
    }

    debug_off();
    reset_current_module_entity();
    reset_current_module_statement();
    reset_cumulated_rw_effects();
    reset_proper_rw_effects();
    return true;
}

