/*
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
#include "semantics.h"
#include "transformer.h"
#include "control.h"
#include "effects-generic.h"
#include "effects-simple.h"
#include "effects-convex.h"
#include "text-util.h"
#include "parser_private.h"
#include "accel-util.h"

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
                    for(size_t i=1;i<nb_indices;i++) POP(*iter);
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
        Psysteme dims_syst = entity_declaration_sc(e);
        Psysteme access_syst = region_system(reg);

        volatile Psysteme sr;
        CATCH(overflow_error)
        {
            pips_debug(1, "overflow error\n");
            return ;
        }
        TRY
        {
            sr = sc_cute_convex_hull(access_syst, dims_syst);
            sc_rm(dims_syst);
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
    const char* inserted_pragma = get_string_property("STATEMENT_INSERTION_PRAGMA");
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
typedef struct {
    entity e;
    statement s;
} entity_to_declaring_statement_t;

static void entity_to_declaring_statement_aux(statement s, entity_to_declaring_statement_t *param) {
    if(statement_block_p(s)) {
        if(entity_in_list_p(param->e,statement_declarations(s))) {
            param->s=s;
            gen_recurse_stop(0);
        }
    }
}


/* returns the statement block declaring entity @p e among all thoses in @p top
 * assumes the entity *is* declared locally
 * c only
 */
static statement entity_to_declaring_statement(entity e, statement top) {
    entity_to_declaring_statement_t param = { e, statement_undefined };
    if(formal_parameter_p(e))
        return top;
    else {
        gen_context_recurse(top,&param,statement_domain,gen_true,entity_to_declaring_statement_aux);
        pips_assert("entity not a local entity ?",!statement_undefined_p(param.s));
        return param.s;
    }
}

static void do_array_expansion_aux(statement s, hash_table expanded) {
    list remove = NIL;
    HASH_FOREACH(entity,e,statement,sp,expanded) {
        if(s==sp)
            remove=CONS(ENTITY,e,remove);
    }
    FOREACH(ENTITY,e,remove)
        hash_del(expanded,e);
    gen_free_list(remove);
}

static bool do_array_expansion(statement s, hash_table expanded) {
    list regions = load_cumulated_rw_effects_list(s);
    set declarations =set_make(set_pointer);
    set_assign_list(declarations,entity_declarations(get_current_module_entity()));
    FOREACH(REGION,reg,regions)
    {
        reference r = region_any_reference(reg);
        entity e = reference_variable(r);
        if(set_belong_p(declarations,e) &&
                hash_get(expanded,e)==HASH_UNDEFINED_VALUE &&
                array_entity_p(e) ) {

            statement sdecl = entity_to_declaring_statement(e,get_current_module_statement());
            transformer tr = transformer_range(
                    load_statement_precondition(sdecl));

            list phis = expressions_to_entities(reference_indices(r));
            Psysteme access_syst = region_system(reg);
            Psysteme decl = entity_declaration_sc(e);

            volatile Psysteme sr;
            CATCH(overflow_error)
            {
                pips_debug(1, "overflow error\n");
                return true;
            }
            TRY
            {
                Psysteme tmp = sc_cute_convex_hull(access_syst,decl);
                tmp=
                    sc_safe_append(tmp,
                            predicate_system(transformer_relation(tr)));
                sc_nredund(&tmp);
                Pbase pb = list_to_base(phis);
                sr = sc_rectangular_hull(tmp,pb);
                sc_rm(tmp);
                UNCATCH(overflow_error);
            }
            /* if we reach this point, we are ready for backward translation from vecteur to dimensions :) */
            list new_dimensions = NIL;
            bool ok=true;
            FOREACH(ENTITY,phi,phis)
            {
                Pcontrainte lower,upper;
                constraints_for_bounds(phi, &sc_inegalites(sr), &lower, &upper);
                if( !CONTRAINTE_UNDEFINED_P(lower) && !CONTRAINTE_UNDEFINED_P(upper))
                {
                    expression elower = constraints_to_loop_bound(lower,phi,true,entity_intrinsic(DIVIDE_OPERATOR_NAME));
                    simplify_minmax_expression(elower,tr);
                    expression eupper = constraints_to_loop_bound(upper,phi,false,entity_intrinsic(DIVIDE_OPERATOR_NAME));
                    simplify_minmax_expression(eupper,tr);
                    new_dimensions=CONS(DIMENSION, make_dimension(elower,eupper),new_dimensions);
                }
                else {
                    pips_user_warning("failed to translate region\n");
                    ok=false;
                }
            }
            transformer_free(tr);
            if(ok) {
                hash_put(expanded,e,s);
                new_dimensions=gen_nreverse(new_dimensions);
                gen_full_free_list(variable_dimensions(type_variable(entity_type(e))));
                variable_dimensions(type_variable(entity_type(e)))=new_dimensions;
                gen_free_list(phis);
                if(formal_parameter_p(e)) {
                    formal f = storage_formal(entity_storage(e));
                    intptr_t i=0,offset = formal_offset(f);
                    FOREACH(PARAMETER,p,module_functional_parameters(get_current_module_entity())) {
                        if(i++ == offset) {
                            dummy d = parameter_dummy(p);
                            if(dummy_identifier_p(d))
                            {
                                entity di = dummy_identifier(d);
                                variable v = type_variable(entity_type(di));
                                gen_full_free_list(variable_dimensions(v));
                                variable_dimensions(v)=gen_full_copy_list(new_dimensions);
                            }
                        }
                    }
                }
            }
        }
    }
    set_free(declarations);
    return true;
}

bool array_expansion(const char *module_name)
{
    /* init */
    set_current_module_entity(module_name_to_entity( module_name ));
    set_current_module_statement((statement) db_get_memory_resource(DBR_CODE, module_name, true) );
    set_cumulated_rw_effects((statement_effects)db_get_memory_resource(DBR_REGIONS, module_name, true));
    module_to_value_mappings(get_current_module_entity());
    set_precondition_map( (statement_mapping) db_get_memory_resource(DBR_PRECONDITIONS, module_name, true) );
    debug_on("ARRAY_EXPANSION_DEBUG_LEVEL");

    /* do */
    hash_table expanded = hash_table_make(hash_pointer,HASH_DEFAULT_SIZE);
    gen_context_recurse(get_current_module_statement(),expanded,
            statement_domain,do_array_expansion,do_array_expansion_aux);
    hash_table_free(expanded);

    /* validate */
    module_reorder(get_current_module_statement());
    DB_PUT_MEMORY_RESOURCE(DBR_CODE, module_name,get_current_module_statement());

    debug_off();
    reset_current_module_entity();
    reset_current_module_statement();
    reset_cumulated_rw_effects();
    reset_precondition_map();
    free_value_mappings();
    return true;
}
