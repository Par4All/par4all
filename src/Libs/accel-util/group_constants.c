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
 * @file group_constants.c
 * provide several layour for program constants grouping
 * @author Serge Guelton <serge.guelton@enst-bretagne.fr>
 * @date 2010-05-12
 */
#ifdef HAVE_CONFIG_H
    #include "pips_config.h"
#endif
#include <ctype.h>


#include "genC.h"
#include "ri-util.h"
#include "effects.h"
#include "effects-util.h"
#include "pipsdbm.h"
#include "resources.h"
#include "properties.h"
#include "misc.h"
#include "control.h"
#include "effects-generic.h"
#include "effects-convex.h"
#include "accel-util.h"

static void do_group_constant_entity(expression exp, set constants)
{
    if(expression_call_p(exp))
    {
        call c = expression_call(exp);
        value cv = entity_initial(call_function(c));
        if (value_symbolic_p(cv))
            set_add_element(constants,constants,call_function(c));
        else if (value_constant_p(cv) && get_bool_property("GROUP_CONSTANTS_LITERAL")) {
            set_add_element(constants,constants,call_function(c));

        }
    }
}

/* remove entities involved in a range from the set of constants */
static bool do_group_statement_constant_filter(range r,set constants) {
    set ref = get_referenced_entities(r);
    set_difference(constants,constants,ref);
    set_free(ref);
    return false;
}

static bool not_type_chunk_p(void* obj) {
    return !INSTANCE_OF(type,(gen_chunkp)obj);
}

/* remove entities that are never referenced by @p st from @p s */
static void do_group_statement_constant_prune(statement st, set s)
{
    set ref = get_referenced_entities_filtered(st,not_type_chunk_p,entity_not_constant_or_intrinsic_p);
    set_intersection(s,s,ref);
    set_free(ref);
}

/* gather all entities that are read and constant in statement @p s in 
 * the set @p of constant entities */
static void do_group_statement_constant(statement st, set constants)
{
    list regions = load_rw_effects_list(st);
    list read_regions = regions_read_regions(regions);
    list write_regions = regions_write_regions(regions);
    set written_entities = set_make(set_pointer);

    /* first gather all written entities */
    FOREACH(REGION,reg,write_regions)
    {
        reference ref = region_any_reference(reg);
        entity var = reference_variable(ref);
        set_add_element(written_entities,written_entities,var);
    }
    gen_free_list(write_regions);


    /* then search among all read and not written variables
     * those who have constant phis
     */
    FOREACH(REGION,reg,read_regions)
    {
        reference ref = region_any_reference(reg);
        entity var = reference_variable(ref);
        if(!set_belong_p(written_entities,var))
        {
            bool found_not_constant_constraint = false;
            Psysteme sc = sc_dup(region_system(reg));
            sc_transform_eg_in_ineg(sc);
            FOREACH(EXPRESSION,index,reference_indices(ref))
            {
                Variable phi = expression_to_entity(index);
                bool phi_found_p =false;
                for(Pcontrainte iter = sc_inegalites(sc);iter;iter=contrainte_succ(iter))
                {
                    Pvecteur cvect = contrainte_vecteur(iter);
                    Value phi_coeff = vect_coeff(phi,cvect);
                    /* we have found the right vector, now check for other vectors */
                    if(phi_coeff != VALUE_ZERO )
                    {
                        phi_found_p=true;
                        Pvecteur other_vects = vect_del_var(cvect,phi);
                        found_not_constant_constraint|=vect_size(other_vects)!=1 || VARIABLE_DEFINED_P(other_vects->var);
                        vect_rm(other_vects);
                    }
                }
                found_not_constant_constraint |=!phi_found_p;
            }
            if(!found_not_constant_constraint) {
                    set_add_element(constants,constants,var);
            }
            else {
                /* check the size too : small means mask */
                Ppolynome pp = region_enumerate(reg);
                if(!POLYNOME_UNDEFINED_P(pp)) {
                    expression epp = polynome_to_expression(pp);
                    intptr_t ipp;
                    if(expression_integer_value(epp,&ipp) && ipp < 33)
                        found_not_constant_constraint=false;
                    free_expression(epp);
                    polynome_free(pp);
                }
                if(!found_not_constant_constraint) 
                    set_add_element(constants,constants,var);
            }
        }
    }
    /* then prune this set, because it contains the preconditions too */
    do_group_statement_constant_prune(st,constants);

    /* eventually filter out some entities involved in range computation */
    if(get_bool_property("GROUP_CONSTANTS_SKIP_LOOP_RANGE"))
        gen_context_recurse(st,constants,range_domain,do_group_statement_constant_filter,gen_null);
    gen_free_list(read_regions);
    set_free(written_entities);
}

typedef enum {
    TERAPIX_GROUPING,
    GROUPING_UNDEFINED
} grouping_layout;

static grouping_layout get_grouping_layout()
{
    const char* layout = get_string_property("GROUP_CONSTANTS_LAYOUT");
    if(same_string_p(layout,"terapix")) return TERAPIX_GROUPING;
    return GROUPING_UNDEFINED;
}

static void* do_group_basics_maximum_reduce(void *v,const list l)
{
    return basic_maximum((basic)v,entity_basic(ENTITY(CAR(l))));
}

static basic do_group_basics_maximum(list entities)
{
    if(ENDP(entities)) return basic_undefined;
    else return (basic)gen_reduce(entity_basic(ENTITY(CAR(entities))),do_group_basics_maximum_reduce,CDR(entities));
}

static void *do_group_count_elements_reduce(void * v, const list l)
{
    entity e = ENTITY(CAR(l));
    if( (entity_variable_p(e) && entity_scalar_p(e)) || entity_constant_p(e))
        return  make_op_exp(PLUS_OPERATOR_NAME,copy_expression((expression)v),int_to_expression(1));
    else
        return make_op_exp(PLUS_OPERATOR_NAME,copy_expression((expression)v),SizeOfDimensions(variable_dimensions(type_variable(ultimate_type(entity_type(e))))));
}

static expression do_group_count_elements(list entities)
{
    return (expression)gen_reduce(int_to_expression(0),do_group_count_elements_reduce,entities);
}

static bool group_constant_range_filter(range r, set constants)
{
    return !get_bool_property("GROUP_CONSTANTS_SKIP_LOOP_RANGE");
}

static entity constant_holder;
static bool do_grouping_filter_out_self(expression exp)
{
    if(gen_get_ancestor(range_domain,exp) && get_bool_property("GROUP_CONSTANTS_SKIP_LOOP_RANGE") )
        return false;
    if(expression_reference_p(exp))
    {
        reference ref = expression_reference(exp);
        return !same_entity_p(reference_variable(ref),constant_holder);
    }
    return true;
}


typedef struct {
    entity old;
    entity new;
    expression offset;
} grouping_context;

static void do_grouping_replace_reference_by_expression_walker(expression exp,grouping_context *ctxt)
{
    if(expression_reference_p(exp))
    {
        reference ref = expression_reference(exp);
        if(same_entity_p(reference_variable(ref),ctxt->old))
        {
            /* compute new index */
            expression current_index = reference_offset(ref);
            /* perform substitution */
            reference_variable(ref)=ctxt->new;
            gen_full_free_list(reference_indices(ref));
            reference_indices(ref)=make_expression_list(make_op_exp(PLUS_OPERATOR_NAME,current_index,copy_expression(ctxt->offset)));
        }
    }
}

static void do_grouping_replace_reference_by_expression(void *in,entity old,entity new,expression offset)
{
    grouping_context ctxt = { old,new,offset };
    gen_context_recurse(in,&ctxt,expression_domain,do_grouping_filter_out_self,do_grouping_replace_reference_by_expression_walker);
}

static void do_group_constants_terapix(statement in,set constants)
{
    list lconstants = set_to_sorted_list(constants,(gen_cmp_func_t)compare_entities);
    basic max = do_group_basics_maximum(lconstants);
    if(!basic_undefined_p(max))
    {
        expression size = do_group_count_elements(lconstants);
        constant_holder = make_new_array_variable_with_prefix(
                get_string_property("GROUP_CONSTANTS_HOLDER"),
                get_current_module_entity(),
                basic_overloaded_p(max)?make_basic_int(DEFAULT_INTEGER_TYPE_SIZE):copy_basic(max),
                CONS(DIMENSION,make_dimension(int_to_expression(0),make_op_exp(MINUS_OPERATOR_NAME,size,int_to_expression(1))),NIL)
                );

        /* it may not be possible to initialize statically the array, so use loop initialization
         * to be more general
         */
        list initializations_holder_seq = NIL;
        expression index = int_to_expression(0);

        FOREACH(ENTITY,econstant,lconstants)
        {
            if((entity_variable_p(econstant) && entity_scalar_p(econstant)) || entity_constant_p(econstant))
            {
                expression new_constant_exp = reference_to_expression(make_reference(constant_holder,make_expression_list(copy_expression(index))));
                initializations_holder_seq=CONS(STATEMENT,
                        make_assign_statement(
                            new_constant_exp,
                            entity_to_expression(econstant)
                            ),
                        initializations_holder_seq);
                replace_entity_by_expression_with_filter(in,econstant,new_constant_exp,do_grouping_filter_out_self);
                index=make_op_exp(PLUS_OPERATOR_NAME,index,int_to_expression(1));
            }
            else {
                list indices = NIL;
                reference lhs = make_reference(constant_holder,NIL);// the indices will be set later
                reference rhs = make_reference(econstant,NIL);// the indices will be set later I told you
                statement body = make_assign_statement(reference_to_expression(lhs),reference_to_expression(rhs));
                list dimensions = gen_copy_seq(variable_dimensions(type_variable(ultimate_type(entity_type(econstant)))));
                dimensions=gen_nreverse(dimensions);
                FOREACH(DIMENSION,dim,dimensions)
                {
                    entity lindex = make_new_scalar_variable_with_prefix("z",get_current_module_entity(),make_basic_int(DEFAULT_INTEGER_TYPE_SIZE));
                    AddLocalEntityToDeclarations(lindex,get_current_module_entity(),in);
                    loop l = make_loop(lindex,dimension_to_range(dim),body,entity_empty_label(),make_execution_sequential(),NIL);
                    body=instruction_to_statement(make_instruction_loop(l));
                    indices=CONS(EXPRESSION,entity_to_expression(lindex),indices);
                }
                reference_indices(rhs)=indices;
                reference_indices(lhs)=make_expression_list(make_op_exp(PLUS_OPERATOR_NAME,copy_expression(index),reference_offset(rhs)));
                initializations_holder_seq=CONS(STATEMENT,body,initializations_holder_seq);
                do_grouping_replace_reference_by_expression(in,econstant,constant_holder,index);
                index=make_op_exp(PLUS_OPERATOR_NAME,index,SizeOfDimensions(dimensions));
                gen_free_list(dimensions);
            }
        }
        insert_statement(in,make_block_statement(gen_nreverse(initializations_holder_seq)),true);
        AddLocalEntityToDeclarations(constant_holder,get_current_module_entity(),in);
    }
    gen_free_list(lconstants);
}



static void do_group_constants(statement in,set constants)
{
    /* as of now, put everything in an array, no matter of the real type, yes it is horrible,
     * but it matches my needs for terapix, where everything as the same type
     * later on, you may want to use a structure to pass parameters*/
    switch(get_grouping_layout()) {
        case TERAPIX_GROUPING:
            return do_group_constants_terapix(in,constants);
        case GROUPING_UNDEFINED:
            pips_user_error("no valid grouping layout given\n");
    }
}

bool
group_constants(const char *module_name)
{
    /* prelude */
    set_current_module_entity(module_name_to_entity( module_name ));
    set_current_module_statement((statement) db_get_memory_resource(DBR_CODE, module_name, true) );
    set_rw_effects((statement_effects) db_get_memory_resource(DBR_REGIONS,module_name,true) );

    set/*of entities*/ constants=set_make(set_pointer);;

    /* gather statement constants */
    statement constant_statement = find_statement_from_label_name(get_current_module_statement(),module_name,get_string_property("GROUP_CONSTANTS_STATEMENT_LABEL"));
    if(statement_undefined_p(constant_statement))  constant_statement=get_current_module_statement();
    do_group_statement_constant(constant_statement,constants);

    /* gather constants */
    gen_context_multi_recurse(constant_statement,constants,
            range_domain,group_constant_range_filter,gen_null,
            reference_domain,gen_false,gen_null,
            expression_domain,gen_true,do_group_constant_entity,NULL);

    /* pack all constants and perform replacement */
    do_group_constants(constant_statement,constants);

    set_free(constants);

    /* validate */
    module_reorder(get_current_module_statement());
    DB_PUT_MEMORY_RESOURCE(DBR_CODE, module_name,get_current_module_statement());

    /*postlude*/
    reset_current_module_entity();
    reset_current_module_statement();
    reset_rw_effects();
    return true;
}
