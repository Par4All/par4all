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
 * @file isolate_statement.c
 * transfer statement to isolate memory
 * @author Serge Guelton <serge.guelton@enst-bretagne.fr>
 * @date 2010-05-01
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
#include "callgraph.h"
#include "effects-generic.h"
#include "effects-simple.h"
#include "effects-convex.h"
#include "preprocessor.h"
#include "text-util.h"
#include "transformations.h"
#include "parser_private.h"
#include "syntax.h"
#include "c_syntax.h"
#include "locality.h"
#include "expressions.h"
#include "semantics.h"
#include "transformer.h"
#include "accel-util.h"


/**
 * isolate_statement
 */

typedef struct {
    entity old;
    entity new;
    list offsets;
} isolate_param;

/** 
 * replace reference @p r on entity @p p->old  by a reference on entity @p p->new with offsets @p p->offsets
 */
static void isolate_patch_reference(reference r, isolate_param * p)
{
    if(same_entity_p(reference_variable(r),p->old))
    {
        reference_variable(r)=p->new;
        list offsets = p->offsets;
        FOREACH(EXPRESSION,index,reference_indices(r))
        {
            expression offset = EXPRESSION(CAR(offsets));
            unnormalize_expression(index);
            expression_syntax(index)=
                make_syntax_call(
                        make_call(
                            entity_intrinsic(MINUS_OPERATOR_NAME),
                            make_expression_list(
                                copy_expression(index),
                                copy_expression(offset)
                                )
                            )
                        );
            NORMALIZE_EXPRESSION(index);
            POP(offsets);
        }
    }
}

static void isolate_patch_entities(void * ,entity , entity ,list );
/** 
 * run isolate_patch_entities on all declared entities from @p s
 */
static void isolate_patch_statement(statement s, isolate_param *p)
{
    FOREACH(ENTITY,e,statement_declarations(s))
    {
        if(!value_undefined_p(entity_initial(e)))
            isolate_patch_entities(entity_initial(e),p->old,p->new,p->offsets);
    }
}

/** 
 * replace all references on entity @p old by references on entity @p new and adds offset @p offsets to its indices
 */
static void isolate_patch_entities(void * where,entity old, entity new,list offsets)
{
    isolate_param p = { old,new,offsets };
    gen_context_multi_recurse(where,&p,
            reference_domain,gen_true,isolate_patch_reference,
            statement_domain,gen_true,isolate_patch_statement,
            0);
}

static bool
expression_minmax_p(expression e)
{
    if(expression_call_p(e))
    {
        entity op = call_function(expression_call(e));
        return ENTITY_MIN_P(op) || ENTITY_MAX_P(op);
    }
    return false;
}

/* replace caller by field , where field is conatianed by caller */
static void local_assign_expression(expression caller, expression field)
{
     syntax s = expression_syntax(field) ;
     expression_syntax(field)=syntax_undefined;
     free_syntax(expression_syntax(caller));
     expression_syntax(caller)=s;
     free_normalized(expression_normalized(caller));
}

static void bounds_of_expression(expression e, transformer tr,bool is_max)
{
    intptr_t lbound, ubound;
    if(precondition_minmax_of_expression(e,tr,&lbound,&ubound))
    {
        free_syntax(expression_syntax(e));
        free_normalized(expression_normalized(e));
        expression new = int_to_expression(is_max ? ubound : lbound);
        expression_syntax(e)=expression_syntax(new);
        expression_normalized(e)=expression_normalized(new);
        expression_syntax(new)=syntax_undefined;
        expression_normalized(new)=normalized_undefined;
        free_expression(new);
    }
}
static void upperbound_of_expression(expression e, transformer tr)
{
    bounds_of_expression(e,tr,true);
}
#if 0
static void lowerbound_of_expression(expression e, transformer tr)
{
    bounds_of_expression(e,tr,false);
}
#endif

static void simplify_minmax_expression(expression e,transformer tr)
{
    call c =expression_call(e);
    bool is_max = ENTITY_MAX_P(call_function(c));

    expression lhs = binary_call_lhs(c);
    expression rhs = binary_call_rhs(c);
    intptr_t lhs_lbound,lhs_ubound,rhs_lbound,rhs_ubound;
    if(precondition_minmax_of_expression(lhs,tr,&lhs_lbound,&lhs_ubound) &&
            precondition_minmax_of_expression(rhs,tr,&rhs_lbound,&rhs_ubound))
    {
        if(is_max)
        {
            if(lhs_lbound >=rhs_ubound) local_assign_expression(e,lhs);
            else if(rhs_lbound >= lhs_ubound) local_assign_expression(e,rhs);
        }
        else
        {
            if(lhs_lbound >=rhs_ubound) local_assign_expression(e,rhs);
            else if(rhs_lbound >= lhs_ubound) local_assign_expression(e,lhs);
        }
    }
}


/** 
 * generate a list of dimensions @p dims and of offsets @p from a region @p r
 * for example if r = a[phi0,phi1] 0<=phi0<=2 and 1<=phi1<=4
 * we get dims = ( (0,3), (0,4) )
 * and offsets = ( 0 , 1 )
 * if @p exact is set to false, we are allowed to give an upperbound to the dimensions
 * 
 * @return false if we were enable to gather enough informations
 */
static
bool region_to_minimal_dimensions(region r, transformer tr, list * dims, list *offsets,bool exact)
{
    pips_assert("empty parameters\n",ENDP(*dims)&&ENDP(*offsets));
    reference ref = region_any_reference(r);
    Psysteme sc = sc_dup(region_system(r));
    sc_transform_eg_in_ineg(sc);
    FOREACH(EXPRESSION,index,reference_indices(ref))
    {
        Variable phi = expression_to_entity(index);
        Pcontrainte lower,upper;
        constraints_for_bounds(phi, &sc_inegalites(sc), &lower, &upper);
        if( !CONTRAINTE_UNDEFINED_P(lower) && !CONTRAINTE_UNDEFINED_P(upper))
        {
            /* this is a constant : the dimension is 1 and the offset is the bound */
            if(bounds_equal_p(phi,lower,upper))
            {
                expression bound = constraints_to_loop_bound(lower,phi,true,entity_intrinsic(DIVIDE_OPERATOR_NAME));
                *dims=CONS(DIMENSION,make_dimension(int_to_expression(0),int_to_expression(0)),*dims);
                *offsets=CONS(EXPRESSION,bound,*offsets);
            }
            /* this is a range : the dimension is eupper-elower +1 and the offset is elower */
            else
            {
                expression elower = constraints_to_loop_bound(lower,phi,true,entity_intrinsic(DIVIDE_OPERATOR_NAME));
                expression eupper = constraints_to_loop_bound(upper,phi,false,entity_intrinsic(DIVIDE_OPERATOR_NAME));
                if(expression_minmax_p(elower))
                    simplify_minmax_expression(elower,tr);
                if(expression_minmax_p(eupper))
                    simplify_minmax_expression(eupper,tr);
                expression offset = copy_expression(elower);

                expression dim = make_op_exp(MINUS_OPERATOR_NAME,eupper,elower);
                if(!exact && (expression_minmax_p(elower)||expression_minmax_p(eupper)))
                    upperbound_of_expression(dim,tr);

                *dims=CONS(DIMENSION,
                        make_dimension(
                            int_to_expression(0),
                            dim
                            ),*dims);
                *offsets=CONS(EXPRESSION,offset,*offsets);
            }
        }
        else {
            pips_user_warning("failed to analyse region\n");
            return false;
        }
    }
    *dims=gen_nreverse(*dims);
    *offsets=gen_nreverse(*offsets);
    return true;
}

/** 
 * 
 * @return region from @p regions on entity @p e
 */
static region find_region_on_entity(entity e,list regions)
{
    FOREACH(REGION,r,regions)
        if(same_entity_p(e,reference_variable(region_any_reference(r)))) return r;
    return region_undefined;
}

/** 
 * @return list of expressions so that for all i, out(i) = global(i)+local(i)
 */
static list isolate_merge_offsets(list global, list local)
{
    list out = NIL;
    FOREACH(EXPRESSION,gexp,global)
    {
        expression lexp = EXPRESSION(CAR(local));
        out=CONS(EXPRESSION,make_op_exp(PLUS_OPERATOR_NAME,copy_expression(gexp),copy_expression(lexp)),out);
        POP(local);
    }
    return gen_nreverse(out);
}

/** 
 * @return a range suitable for iteration over all the elements of dimension @p d
 */
range dimension_to_range(dimension d)
{
    return make_range(
            copy_expression(dimension_lower(d)),
            copy_expression(dimension_upper(d)),
            int_to_expression(1));
}

/** 
 * @return a list of @p nb new integer entities 
 */
static list isolate_generate_indices(size_t nb)
{
    list indices = NIL;
    while(nb--)
    {
        entity e = make_new_scalar_variable_with_prefix("i",get_current_module_entity(),make_basic_int(DEFAULT_INTEGER_TYPE_SIZE));
        AddEntityToCurrentModule(e);
        indices=CONS(ENTITY,e,indices);
    }
    return indices;
}

/** 
 * @return a list of expressions, one for each entity in @p indices
 */
static list isolate_indices_to_expressions(list indices)
{
    list expressions=NIL;
    FOREACH(ENTITY,e,indices)
        expressions=CONS(EXPRESSION,entity_to_expression(e),expressions);
    return gen_nreverse(expressions);
}

typedef enum {
    transfer_in,
    transfer_out
} isolate_transfer;
#define transfer_in_p(e) ( (e) == transfer_in )
#define transfer_out_p(e) ( (e) == transfer_out )

/** 
 * 
 * @return a statement holding the loops necessary to initialize @p new from @p old,
 * knowing the dimension of the isolated entity @p dimensions and its offsets @p offsets and the direction of the transfer @p t
 */
static statement isolate_make_array_transfer(entity old,entity new, list dimensions, list offsets,isolate_transfer t)
{
    /* first create the assignment : we need a list of indices */
    list index_entities = isolate_generate_indices(gen_length(dimensions));
    list index_expressions = isolate_indices_to_expressions(index_entities);
    list index_expressions_with_offset = 
        isolate_merge_offsets(index_expressions,offsets);


    statement body = make_assign_statement(
            reference_to_expression(
                make_reference(new,transfer_in_p(t)?index_expressions:index_expressions_with_offset)
                ),
            reference_to_expression(
                make_reference(old,transfer_in_p(t)?index_expressions_with_offset:index_expressions)
                )
            );

    FOREACH(DIMENSION,d,dimensions)
    {
        entity index = ENTITY(CAR(index_entities));
        body = instruction_to_statement(
                make_instruction_loop(
                    make_loop(
                        index,
                        dimension_to_range(d),
                        body,
                        entity_empty_label(),
                        make_execution_sequential(),
                        NIL
                        )
                    )
                );
        POP(index_entities);
    }
    /* add a nice comment */
    asprintf(&statement_comments(body),"/* transfer loop generated by PIPS from %s to %s */",entity_user_name(old),entity_user_name(new));
    return body;
}

/** 
 * isolate statement @p s from the outer memory, generating appropriate local array copy and copy code
 */
static void do_isolate_statement(statement s)
{
    list regions = load_cumulated_rw_effects_list(s);

    list read_regions = regions_read_regions(regions);
    list write_regions = regions_write_regions(regions);
    transformer tr = transformer_range(load_statement_precondition(s));

    statement prelude=make_empty_block_statement(),postlude=make_empty_block_statement();
    set visited_entities = set_make(set_pointer);

    FOREACH(REGION,reg,regions)
    {
        reference r = region_any_reference(reg);
        entity e = reference_variable(r);
        /* check we have not already dealt with this variable */
        if(!set_belong_p(visited_entities,e))
        {
            set_add_element(visited_entities,visited_entities,e);
            /* get the associated read and write regions, used for copy-in and copy-out
             * later on, in and out regions may be used instead
             * */
            region read_region = find_region_on_entity(e,read_regions);
            region write_region = find_region_on_entity(e,write_regions);

            /* compute their convex hull : that's what we need to allocate
             * in that case, the read and write regions must be used
             * */
            region rw_region = 
                region_undefined_p(read_region)?write_region:
                region_undefined_p(write_region)?read_region:
                regions_must_convex_hull(read_region,write_region);

            /* based on the rw_region, we can allocate a new entity with proper dimensions
             */
            list offsets = NIL,dimensions=NIL;
            if(region_to_minimal_dimensions(rw_region,tr,&dimensions,&offsets,false))
            {
                /* a scalar */
                if(ENDP(dimensions))
                {
                }
                /* an array */
                else
                {
                    /* create the new entity */
                    entity new = make_new_array_variable_with_prefix(entity_local_name(e),get_current_module_entity(),copy_basic(entity_basic(e)),dimensions);
                    AddLocalEntityToDeclarations(new,get_current_module_entity(),s);

                    /* replace it everywhere, and patch references*/
                    isolate_patch_entities(s,e,new,offsets);

                    /* generate the copy - in from read region */
                    if(!region_undefined_p(read_region))
                    {
                        list read_dimensions=NIL,read_offsets=NIL;
                        if(region_to_minimal_dimensions(read_region,tr,&read_dimensions,&read_offsets,true))
                        {
                            insert_statement(prelude,isolate_make_array_transfer(e,new,read_dimensions,read_offsets,transfer_in),true);
                        }
                        else
                        {
                            pips_user_warning("failed to recover information from read region\n");
                            return ;
                        }
                    }
                    /* and the copy-out from write region */
                    if(!region_undefined_p(write_region))
                    {
                        list write_dimensions=NIL,write_offsets=NIL;
                        if(region_to_minimal_dimensions(write_region,tr,&write_dimensions,&write_offsets,true))
                        {
                            insert_statement(postlude,isolate_make_array_transfer(new,e,write_dimensions,write_offsets,transfer_out),false);
                        }
                        else
                        {
                            pips_user_warning("failed to recover information from write region\n");
                            return ;
                        }
                    }
                }

            }
            else
            {
                pips_user_warning("failed to convert regions to minimal array dimensions, using whole array instead\n");
                return ;
            }

        }

    }
    insert_statement(s,prelude,true);
    insert_statement(s,postlude,false);




    set_free(visited_entities);
    gen_free_list(read_regions);
    gen_free_list(write_regions);
    free_transformer(tr);
}

bool
isolate_statement(string module_name)
{
    set_current_module_entity(module_name_to_entity( module_name ));
    set_current_module_statement((statement) db_get_memory_resource(DBR_CODE, module_name, true) );
    set_cumulated_rw_effects((statement_effects)db_get_memory_resource(DBR_REGIONS, module_name, true));
    module_to_value_mappings(get_current_module_entity());
    set_precondition_map( (statement_mapping) db_get_memory_resource(DBR_PRECONDITIONS, module_name, true) );

    string stmt_label=get_string_property("ISOLATE_STATEMENT_LABEL");
    statement statement_to_isolate = find_statement_from_label_name(get_current_module_statement(),get_current_module_name(),stmt_label);
    do_isolate_statement(statement_to_isolate);



    /* validate */
    module_reorder(get_current_module_statement());
    DB_PUT_MEMORY_RESOURCE(DBR_CODE, module_name,get_current_module_statement());

    reset_current_module_entity();
    reset_current_module_statement();
    reset_cumulated_rw_effects();
    reset_precondition_map();
    free_value_mappings();

    return true;
}

