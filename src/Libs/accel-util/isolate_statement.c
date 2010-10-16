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
#include "effects.h"
#include "ri-util.h"
#include "effects-util.h"
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
#include "semantics.h"
#include "transformer.h"
#include "callgraph.h"
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
        list offsets = p->offsets;
        list indices = reference_indices(r);
        FOREACH(EXPRESSION,index,indices)
        {
            if(!ENDP(offsets)) {
                expression offset = EXPRESSION(CAR(offsets));
                if(!entity_field_p(reference_variable(expression_reference(offset)))){
                    update_expression_syntax(index,
                            make_syntax_call(
                                make_call(
                                    entity_intrinsic(MINUS_OPERATOR_NAME),
                                    make_expression_list(
                                        copy_expression(index),
                                        copy_expression(offset)
                                        )
                                    )
                                )
                            );
                }
                POP(offsets);
            }
        }
        syntax snew = make_syntax_subscript(
                make_subscript(
                    MakeUnaryCall(
                        entity_intrinsic(DEREFERENCING_OPERATOR_NAME),
                        entity_to_expression(p->new)
                        ),
                    indices)
                );
        expression parent = (expression)gen_get_ancestor(expression_domain,r);
        expression_syntax(parent)=syntax_undefined;
        update_expression_syntax(parent,snew);

    }
}

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
void isolate_patch_entities(void * where,entity old, entity new,list offsets)
{
    isolate_param p = { old,new,offsets };
    gen_context_multi_recurse(where,&p,
            reference_domain,gen_true,isolate_patch_reference,
            statement_domain,gen_true,isolate_patch_statement,
            0);
}

static
bool expression_minmax_p(expression e)
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
static void lowerbound_of_expression(expression e, transformer tr)
{
    bounds_of_expression(e,tr,false);
}

void simplify_minmax_expression(expression e,transformer tr)
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
 * if at least one of the resulting dimension can be 0 (accroding to preconditions)
 * @p dimension_may_be_null is set to true
 * 
 * @return false if we were enable to gather enough informations
 */
bool region_to_minimal_dimensions(region r, transformer tr, list * dims, list *offsets,bool exact, expression *condition)
{
    pips_assert("empty parameters\n",ENDP(*dims)&&ENDP(*offsets));
    reference ref = region_any_reference(r);
    for(list iter = reference_indices(ref);!ENDP(iter); POP(iter))
    {
        expression index = EXPRESSION(CAR(iter));
        Variable phi = expression_to_entity(index);
        if(variable_phi_p((entity)phi)) {
            Psysteme sc = sc_dup(region_system(r));
            sc_transform_eg_in_ineg(sc);
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

                    bool compute_upperbound_p = 
                        !exact && (expression_minmax_p(elower)||expression_minmax_p(eupper));
                    expression dim = make_op_exp(MINUS_OPERATOR_NAME,eupper,elower);
                    if(compute_upperbound_p)
                        upperbound_of_expression(dim,tr);

                    /* sg : check if lower bound can be 0, in that case issue a ward */
                    if(condition!=0) {
                        expression lowerbound = copy_expression(dim);
                        lowerbound_of_expression(lowerbound,tr);
                        intptr_t lowerbound_value;
                        if(!expression_integer_value(lowerbound,&lowerbound_value) ||
                                lowerbound_value<=0) {
                            expression thetest = 
                                MakeBinaryCall(
                                        entity_intrinsic(GREATER_THAN_OPERATOR_NAME),
                                        copy_expression(dim),
                                        int_to_expression(0)
                                        );
                            if(expression_undefined_p(*condition))
                                *condition=thetest;
                            else
                                *condition=MakeBinaryCall(
                                        entity_intrinsic(AND_OPERATOR_NAME),
                                        *condition,
                                        thetest
                                        );
                        }
                    }

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
                sc_free(sc);
                return false;
            }
            sc_free(sc);
        }
        /* index is a field ... */
        else { /* and the last field, store it as an extra dimension */
            *dims=CONS(DIMENSION,
                    make_dimension(
                        int_to_expression(0),
                        int_to_expression(0)
                        ),*dims);
            *offsets=CONS(EXPRESSION,copy_expression(index),*offsets);
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
region find_region_on_entity(entity e,list regions)
{
    FOREACH(REGION,r,regions)
        if(same_entity_p(e,reference_variable(region_any_reference(r)))) return r;
    return region_undefined;
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

typedef enum {
    transfer_in,
    transfer_out
} isolate_transfer;
#define transfer_in_p(e) ( (e) == transfer_in )
#define transfer_out_p(e) ( (e) == transfer_out )


/* because of the way we build offsets list, it may contains struct field
 * so we cannot rely on make_reference only
 * fixes entity type as well ...
 * fix it here
 */
expression region_reference_to_expression(reference r)
{
    entity e = reference_variable(r);
    list indices = gen_full_copy_list(reference_indices(r));
    entity f = entity_undefined;
    size_t where = 0;
    FOREACH(EXPRESSION,exp,indices) {
        if(entity_field_p(f=reference_variable(expression_reference(exp))))
            break;
        where++;
    }
    list tail = gen_nthcdr(where,indices);
    if(where) {
        CDR(gen_nthcdr(where-1,indices))=NIL;
    }
    if(ENDP(tail))
        return reference_to_expression(make_reference(e,indices));
    else {
        reference fake = make_reference(f,CDR(tail));
        expression res =  binary_intrinsic_expression(
                FIELD_OPERATOR_NAME,
                reference_to_expression(make_reference(e,indices)),
                region_reference_to_expression(fake));
        free_reference(fake);
        return res;
    }
}

/** 
 * @return a statement holding the loop necessary to initialize @p new from @p old,
 * knowing the dimension of the isolated entity @p dimensions and its offsets @p offsets and the direction of the transfer @p t
 */

bool
isolate_statement(const char* module_name)
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
    DB_PUT_MEMORY_RESOURCE(DBR_CALLEES, module_name, compute_callees(get_current_module_statement()));

    reset_current_module_entity();
    reset_current_module_statement();
    reset_cumulated_rw_effects();
    reset_precondition_map();
    free_value_mappings();

    return true;
}

