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
 * @file array_to_pointer.c
 * transform arrays to low-level pointers or 1D arrays
 * @author Serge Guelton <serge.guelton@enst-bretagne.fr>
 * @date 2009-07-01
 */
#ifdef HAVE_CONFIG_H
    #include "pips_config.h"
#endif
#include <ctype.h>

#include "genC.h"
#include "ri-util.h"
#include "pipsdbm.h"
#include "resources.h"
#include "properties.h"
#include "misc.h"
#include "control.h"
#include "preprocessor.h"
#include "accel-util.h"

typedef enum {
    NO_CONVERSION,
    ARRAY_1D,
    POINTER
}array_to_pointer_conversion_mode ;

static
array_to_pointer_conversion_mode get_array_to_pointer_conversion_mode()
{
    string mode = get_string_property("ARRAY_TO_POINTER_CONVERT_PARAMETERS");
    if(!mode || empty_string_p(mode)) return NO_CONVERSION;
    else if(same_string_p(mode,"1D")) return ARRAY_1D;
    else if(same_string_p(mode,"POINTER")) return POINTER;
    else pips_user_error("bad value %s for property ARRAY_TO_POINTER_CONVERT_PARAMETERS\n",mode);
    return NO_CONVERSION;
}
expression reference_offset(reference ref)
{
    expression address_computation = EXPRESSION(CAR(reference_indices(ref)));

    /* iterate on the dimensions & indices to create the index expression */
    list dims = variable_dimensions(type_variable(entity_type(reference_variable(ref))));
    list indices = reference_indices(ref);
    POP(indices);
    if(!ENDP(dims)) POP(dims); // the first dimension is unused
    FOREACH(DIMENSION,dim,dims)
    {
        expression dimension_size = make_op_exp(
                PLUS_OPERATOR_NAME,
                make_op_exp(
                    MINUS_OPERATOR_NAME,
                    copy_expression(dimension_upper(dim)),
                    copy_expression(dimension_lower(dim))
                    ),
                make_expression_1());

        if( !ENDP(indices) ) { /* there may be more dimensions than indices */
            expression index_expression = EXPRESSION(CAR(indices));
            address_computation = make_op_exp(
                    PLUS_OPERATOR_NAME,
                    index_expression,
                    make_op_exp(
                        MULTIPLY_OPERATOR_NAME,
                        dimension_size,address_computation
                        )
                    );
            POP(indices);
        }
        else {
            address_computation = make_op_exp(
                    MULTIPLY_OPERATOR_NAME,
                    dimension_size,address_computation
                    );
        }
    }

    /* there may be more indices than dimensions */
    FOREACH(EXPRESSION,e,indices)
    {
        address_computation = make_op_exp(
                PLUS_OPERATOR_NAME,
                address_computation,e
                );
    }
    return address_computation ;
}


/**
 * transform each subscript in expression @a exp into the equivalent pointer arithmetic expression
 *
 * @param exp expression to inspect
 *
 * @return true
 */
static
bool expression_array_to_pointer(expression exp, bool in_init)
{
    if(expression_reference_p(exp))
    {
        reference ref = expression_reference(exp);
        if( ! ENDP(reference_indices(ref) ) )
        {
            /* we need to check if we know the dimension of this reference */
            size_t nb_indices =gen_length(reference_indices(ref));
            size_t nb_dims =gen_length(variable_dimensions(type_variable(entity_type(reference_variable(ref))))) ;

            /* if the considered reference is a formal parameter and the property is properly set,
             * we are allowded to convert formal parameters such as int a[n][12] into int *a
             */
            bool force_cast = true;
            if( get_array_to_pointer_conversion_mode()==POINTER && 
                    ! get_bool_property("ARRAY_TO_POINTER_FLATTEN_ONLY") &&
                    formal_parameter_p(reference_variable(ref)) )
            {
                force_cast=false;
            }

            /* create a new reference without subscripts */
            reference ref_without_indices = make_reference(reference_variable(ref),NIL);

            expression base_ref = reference_to_expression(ref_without_indices);
            /* create a pointer if needed */
            if( !get_bool_property("ARRAY_TO_POINTER_FLATTEN_ONLY"))
            {
                /* get the base type of the reference */
                type type_without_indices = make_type_variable(make_variable(
                            copy_basic(variable_basic(type_variable(entity_type(reference_variable(ref))))),
                            NIL,
                            gen_full_copy_list(variable_qualifiers(type_variable(entity_type(reference_variable(ref)))))));


                /* create an expression for the new reference, possibly casted */
                if( force_cast && ! basic_pointer_p( variable_basic(type_variable(entity_type(reference_variable(ref) ) ) ) ) )
                {
                    base_ref = make_expression(
                            make_syntax_cast(
                                make_cast(
                                    make_type_variable(
                                        make_variable(
                                            make_basic_pointer(type_without_indices),NIL,NIL
                                            )
                                        ),
                                    base_ref)
                                ),
                            normalized_undefined);
                }
            }
            expression address_computation = reference_offset(ref);

            /* we now either add the DEREFERENCING_OPERATOR, or the [] */
            syntax new_syntax = syntax_undefined;
            if(!in_init && get_bool_property("ARRAY_TO_POINTER_FLATTEN_ONLY")) {
                reference_indices(ref_without_indices)=make_expression_list(address_computation);
                new_syntax=make_syntax_reference(ref_without_indices);
            }
            else {
                if(nb_indices == nb_dims || nb_dims == 0 ) {

                    new_syntax=make_syntax_call(
                            make_call(
                                CreateIntrinsic(DEREFERENCING_OPERATOR_NAME),
                                CONS(EXPRESSION,MakeBinaryCall(
                                        entity_intrinsic(PLUS_C_OPERATOR_NAME),
                                        base_ref,
                                        address_computation), NIL)
                                )
                            );
                }
                else {
                    new_syntax = make_syntax_call(
                            make_call(CreateIntrinsic(PLUS_C_OPERATOR_NAME),
                                make_expression_list(base_ref,address_computation))
                            );
                }
            }

            /* free stuffs */
            unnormalize_expression(exp);
            gen_free_list(reference_indices(ref));
            reference_indices(ref)=NIL;
            free_syntax(expression_syntax(exp));

            /* validate changes */
            expression_syntax(exp)=new_syntax;
        }

    }
    /* not tested */
    else if( syntax_subscript_p(expression_syntax(exp) ) )
    {
        subscript s = syntax_subscript(expression_syntax(exp));
        pips_assert("non empty subscript",!ENDP(subscript_indices(s)));
        call c = make_call(
                CreateIntrinsic(PLUS_C_OPERATOR_NAME),
                make_expression_list(
                    copy_expression(subscript_array(s)),
                    EXPRESSION(CAR(subscript_indices(s)))
                    ));
        list indices = subscript_indices(s);
        POP(indices);
        FOREACH(EXPRESSION,e,indices)
        {
            c = make_call(
                    CreateIntrinsic(PLUS_OPERATOR_NAME),
                    make_expression_list(call_to_expression(c),e));
        }
        unnormalize_expression(exp);
        gen_free_list(subscript_indices(s));
        subscript_indices(s)=NIL;
        free_syntax(expression_syntax(exp));
        expression_syntax(exp)=make_syntax_call(c);


    }
    return true;
}

/**
 * call expression_array_to_pointer on each entity declared in statement @s
 *
 * @param s statement to inspect
 *
 * @return true
 */
static
bool declaration_array_to_pointer(statement s,bool __attribute__((unused)) in_init)
{
    FOREACH(ENTITY,e,statement_declarations(s))
        gen_context_recurse(entity_initial(e),(void*)true,expression_domain,expression_array_to_pointer,gen_null);
    return true;
}

static
void make_pointer_from_variable(variable param)
{
    list parameter_dimensions = variable_dimensions(param);
    switch(get_array_to_pointer_conversion_mode())
    {
        case NO_CONVERSION:return;
        case ARRAY_1D:
            if(gen_length(parameter_dimensions) > 1)
            {
                list iter = parameter_dimensions;
                expression full_length = expression_undefined;
                {
                    dimension d = DIMENSION(CAR(iter));
                    full_length = SizeOfDimension(d);
                    POP(iter);
                }
                FOREACH(DIMENSION,d,iter)
                {
                    full_length=make_op_exp(
                            MULTIPLY_OPERATOR_NAME,
                            SizeOfDimension(d),
                            full_length);
                }
                gen_full_free_list(parameter_dimensions);
                variable_dimensions(param)=
                    CONS(DIMENSION,
                            make_dimension(
                                int_to_expression(0),
                                make_op_exp(MINUS_OPERATOR_NAME,full_length,int_to_expression(1))
                                ),
                            NIL);
            } break;
        case POINTER:
            if(!ENDP(parameter_dimensions))
            {
                gen_full_free_list(parameter_dimensions);
                variable_dimensions(param)=NIL;
                basic parameter_basic = variable_basic(param);
                basic new_parameter_basic = make_basic_pointer(
                        make_type_variable(
                            make_variable(parameter_basic,NIL,NIL)
                            )
                        );
                variable_basic(param)=new_parameter_basic;
            }
            break;

    }
}

static void make_pointer_from_all_variable(void* obj)
{
    gen_recurse(obj,variable_domain,gen_true,make_pointer_from_variable);
}
static
void make_pointer_entity_from_reference_entity(entity e)
{
    variable param = type_variable(entity_type(e));
    make_pointer_from_all_variable(param);
}

static void
reduce_array_declaration_dimension(statement s)
{
    FOREACH(ENTITY,e,statement_declarations(s))
    {
        type t = ultimate_type(entity_type(e));
        if(type_variable_p(t))
        {
            variable v = type_variable(t);
            if(!ENDP(variable_dimensions(v)))
            {
                expression new_dim = expression_undefined;
                FOREACH(DIMENSION,d,variable_dimensions(v))
                {
                    new_dim = expression_undefined_p(new_dim)?
                        SizeOfDimension(d):
                        make_op_exp(MULTIPLY_OPERATOR_NAME,
                                new_dim,
                                SizeOfDimension(d)
                                );
                    print_expression(new_dim);
                }
                gen_full_free_list(variable_dimensions(v));
                variable_dimensions(v)=CONS(DIMENSION,make_dimension(make_expression_0(),make_op_exp(MINUS_OPERATOR_NAME,new_dim,make_expression_1())),NIL);
            }
        }
    }
}

bool array_to_pointer(char *module_name)
{
    /* prelude */
    set_current_module_entity(module_name_to_entity( module_name ));
    set_current_module_statement((statement) db_get_memory_resource(DBR_CODE, module_name, TRUE) );

    /* run transformation */
    if(!c_module_p(get_current_module_entity()))
        pips_user_warning("this transformation will have no effect on a fortran module\n");
    else
    {
        gen_context_multi_recurse(get_current_module_statement(),(void*)false,
                expression_domain,expression_array_to_pointer,gen_null,
                statement_domain,declaration_array_to_pointer,gen_null,
                NULL);
        /* now fix array declarations : one dimension for everyone ! */
        gen_recurse(get_current_module_statement(),statement_domain,gen_true,reduce_array_declaration_dimension);

        /* eventually change the signature of the module
         * tricky : signature must be changed in two places !
         */
        {
            FOREACH(ENTITY,e,code_declarations(value_code(entity_initial(get_current_module_entity()))))
            {
                if(formal_parameter_p(e))
                    make_pointer_entity_from_reference_entity(e);
            }
            FOREACH(PARAMETER,p,functional_parameters(type_functional(entity_type(get_current_module_entity()))))
            {
                dummy d = parameter_dummy(p);
                if(dummy_identifier_p(d))
                    make_pointer_entity_from_reference_entity(dummy_identifier(d));
                type t = parameter_type(p);
                make_pointer_from_all_variable(t);
            }
        }

    }

    /* validate */
    module_reorder(get_current_module_statement());
    DB_PUT_MEMORY_RESOURCE(DBR_CODE, module_name,get_current_module_statement());

    /*postlude*/
    reset_current_module_entity();
    reset_current_module_statement();
    return true;
}

