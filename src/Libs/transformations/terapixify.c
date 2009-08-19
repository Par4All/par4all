/*
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

/** 
 * @file terapixify.c
 * apply transformations required to generate terapix microcode
 * @author Serge Guelton <serge.guelton@enst-bretagne.fr>
 * @date 2009-07-01
 */

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "ri-util.h"
#include "text.h"
#include "pipsdbm.h"
#include "resources.h"
#include "properties.h"
#include "misc.h"
#include "control.h"
#include "effects-generic.h"
#include "preprocessor.h"
#include "text-util.h"
#include "transformations.h"
#include "parser_private.h"
#include "syntax.h"
#include "c_syntax.h"

/** 
 * create a statement eligible for outlining into a kernel
 * #1 find the loop flagged with loop_label
 * #2 make sure we know thebound of the loop
 * #2' flag the kernel with a pragma
 * #3 call index set splitting on this loop to get a first loop with range_count multiple of KERNEL_NBNODES
 * #4 perform strip mining on this loop to make the kernel appear
 * #5 supress the generated loop and replace its index by the appropriate call to KERNEL_ID
 * 
 * @param s statement where the kernel can be found
 * @param loop_label label of the loop to be turned into a kernel
 * 
 * @return true as long as the kernel is not found
 */
static
bool do_kernelize(statement s, entity loop_label) 
{
    if( same_entity_p(statement_label(s),loop_label) ||
            (statement_loop_p(s) && same_entity_p(loop_label(statement_loop(s)),loop_label)))
    {
        if( !instruction_loop_p(statement_instruction(s)) )
            pips_user_error("you choosed a label of a non-doloop statement\n");

        loop l = instruction_loop(statement_instruction(s));

        /* gather and check parameters */
        int nb_nodes = get_int_property("KERNEL_NBNODES");
        while(!nb_nodes)
        {
            string ur = user_request("number of nodes for your kernel?\n");
            nb_nodes=atoi(ur);
        }

        /* verify the loop is parallel */
        if( execution_sequential_p(loop_execution(l)) )
            pips_user_error("you tried to kernelize a sequential loop\n");

        /* perform index set splitting to get the good loop range */
        int count;
        if(!range_count(loop_range(l),&count))
            pips_user_error("unable to count the number of iterations in given loop\n");
        int increment_val;
        expression_integer_value(range_increment(loop_range(l)),&increment_val);
        int split_index = increment_val*(count - count%nb_nodes) ;
        entity split_index_entity = 
            make_C_or_Fortran_constant_entity(
                    itoa(split_index),
                    is_basic_int,
                    DEFAULT_INTEGER_TYPE_SIZE,
                    fortran_module_p(get_current_module_entity())
                    );
        index_set_split_loop(s,split_index_entity);
        /* now s is a block with two loops, we are interested in the first one */

        s= STATEMENT(CAR(statement_block(s)));

        /* we can strip mine the loop */
        loop_strip_mine(s,nb_nodes,-1);
        l = statement_loop(s);

        /* it's safe to skip the second level loop, because of the index set splitting */
        statement replaced_loop = loop_body(l);
        instruction erased_instruction = statement_instruction(replaced_loop);
        entity outermost_loop_index = loop_index(instruction_loop(erased_instruction));

        entity kernel_id = FindEntity(TOP_LEVEL_MODULE_NAME,"KERNEL_ID");
        if(entity_undefined_p(kernel_id))
            pips_user_error("KERNEL_ID not defined !\n");

        instruction assign = make_assign_instruction(make_expression_from_entity(outermost_loop_index),
                MakeBinaryCall(entity_intrinsic(c_module_p(get_current_module_entity())?PLUS_C_OPERATOR_NAME:PLUS_OPERATOR_NAME),
                    MakeNullaryCall(kernel_id),make_expression_from_entity(loop_index(l))
                )
                );
        statement_instruction(replaced_loop) = 
            make_instruction_block(
                    make_statement_list(make_stmt_of_instr(assign),loop_body(instruction_loop(erased_instruction)))
                    );

        /* as the newgen free is recursive, we use a trick to prevent the recursion */
        loop_body(instruction_loop(erased_instruction)) = make_continue_statement(entity_empty_label());
        free_instruction(erased_instruction);

        /* flag the remaining loop to be proceed next*/
        extensions se = statement_extensions(s);
        extensions_extension(se)=
            CONS(EXTENSION,make_extension(make_pragma_string(strdup(concatenate(OUTLINE_IGNORE," ",entity_user_name(kernel_id))))),
                CONS(EXTENSION,make_extension(make_pragma_string(strdup(OUTLINE_PRAGMA))),extensions_extension(se))
            );

        /* job done */
        gen_recurse_stop(NULL);

    }
    return true;
}


/** 
 * turn a loop flagged with LOOP_LABEL into a kernel (GPU, terapix ...)
 * 
 * @param module_name name of the module
 * 
 * @return true
 */
bool kernelize(char * module_name)
{
    /* prelude */
    set_current_module_entity(module_name_to_entity( module_name ));
    set_current_module_statement((statement) db_get_memory_resource(DBR_CODE, module_name, TRUE) );

    /* retreive loop label */
    string loop_label_name = get_string_property_or_ask("LOOP_LABEL","label of the loop to turn into a kernel ?\n");
    entity loop_label_entity = find_label_entity(module_name,loop_label_name);
    if( entity_undefined_p(loop_label_entity) )
        pips_user_error("label '%s' not found in module '%s' \n",loop_label_name,module_name);


    /* run terapixify */
    gen_context_recurse(get_current_module_statement(),loop_label_entity,statement_domain,do_kernelize,gen_null);

    /* validate */
    module_reorder(get_current_module_statement());
    DB_PUT_MEMORY_RESOURCE(DBR_CODE, module_name,get_current_module_statement());
    DB_PUT_MEMORY_RESOURCE(DBR_CALLEES, module_name, compute_callees(get_current_module_statement()));

    /*postlude*/
    reset_current_module_entity();
    reset_current_module_statement();
    return true;
}

static
bool cannot_terapixify(gen_chunk * elem, bool *can_terapixify)
{
    pips_user_warning("found invalid construct of type %d\n",elem->i);
    return *can_terapixify=false;
}

static 
bool can_terapixify_call_p(call c, bool *can_terapixify)
{
    if( !value_intrinsic_p(entity_initial(call_function((c)))) && ! call_constant_p(c) )
    {
        pips_user_warning("found invalid call to %s\n",entity_user_name(call_function(c)));
        return *can_terapixify=false;
    }
    return true;
}

struct entity_bool { entity e; bool b; };

static
void entity_used_in_reference_walker(reference r, struct entity_bool *eb)
{
    if(same_entity_p(reference_variable(r),eb->e)) eb->b=true;
}

static
void entity_used_in_loop_bound_walker(loop l, struct entity_bool *eb)
{
    gen_context_recurse(loop_range(l),eb,reference_domain,gen_true,entity_used_in_reference_walker);
}

static
bool  entity_used_in_loop_bound_p(entity e)
{
    struct entity_bool eb = { e, false };
    gen_context_recurse(get_current_module_statement(),&eb,loop_domain,gen_true,entity_used_in_loop_bound_walker);
    return eb.b;
}

bool normalize_microcode( char * module_name)
{
    bool can_terapixify =true;
    /* prelude */
    set_current_module_entity(module_name_to_entity( module_name ));
    set_current_module_statement((statement) db_get_memory_resource(DBR_CODE, module_name, TRUE) );
    set_cumulated_rw_effects((statement_effects)db_get_memory_resource(DBR_CUMULATED_EFFECTS,module_name,TRUE));

    /* checks */

    /* make sure 
     * - only do loops remain
     * - no call to external functions
     */
    gen_context_multi_recurse(get_current_module_statement(),&can_terapixify,
            whileloop_domain,cannot_terapixify,gen_null,
            forloop_domain,cannot_terapixify,gen_null,
            call_domain,can_terapixify_call_p,gen_null,
            NULL);

    /* now, try to guess the goal of the parameters 
     * - read-only arrays might be mask, but can also be images (depend of their size ?)
     * - written arrays must be images
     * - integer are loop parameters
     * - others are not allowded
     */
    FOREACH(ENTITY,e,code_declarations(value_code(entity_initial(get_current_module_entity()))))
    {
        if(formal_parameter_p(e))
        {
            variable v = type_variable(entity_type(e));
            if( !ENDP(variable_dimensions(v)) ) /* it's an array */
            {
                bool parameter_written = find_write_effect_on_entity(get_current_module_statement(),e);
                if( parameter_written ) /* it's an image */
                {
                }
                else /* cannot tell if it's a kernel or an image*/
                {
                }
            }
            else if( entity_used_in_loop_bound_p(e) ) 
            {
            }
            else {
                pips_user_warning("parameter %s is not valid\n",entity_user_name(e));
                can_terapixify=false;
            }

        }
    }


    /*postlude*/
    reset_current_module_entity();
    reset_current_module_statement();
    reset_cumulated_rw_effects();
    return can_terapixify;
}

/** 
 * have a look to the pipsmake-rc description
 * basically call kernelize then outlining
 * 
 * @param module_name name of the module
 * 
 * @return true
 */
bool terapixify(__attribute__((unused)) char * module_name)
{
    return true; /* everything is done in pipsmake-rc */
}

/** 
 * transform each subscript in expression @a exp into the equivalent pointer arithmetic expression
 * 
 * @param exp expression to inspect
 * 
 * @return true 
 */
static
bool expression_array_to_pointer(expression exp)
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
            if( get_bool_property("ARRAY_TO_POINTER_CONVERT_PARAMETERS")
                    && formal_parameter_p(reference_variable(ref)) )
            {
                force_cast=false;
            }

            /* create a new reference without subscripts */
            reference ref_without_indices = make_reference(reference_variable(ref),NIL);
            /* get the base type of the reference */
            type type_without_indices = make_type_variable(make_variable(
                        copy_basic(variable_basic(type_variable(entity_type(reference_variable(ref))))),
                        NIL,
                        gen_full_copy_list(variable_qualifiers(type_variable(entity_type(reference_variable(ref)))))));

            expression address_computation = EXPRESSION(CAR(reference_indices(ref)));
            /* create an expression for the new reference, possibly casted */
            expression base_ref = reference_to_expression(ref_without_indices);
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

            /* iterate on the dimensions & indices to create the pointer expression */
            list dims = variable_dimensions(type_variable(entity_type(reference_variable(ref))));
            list indices = reference_indices(ref);
            POP(indices);
            if(!ENDP(dims)) POP(dims); // the first dimension is unused
            FOREACH(DIMENSION,dim,dims)
            {
                expression dimension_size = MakeBinaryCall(
                        CreateIntrinsic(PLUS_OPERATOR_NAME),
                        MakeBinaryCall(
                            CreateIntrinsic(MINUS_OPERATOR_NAME),
                            copy_expression(dimension_upper(dim)),
                            copy_expression(dimension_lower(dim))
                            ),
                        make_expression_1());

                if( !ENDP(indices) ) { /* there may be more dimensions than indices */
                    expression index_expression = EXPRESSION(CAR(indices));
                    address_computation = MakeBinaryCall(
                            CreateIntrinsic(PLUS_OPERATOR_NAME),
                            index_expression,
                            MakeBinaryCall(
                                CreateIntrinsic(MULTIPLY_OPERATOR_NAME),
                                dimension_size,address_computation
                                )
                            );
                    POP(indices);
                }
                else {
                    address_computation = MakeBinaryCall(
                            CreateIntrinsic(MULTIPLY_OPERATOR_NAME),
                            dimension_size,address_computation
                            );
                }
            }

            /* there may be more indices than dimensions */
            FOREACH(EXPRESSION,e,indices)
            {
                address_computation = MakeBinaryCall(
                        CreateIntrinsic(PLUS_OPERATOR_NAME),
                        address_computation,e
                        );
            }

            /* we now add the DEREFERENCING_OPERATOR, if needed */
            syntax new_syntax = syntax_undefined;
            if(nb_indices == nb_dims || nb_dims == 0 ) {
                new_syntax=make_syntax_call(
                        make_call(
                            CreateIntrinsic(DEREFERENCING_OPERATOR_NAME),
                            CONS(EXPRESSION,MakeBinaryCall(
                                    CreateIntrinsic(PLUS_C_OPERATOR_NAME),
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
bool declaration_array_to_pointer(statement s)
{
    FOREACH(ENTITY,e,statement_declarations(s))
        gen_recurse(entity_initial(e),expression_domain,expression_array_to_pointer,gen_null);
    return true;
}

static
void make_pointer_from_variable(variable param)
{
    list parameter_dimensions = variable_dimensions(param);
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
}
static
void make_pointer_entity_from_reference_entity(entity e)
{
    variable param = type_variable(entity_type(e));
    make_pointer_from_variable(param);
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
        gen_multi_recurse(get_current_module_statement(),
                expression_domain,expression_array_to_pointer,gen_null,
                statement_domain,declaration_array_to_pointer,gen_null,
                NULL);
        /* if this property is set, we also change the signature of the module
         * tricky : signature must be change in two places !
         */
        if( get_bool_property("ARRAY_TO_POINTER_CONVERT_PARAMETERS") )
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
                make_pointer_from_variable(type_variable(t));
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

