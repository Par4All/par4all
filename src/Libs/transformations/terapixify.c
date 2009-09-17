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
#include "effects-simple.h"
#include "preprocessor.h"
#include "text-util.h"
#include "transformations.h"
#include "parser_private.h"
#include "syntax.h"
#include "c_syntax.h"

/**
 * create a statement eligible for outlining into a kernel
 * #1 find the loop flagged with loop_label
 * #2 make sure the loop is // with local index
 * #3 perform strip mining on this loop to make the kernel appear
 * #4 perform two outlining to separate kernel from host
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
        int nb_nodes = get_int_property("KERNELIZE_NBNODES");
        while(!nb_nodes)
        {
            string ur = user_request("number of nodes for your kernel?\n");
            nb_nodes=atoi(ur);
        }

        /* verify the loop is parallel */
        if( execution_sequential_p(loop_execution(l)) )
            pips_user_error("you tried to kernelize a sequential loop\n");
        if( !entity_is_argument_p(loop_index(statement_loop(s)),loop_locals(statement_loop(s))) )
            pips_user_error("you tried to kernelize a loop whose index is not private\n");

        /* we can strip mine the loop */
        loop_strip_mine(s,nb_nodes,-1);
        /* unfortunetly, the strip mining does not exactly does what we
	   want, fix it here

	   it is legal because we know the loop index is private,
	   otherwise the end value of the loop index may be used
	   incorrectly...
	*/
        {
            statement s2 = loop_body(statement_loop(s));
            entity outer_index = loop_index(statement_loop(s));
            entity inner_index = loop_index(statement_loop(s2));
            replace_entity(s2,inner_index,outer_index);
            loop_index(statement_loop(s2))=outer_index;
            replace_entity(loop_range(statement_loop(s2)),outer_index,inner_index);
            if(!ENDP(loop_locals(statement_loop(s2)))) replace_entity(loop_locals(statement_loop(s2)),outer_index,inner_index);
            loop_index(statement_loop(s))=inner_index;
            replace_entity(loop_range(statement_loop(s)),outer_index,inner_index);
            gen_remove_once(&entity_declarations(get_current_module_entity()),outer_index);
            gen_remove_once(&statement_declarations(get_current_module_statement()),outer_index);
            loop_body(statement_loop(s))=make_block_statement(make_statement_list(s2));
            AddLocalEntityToDeclarations(outer_index,get_current_module_entity(),loop_body(statement_loop(s)));
            l = statement_loop(s);
        }

        /* validate changes */
        entity cme = get_current_module_entity();
        statement cms = get_current_module_statement();
        module_reorder(get_current_module_statement());
        DB_PUT_MEMORY_RESOURCE(DBR_CODE, get_current_module_name(),get_current_module_statement());
        DB_PUT_MEMORY_RESOURCE(DBR_CALLEES, get_current_module_name(), compute_callees(get_current_module_statement()));
        reset_current_module_entity();
        reset_current_module_statement();

        /* recompute effects */
        proper_effects(module_local_name(cme));
        cumulated_effects(module_local_name(cme));
        set_current_module_entity(cme);
        set_current_module_statement(cms);
        set_cumulated_rw_effects((statement_effects)db_get_memory_resource(DBR_PROPER_EFFECTS, get_current_module_name(), TRUE));
        /* outline the work and kernel parts*/
        string kernel_name=get_string_property_or_ask("KERNELIZE_KERNEL_NAME","name of the kernel ?");
        string host_call_name=get_string_property_or_ask("KERNELIZE_HOST_CALL_NAME","name of the fucntion to call the kernel ?");
        outliner(kernel_name,make_statement_list(loop_body(l)));
        outliner(host_call_name,make_statement_list(s));
        reset_cumulated_rw_effects();



#if 0
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
#endif

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
    printf("found invalid construct of type %d\n",elem->i);
    return *can_terapixify=false;
}

static
bool can_terapixify_call_p(call c, bool *can_terapixify)
{
    if( !value_intrinsic_p(entity_initial(call_function((c)))) && ! call_constant_p(c) )
    {
        printf("found invalid call to %s\n",entity_user_name(call_function(c)));
        return *can_terapixify=false;
    }
    return true;
}

static
bool can_terapixify_expression_p(expression e, bool *can_terapixify)
{
    basic b = expression_basic(e);
    if(!basic_int_p(b) && ! basic_overloaded_p(b) )
    {
        list ewords = words_expression(e);
        string estring = words_to_string(ewords);
        string bstring = basic_to_string(b);
        printf("found invalid expression %s of basic %s\n",estring, bstring);
        free(bstring);
        free(estring);
        gen_free_list(ewords);
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
     * - no float / double etc (TODO)
     */
    gen_context_multi_recurse(get_current_module_statement(),&can_terapixify,
            whileloop_domain,cannot_terapixify,gen_null,
            forloop_domain,cannot_terapixify,gen_null,
            call_domain,can_terapixify_call_p,gen_null,
            expression_domain,can_terapixify_expression_p,gen_null,
            NULL);

    /* now, try to guess the goal of the parameters
     * - parameters are 16 bits signed integers (TODO)
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
                    printf("%s seems an image\n",entity_user_name(e));
                }
                else /* cannot tell if it's a kernel or an image*/
                {
                    int array_size=1;
                    FOREACH(DIMENSION,d,variable_dimensions(v))
                    {
                        int d_size;
                        if(SizeOfDimension(d,&d_size)) {
                            array_size*=d_size;
                        }
                        else {
                            array_size=-1;
                            break;
                        }
                    }
                    if( array_size > 0 && 56 >= array_size ) {
                        printf("%s seems a kernel\n",entity_user_name(e));
                    }
                    else {
                        printf("%s seems an image\n",entity_user_name(e));
                    }
                }
            }
            else if( entity_used_in_loop_bound_p(e) )
            {
            }
            else {
                printf("parameter %s is not valid\n",entity_user_name(e));
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
            if( get_bool_property("ARRAY_TO_POINTER_CONVERT_PARAMETERS") && ! get_bool_property("ARRAY_TO_POINTER_FLATTEN_ONLY")
                    && formal_parameter_p(reference_variable(ref)) )
            {
                force_cast=false;
            }

            /* create a new reference without subscripts */
            reference ref_without_indices = make_reference(reference_variable(ref),NIL);

            expression base_ref = reference_to_expression(ref_without_indices);
            expression address_computation = EXPRESSION(CAR(reference_indices(ref)));
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

            /* iterate on the dimensions & indices to create the index expression */
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

            /* we now either add the DEREFERENCING_OPERATOR, or the [] */
            syntax new_syntax = syntax_undefined;
            if(nb_indices == nb_dims || nb_dims == 0 ) {
                if(get_bool_property("ARRAY_TO_POINTER_FLATTEN_ONLY")) {
                    reference_indices(ref_without_indices)=make_expression_list(address_computation);
                    new_syntax=make_syntax_reference(ref_without_indices);
                }
                else {

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
            }
            else
            {
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

static
void two_addresses_code_generator(statement s)
{
    if(statement_call_p(s))
    {
        call c = statement_call(s);
        if(ENTITY_ASSIGN_P(call_function(c)))
        {
            list args = call_arguments(c);
            expression lhs = EXPRESSION(CAR(args));
            expression rhs = EXPRESSION(CAR(CDR(args)));
            if(expression_reference_p(lhs) && expression_call_p(rhs) && !expression_constant_p(rhs)) {
                do {
                    rhs=EXPRESSION(CAR(call_arguments(expression_call(rhs))));
                } while(expression_call_p(rhs) && !expression_constant_p(rhs));
                if(! expression_equal_p(lhs,rhs) )
                {
                    /* a=b+c; -> tmp=b;b=b+c;a=b;b=tmp; */
                    statement thecall/*2*/= make_stmt_of_instr(statement_instruction(s));
                    instruction theblock = make_instruction_block(NIL);
                    statement_instruction(s)=theblock;

                    if(expression_constant_p(rhs))
                    {
                        entity tmp = make_new_scalar_variable(get_current_module_entity(),copy_basic(basic_of_expression(rhs)));
                        entity_initial(tmp)=make_value_expression(rhs);
                        AddLocalEntityToDeclarations(tmp,get_current_module_entity(),s);
                        rhs=entity_to_expression(tmp);
                    }
                    entity tmp = make_new_scalar_variable(get_current_module_entity(),copy_basic(basic_of_expression(rhs)));
                    AddLocalEntityToDeclarations(tmp,get_current_module_entity(),s);
                    entity_initial(tmp)=make_value_expression(copy_expression(rhs));
                    statement copy_lhs/*3*/ = make_assign_statement(lhs,copy_expression(rhs));
                    statement copy_tmp/*4*/ = make_assign_statement(copy_expression(rhs),entity_to_expression(tmp));
                    CAR(args).p=(gen_chunkp)copy_expression(rhs);
                    instruction_block(theblock)=make_statement_list(thecall,copy_lhs,copy_tmp);
                    statement_comments(thecall)=statement_comments(s);
                    statement_label(thecall)=statement_label(s);
                    statement_number(thecall)=STATEMENT_NUMBER_UNDEFINED;

                    statement_comments(s)=empty_comments;
                    statement_ordering(s)=STATEMENT_ORDERING_UNDEFINED;
                    statement_label(s)=entity_empty_label();
                    statement_number(s)=STATEMENT_NUMBER_UNDEFINED;
                }
            }
        }
    }
}

bool
generate_two_addresses_code(char *module_name)
{
    /* prelude */
    set_current_module_entity(module_name_to_entity( module_name ));
    set_current_module_statement((statement) db_get_memory_resource(DBR_CODE, module_name, TRUE) );

    gen_recurse(get_current_module_statement(),statement_domain,gen_true,two_addresses_code_generator);

    /* validate */
    module_reorder(get_current_module_statement());
    DB_PUT_MEMORY_RESOURCE(DBR_CODE, module_name,get_current_module_statement());

    /*postlude*/
    reset_current_module_entity();
    reset_current_module_statement();
    return true;
    return true;
}

