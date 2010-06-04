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
 * @file terapixify.c
 * apply transformations required to generate terapix microcode
 * @author Serge Guelton <serge.guelton@enst-bretagne.fr>
 * @date 2009-07-01
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
 * terapixify
 */

static
bool cannot_terapixify(gen_chunk * elem, bool *can_terapixify)
{
    printf("found invalid construct of type %td\n",elem->i);
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
    while( basic_pointer_p(b))
        b = variable_basic(type_variable(ultimate_type(basic_pointer(b))));

    if(!basic_int_p(b) && ! basic_overloaded_p(b))
    {
      list ewords = words_expression(e,NIL);
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

#define TERAPIX_PTRARG_PREFIX "FIFO"
#define TERAPIX_LOOPARG_PREFIX "N"
#define TERAPIX_IMAGE_PREFIX "im"
#define TERAPIX_MASK_PREFIX "ma"
#define TERAPIX_REGISTER_PREFIX "re"

static bool terapix_renamed_local_p(const char* s, const char* prefix)
{
    string found = strstr(s,prefix);
    if(found)
    {
        for(found+=strlen(prefix)+1;*found;++found)
            if(!isdigit(*found)) return false;
        return true;
    }
    return false;
}
static bool terapix_renamed_p(const char *s)
{
    return terapix_renamed_local_p(s,TERAPIX_PTRARG_PREFIX) || 
        terapix_renamed_local_p(s,TERAPIX_LOOPARG_PREFIX) ||
        terapix_renamed_local_p(s,TERAPIX_IMAGE_PREFIX) ||
        terapix_renamed_local_p(s,TERAPIX_MASK_PREFIX) ||
        terapix_renamed_local_p(s,TERAPIX_REGISTER_PREFIX);
}

static
void terapix_argument_handler(entity e, string arg_prefix, size_t *arg_cnt,string ass_prefix, size_t *ass_cnt)
{
    /* change parameter name and generate an assignment */
    if(arg_prefix && !terapix_renamed_p(entity_user_name(e)) ) {
        string new_name;
        asprintf(&new_name,"%s" MODULE_SEP_STRING  "%s%u",entity_module_name(e),arg_prefix,(*arg_cnt)++);
        entity ne = make_entity_copy_with_new_name(e,new_name,false);
        free(new_name);

        for(list iter = code_declarations(value_code(entity_initial(get_current_module_entity())));
                !ENDP(iter);
                POP(iter))
        {
            entity ee = ENTITY(CAR(iter));
            if(same_entity_p(e,ee)) {
                CAR(iter).p=(gen_chunkp)ne;
            }
        }
        /* we now have FIFOx in ne and will generate an assignment from ne to e 
         * we also have to change the storage for e ...
         * and for images, add a dereferencing operator
         */
        free_storage(entity_storage(e)); entity_storage(e) = storage_undefined;
        AddEntityToCurrentModule(e);
        expression assigned ;
        if( ass_prefix && same_string_p(ass_prefix,TERAPIX_IMAGE_PREFIX))
        {
            basic bt = entity_basic(e);
            type new_type = copy_type(basic_pointer(bt));
            free_type(entity_type(e));
            entity_type(e) = new_type;
            assigned= MakeUnaryCall(entity_intrinsic(DEREFERENCING_OPERATOR_NAME),entity_to_expression(ne));
            expression pattern = MakeUnaryCall(entity_intrinsic(DEREFERENCING_OPERATOR_NAME),entity_to_expression(e)),
                       substitute = entity_to_expression(e);
            substitute_expression(get_current_module_statement(),pattern, substitute);
            free_expression(pattern);free_expression(substitute);
        }
        else
            assigned = entity_to_expression(ne);
        statement ass = make_assign_statement(entity_to_expression(e),assigned);
        insert_statement(get_current_module_statement(),ass,true);
    }

    /* to respect terapix asm, we also have to change the name of variable e */
    if(ass_prefix && !terapix_renamed_p(entity_user_name(e))) {
        string new_name;
        asprintf(&new_name,"%s" MODULE_SEP_STRING "%s%u",entity_module_name(e),ass_prefix,(*ass_cnt)++);
        entity ne = make_entity_copy_with_new_name(e,new_name,false);
        AddEntityToCurrentModule(ne);
        free(new_name);
        replace_entity(get_current_module_statement(),e,ne);
    }
}

static
bool terapix_suitable_loop_bound_walker(reference r,bool *suitable)
{
    return (*suitable) &= formal_parameter_p(reference_variable(r));
}

static
bool terapix_suitable_loop_bound_p(expression exp)
{
    bool suitable=true;
    gen_context_recurse(exp,&suitable,reference_domain,terapix_suitable_loop_bound_walker,gen_null);
    return suitable;
}

typedef struct {
    hash_table ht;
    size_t *cnt;
} terapix_loop_handler_param;

static void
terapix_loop_handler(statement sl,terapix_loop_handler_param *p)
{
    if(statement_loop_p(sl)){
        loop l = statement_loop(sl);
        range r = loop_range(l);
        expression nb_iter = range_to_expression(r,range_to_nbiter);
        entity loop_bound = entity_undefined;
        if(terapix_suitable_loop_bound_p(nb_iter))
        {
            /* generate new entity if needed */
            if(expression_reference_p(nb_iter)) /* use the reference , but we must rename it however !*/
            {
                loop_bound=reference_variable(expression_reference(nb_iter));
                string new_name;
                asprintf(&new_name,"%s" MODULE_SEP_STRING TERAPIX_LOOPARG_PREFIX "%u",get_current_module_name(),(*p->cnt)++);
                entity new_loop_bound=make_entity_copy_with_new_name(loop_bound,new_name,false);
                for(list iter = code_declarations(value_code(entity_initial(get_current_module_entity())));
                        !ENDP(iter);
                        POP(iter))
                {
                    entity ee = ENTITY(CAR(iter));
                    if(same_entity_p(loop_bound,ee)) {
                        CAR(iter).p=(gen_chunkp)new_loop_bound;
                    }
                }
                replace_entity(get_current_module_statement(),loop_bound,new_loop_bound);
                loop_bound=new_loop_bound;
            }
            else {
                string new_name;
                asprintf(&new_name,TERAPIX_LOOPARG_PREFIX "%u",(*p->cnt)++);
                loop_bound=make_scalar_integer_entity(new_name,get_current_module_name());
                value v = entity_initial(loop_bound);
                free_constant(value_constant(v));
                value_tag(v)=is_value_expression;
                value_expression(v)=nb_iter;
                AddEntityToCurrentModule(loop_bound);
                free(new_name);
            }

            /* patch loop */
            free_expression(range_lower(loop_range(l)));
            range_lower(loop_range(l))=make_expression_1();
            free_expression(range_upper(loop_range(l)));
            range_upper(loop_range(l))=entity_to_expression(loop_bound);

            /* convert the loop to a while loop */
            list statements = make_statement_list(
                    copy_statement(loop_body(l)),
                    make_assign_statement(entity_to_expression(loop_index(l)),make_op_exp(PLUS_OPERATOR_NAME,entity_to_expression(loop_index(l)),make_expression_1()))
            );
            whileloop wl = make_whileloop(
                    MakeBinaryCall(entity_intrinsic(LESS_OR_EQUAL_OPERATOR_NAME),entity_to_expression(loop_index(l)),entity_to_expression(loop_bound)),
                    make_block_statement(statements),
                    entity_empty_label(),
                    make_evaluation_before());
            sequence seq = make_sequence(
                    make_statement_list(
                        make_assign_statement(entity_to_expression(loop_index(l)),make_expression_1()),
                        make_stmt_of_instr(make_instruction_whileloop(wl))
                        )
                    );

            update_statement_instruction(sl,make_instruction_sequence(seq));

            /* save change for futher processing */
            hash_put(p->ht,loop_bound,nb_iter);
        }
    }
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
     * - no float / double etc
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
    size_t nb_fifo = 0;
    size_t nb_lu = 0;
    size_t nb_ptr = 0;
    size_t nb_re = 0;
    FOREACH(ENTITY,e,code_declarations(value_code(entity_initial(get_current_module_entity()))))
    {
        if(!entity_area_p(e))
        {
            if(formal_parameter_p(e))
            {
                variable v = type_variable(entity_type(e));
                basic vb = variable_basic(v);
                if( basic_pointer_p(vb) ) /* it's a pointer */
                {
                    string prefix = NULL;
                        type t =ultimate_type(basic_pointer(vb));
                        vb=variable_basic(type_variable(t));
                        /* because of the way we build data, images are int** and masks are int* */
                        if( basic_pointer_p(vb) ) {
                            printf("%s seems an image\n",entity_user_name(e));
                            prefix = TERAPIX_IMAGE_PREFIX;
                        }
                        else {
                            printf("%s seems a mask\n",entity_user_name(e));
                            prefix = TERAPIX_MASK_PREFIX;
                        }
                        terapix_argument_handler(e,TERAPIX_PTRARG_PREFIX,&nb_fifo,prefix,&nb_ptr);
                }
                else if( entity_used_in_loop_bound_p(e) )
                {
                    printf("%s belongs to a loop bound\n",entity_user_name(e));
                    //terapix_argument_handler(e,TERAPIX_LOOPARG_PREFIX,&nb_lu,NULL,NULL);
                }
                else {
                    printf("parameter %s is not valid\n",entity_user_name(e));
                    can_terapixify=false;
                }
            }
            else {
                terapix_argument_handler(e,NULL,NULL,TERAPIX_REGISTER_PREFIX,&nb_re);
            }
        }
    }

    /* rename all declared entities using terasm convention*/
    FOREACH(ENTITY,e,statement_declarations(get_current_module_statement()))
    {
        if(entity_variable_p(e))
        {
            variable v = type_variable(entity_type(e));
            if( basic_pointer_p(variable_basic(v)) ) /* it's a pointer */
                terapix_argument_handler(e,NULL,NULL,TERAPIX_IMAGE_PREFIX,&nb_ptr);
            else if( basic_int_p(variable_basic(v))) /* it's an int */
                terapix_argument_handler(e,NULL,NULL,TERAPIX_REGISTER_PREFIX,&nb_re);
        }
    }

    /* loops in terasm iterate over a given parameter, in the form DO I=1:N 
     * I is hidden to the user and N must be a parameter */
    {
        terapix_loop_handler_param p = {
            .ht = hash_table_make(hash_pointer,HASH_DEFAULT_SIZE),
            .cnt=&nb_lu
        };
        gen_context_recurse(get_current_module_statement(),&p,statement_domain,gen_true,terapix_loop_handler);
    }


    /* validate */
    module_reorder(get_current_module_statement());
    DB_PUT_MEMORY_RESOURCE(DBR_CODE, module_name,get_current_module_statement());

    /*postlude*/
    reset_current_module_entity();
    reset_current_module_statement();
    reset_cumulated_rw_effects();
    return true || can_terapixify;
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
 * 
 * 
 * @param e 
 * 
 * @return 
 */
static bool two_addresses_code_generator_split_p(expression e )
{
    if(expression_call_p(e))
    {
        call c = expression_call(e);
        entity op = call_function(c);
        return ! call_constant_p(c) && (!get_bool_property("GENERATE_TWO_ADDRESSES_CODE_SKIP_DEREFERENCING") || !ENTITY_DEREFERENCING_P(op));
    }
    else
        return false;
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
            if(!two_addresses_code_generator_split_p(lhs) && two_addresses_code_generator_split_p(rhs)) {
                call parent_call = call_undefined;
                do {
                    parent_call=expression_call(rhs);
                    rhs=EXPRESSION(CAR(call_arguments(parent_call)));
                } while(expression_call_p(rhs) && two_addresses_code_generator_split_p(rhs));
                if(! expression_equal_p(lhs,rhs) )
                {
                    /* a=b+c; -> (1) a=b; (2) a=a+c; */
                    statement theassign/*1*/= make_assign_statement(copy_expression(lhs),copy_expression(rhs));
                    statement thecall/*2*/= s;
                    CAR(call_arguments(parent_call)).p=(gen_chunkp)copy_expression(lhs);
                    insert_statement(thecall,theassign,true);
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
}

static void do_terapix_remove_divide(call c)
{
    entity op = call_function(c);
    if(ENTITY_DIVIDE_P(op))
    {
        expression lhs = binary_call_lhs(c);
        expression rhs = binary_call_rhs(c);
        if(extended_expression_constant_p(rhs))
        {
        int accuracy = get_int_property("TERAPIX_REMOVE_DIVIDE_ACCURACY");

        gen_free_list(call_arguments(c));
        call_function(c)=entity_intrinsic(RIGHT_SHIFT_OPERATOR_NAME);

        call_arguments(c)=make_expression_list(
                MakeBinaryCall(
                    entity_intrinsic(MULTIPLY_OPERATOR_NAME),
                    lhs,
                    MakeBinaryCall(
                        entity_intrinsic(LEFT_SHIFT_OPERATOR_NAME),
                        MakeBinaryCall(
                            entity_intrinsic(DIVIDE_OPERATOR_NAME),
                            int_to_expression(1),
                            rhs),
                        int_to_expression(accuracy)
                        )
                    ),
                int_to_expression(accuracy)
                );
        }
        else
            pips_user_error("terapix cannot handle division by a non-constant variable\n");
    }
}
bool
terapix_remove_divide(const char *module_name)
{
    /* prelude */
    set_current_module_entity(module_name_to_entity( module_name ));
    set_current_module_statement((statement) db_get_memory_resource(DBR_CODE, module_name, TRUE) );

    /* converts divide operator into multiply operator:
     * a/cste = a* (1/b) ~= a * ( 128 / cste ) / 128
     */
    gen_recurse(get_current_module_statement(),call_domain,gen_true,do_terapix_remove_divide);

    /* validate */
    DB_PUT_MEMORY_RESOURCE(DBR_CODE, module_name,get_current_module_statement());

    /*postlude*/
    reset_current_module_entity();
    reset_current_module_statement();
    return true;
}
