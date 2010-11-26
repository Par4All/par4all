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
#include "pipsmake.h"
#include "resources.h"
#include "properties.h"
#include "misc.h"
#include "control.h"
#include "effects-generic.h"
#include "effects-simple.h"
#include "effects-convex.h"
#include "expressions.h"
#include "callgraph.h"
#include "text-util.h"
#include "parser_private.h"
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
        for(found+=strlen(prefix);*found;++found)
            if(!isdigit(*found)) return false;
        return true;
    }
    return false;
}

static bool terapix_renamed_entity_p(entity e, const char* prefix) {
    return terapix_renamed_local_p(entity_local_name(e),prefix);
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
        asprintf(&new_name,"%s" MODULE_SEP_STRING  "%s%zd",entity_module_name(e),arg_prefix,(*arg_cnt)++);
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
        if( false && ass_prefix && same_string_p(ass_prefix,TERAPIX_IMAGE_PREFIX))
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
        asprintf(&new_name,"%s" MODULE_SEP_STRING "%s%zd",entity_module_name(e),ass_prefix,(*ass_cnt)++);
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
            set body_entities = get_referenced_entities(loop_body(l));
            /* generate new entity if needed */
            if(expression_reference_p(nb_iter)) /* use the reference , but we must rename it however !*/
            {
                loop_bound=reference_variable(expression_reference(nb_iter));
                string new_name;
                asprintf(&new_name,"%s" MODULE_SEP_STRING TERAPIX_LOOPARG_PREFIX "%zd",get_current_module_name(),(*p->cnt)++);
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
                asprintf(&new_name,"%s" MODULE_SEP_STRING TERAPIX_LOOPARG_PREFIX "%zd",get_current_module_name(),(*p->cnt)++);
                loop_bound=make_scalar_integer_entity(new_name,get_current_module_name());
                value v = entity_initial(loop_bound);
                free_constant(value_constant(v));
                value_tag(v)=is_value_expression;
                value_expression(v)=nb_iter;
                AddEntityToCurrentModule(loop_bound);
                free(new_name);
            }

            /* patch loop */
            free_expression(range_upper(loop_range(l)));
            range_upper(loop_range(l))=entity_to_expression(loop_bound);

            if(set_belong_p(body_entities,loop_bound))
                do_loop_to_while_loop(sl);


            /* save change for futher processing */
            hash_put(p->ht,loop_bound,nb_iter);
        }
    }
}

static int compare_formal_parameters(const void *v0, const void * v1) {
    const entity f0 = *(const entity *)v0,
          f1=*(const entity *)v1;
    intptr_t o0 = formal_offset(storage_formal(entity_storage(f0))),
             o1 = formal_offset(storage_formal(entity_storage(f1)));
    entity e0 = find_ith_formal_parameter(get_current_module_entity(),o0),
           e1 = find_ith_formal_parameter(get_current_module_entity(),o1);
    if(f0==f1) return 0;
    if(!terapix_renamed_entity_p(e0,TERAPIX_LOOPARG_PREFIX)) {
        if(terapix_renamed_entity_p(e1,TERAPIX_LOOPARG_PREFIX))
            return 1;
        else
            return o0 > o1 ? -1 : 1 ;
    }
    else
        return o0 > o1 ? -1 : 1 ;
}

static void normalize_microcode_parameter_orders(entity module) {
    sort_parameters(module,compare_formal_parameters);
}

bool normalize_microcode( char * module_name)
{
    bool can_terapixify =true;
    /* prelude */
    set_current_module_entity(module_name_to_entity( module_name ));
    set_current_module_statement((statement) db_get_memory_resource(DBR_CODE, module_name, true) );
    set_cumulated_rw_effects((statement_effects)db_get_memory_resource(DBR_CUMULATED_EFFECTS,module_name,true));

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
    FOREACH(ENTITY,e,entity_declarations(get_current_module_entity()))
    {
        if(!entity_area_p(e))
        {
            variable v = type_variable(entity_type(e));
            basic vb = variable_basic(v);
            if(formal_parameter_p(e))
            {
                if( basic_pointer_p(vb) ) /* it's a pointer */
                {
                    string prefix = NULL;
                    type t =ultimate_type(basic_pointer(vb));
                    vb=variable_basic(type_variable(t));
                    if( strstr(entity_user_name(e),get_string_property("GROUP_CONSTANTS_HOLDER")) ) {
                        printf("%s seems a mask\n",entity_user_name(e));
                        prefix = TERAPIX_MASK_PREFIX;
                    }
                    else {
                        printf("%s seems an image\n",entity_user_name(e));
                        prefix = TERAPIX_IMAGE_PREFIX;
                    }
                    terapix_argument_handler(e,TERAPIX_PTRARG_PREFIX,&nb_fifo,prefix,&nb_ptr);
                }
                else if( entity_used_in_loop_bound_p(e) )
                {
                    printf("%s belongs to a loop bound\n",entity_user_name(e));
                    //terapix_argument_handler(e,TERAPIX_LOOPARG_PREFIX,&nb_lu,NULL,NULL);
                }
                /* a rom array with only one element, outlining and isolate_statement where too smart :) */
                else if ( strstr(entity_user_name(e),get_string_property("GROUP_CONSTANTS_HOLDER")) &&
                        entity_scalar_p(e)) {

                    entity_type(e) = make_type_variable(
                            make_variable(
                                make_basic_pointer(entity_type(e)),
                                NIL,
                                NIL
                                )
                            );
                    expression repl = MakeUnaryCall(
                            entity_intrinsic(DEREFERENCING_OPERATOR_NAME),
                            entity_to_expression(e)
                            );
                    replace_entity_by_expression(
                            get_current_module_statement(),
                            e,
                            repl);
                    free_expression(repl);
                    /* pips bonus step: the consistency */
                    intptr_t i=1,
                             offset = formal_offset(storage_formal(entity_storage(e)));
                    FOREACH(PARAMETER,p,module_functional_parameters(get_current_module_entity())) {
                        if(i++==offset) {
                            dummy d = parameter_dummy(p);
                            if(dummy_identifier_p(d))
                            {
                                entity di = dummy_identifier(d);
                                entity_type(di) = make_type_variable(
                                        make_variable(
                                            make_basic_pointer(entity_type(di)),
                                            NIL,
                                            NIL
                                            )
                                        );
                            }
                            parameter_type(p) = make_type_variable(
                                    make_variable(
                                        make_basic_pointer(parameter_type(p)),
                                        NIL,
                                        NIL
                                        )
                                    );
                            break;
                        }
                    }
                    list callers = callees_callees((callees)db_get_memory_resource(DBR_CALLERS,get_current_module_name(), true));
                    list callers_statement = callers_to_statements(callers);
                    list call_sites = callers_to_call_sites(callers_statement,get_current_module_entity());
                    pips_assert("only one caller here\n",
                            !ENDP(call_sites) && ENDP(CDR(call_sites)));
                    list args = call_arguments(CALL(CAR(call_sites)));
                    for(intptr_t i=1;i<offset;i++) POP(args);
                    expression *exp = (expression*)REFCAR(args);
                    *exp=
                        MakeUnaryCall(
                                entity_intrinsic(ADDRESS_OF_OPERATOR_NAME),
                                *exp
                                );


                    for(list citer=callers,siter=callers_statement;!ENDP(citer);POP(citer),POP(siter))
                        DB_PUT_MEMORY_RESOURCE(DBR_CODE, STRING(CAR(citer)),STATEMENT(CAR(siter)));
                    db_touch_resource(DBR_CODE,compilation_unit_of_module(get_current_module_name()));
                    gen_free_list(callers_statement);

                    printf("%s seems a mask\n",entity_user_name(e));
                    terapix_argument_handler(e,TERAPIX_PTRARG_PREFIX,&nb_fifo,TERAPIX_MASK_PREFIX,&nb_ptr);
                }
            }
            else if( basic_pointer_p(vb) ) /* it's a pointer */
                terapix_argument_handler(e,NULL,NULL,TERAPIX_IMAGE_PREFIX,&nb_ptr);
            else if( entity_scalar_p(e))
                terapix_argument_handler(e,NULL,NULL,TERAPIX_REGISTER_PREFIX,&nb_re);
            else
                terapix_argument_handler(e,TERAPIX_PTRARG_PREFIX,&nb_fifo,TERAPIX_MASK_PREFIX,&nb_ptr);
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

    /* reorder arguments to match terapix conventions */
    normalize_microcode_parameter_orders(get_current_module_entity());

    /* loops in terasm iterate over a given parameter, in the form DO I=1:N 
     * I is hidden to the user and N must be a parameter */
    {
        terapix_loop_handler_param p = {
            .ht = hash_table_make(hash_pointer,HASH_DEFAULT_SIZE),
            .cnt=&nb_lu
        };
        gen_context_recurse(get_current_module_statement(),&p,statement_domain,gen_true,terapix_loop_handler);
    }

    gen_recurse(get_current_module_statement(),statement_domain,gen_true,statement_remove_extensions);
    gen_recurse(get_current_module_statement(),statement_domain,gen_true,statement_remove_useless_label);


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
    set_current_module_statement((statement) db_get_memory_resource(DBR_CODE, module_name, true) );

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
    set_current_module_statement((statement) db_get_memory_resource(DBR_CODE, module_name, true) );

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

