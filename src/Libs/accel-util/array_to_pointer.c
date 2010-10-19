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
 * @file linearize_array.c
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
#include "effects.h"
#include "effects-util.h"
#include "pipsdbm.h"
#include "pipsmake.h"
#include "resources.h"
#include "properties.h"
#include "misc.h"
#include "control.h"
#include "expressions.h"
#include "accel-util.h"


size_t type_dereferencement_depth(type t) {
    t = ultimate_type(t);
    if(type_variable(t)) {
        variable v = type_variable(t);
        basic b = variable_basic(v);
        return  gen_length(variable_dimensions(v)) +
            (basic_pointer_p(b) ? 1+ type_dereferencement_depth(basic_pointer(b)) : 0) ;
    }
    return 0;
}



static void gather_call_sites(call c, list * sites)
{
    if(same_entity_p(call_function(c),get_current_module_entity()))
        *sites=CONS(CALL,c,*sites);
}
static void gather_call_sites_in_block(statement s, list * sites) {
    if(declaration_statement_p(s)) {
        FOREACH(ENTITY,e,statement_declarations(s)) {
            gen_context_recurse(entity_initial(e),sites,call_domain,gen_true,gather_call_sites);
        }
    }
}

static list callers_to_call_sites(list callers_statement)
{
    list call_sites = NIL;
    FOREACH(STATEMENT,caller_statement,callers_statement)
        gen_context_multi_recurse(caller_statement,&call_sites,
                statement_domain,gen_true,gather_call_sites_in_block,
                call_domain,gen_true,gather_call_sites,0);
    return call_sites;
}
static list callers_to_statements(list callers)
{
    list statements = NIL;
    FOREACH(STRING,caller_name,callers)
    {
        statement caller_statement=(statement) db_get_memory_resource(DBR_CODE,caller_name,true);
        statements=CONS(STATEMENT,caller_statement,statements);
    }
    return statements;
}

static void do_linearize_array_reference(reference r) {
    entity e =reference_variable(r);
    list indices = reference_indices(r);
    if(!ENDP(indices)) {
        type et = ultimate_type(entity_type(e));
        list new_indices = NIL;
        while(!ENDP(indices)) {
            expression new_index = expression_undefined;
            variable v = type_variable(et);
            /* must first check the dimensions , then the pointer type */
            list vdims = variable_dimensions(v);
            /* one dimension variable, nothing to do */
            if(ENDP(vdims)||ENDP(CDR(vdims))) {
            }
            else {
                /* merge all */
                new_index=int_to_expression(0);/* better start with this than nothing */
                while(!ENDP(vdims) && !ENDP(indices) ) {
                    expression curr_exp = EXPRESSION(CAR(indices));
                    new_index=make_op_exp(PLUS_OPERATOR_NAME,
                            make_op_exp(MULTIPLY_OPERATOR_NAME,
                                copy_expression(curr_exp),
                                SizeOfDimensions(CDR(vdims))
                                ),
                            new_index
                            );
                    POP(vdims);
                    POP(indices);
                }
            }
            /* it's a pointer: pop type */
            if(basic_pointer_p(variable_basic(v))) {
                et = basic_pointer(variable_basic(v));
            }
            if(expression_undefined_p(new_index)) {
                new_index =copy_expression(EXPRESSION(CAR(indices)));
                POP(indices);
            }
            new_indices=CONS(EXPRESSION,new_index,new_indices);
        }
        reference_indices(r)=gen_nreverse(new_indices);
    }
}

static void do_linearize_array_subscript(subscript s) {
    pips_user_warning("subscript linearization not handled yet\n");
}

static bool do_linearize_type(type *t, bool *rr) {
    bool linearized =false;
    if(rr)*rr=false;
    variable v = type_variable(*t);
    type ut = ultimate_type(*t);
    variable uv = type_variable(ut);
    size_t uvl = gen_length(variable_dimensions(uv));
    size_t vl = gen_length(variable_dimensions(v));
    if(uvl > 1 ) {
        dimension nd = make_dimension(
                int_to_expression(0),
                make_op_exp(MINUS_OPERATOR_NAME,
                    SizeOfDimensions(variable_dimensions(uv)),
                    int_to_expression(1))
                );
        type nt = copy_type(uvl>vl?ut:*t);
        variable nv = type_variable(nt);
        gen_full_free_list(variable_dimensions(nv));
        variable_dimensions(nv)=CONS(DIMENSION,nd,NIL);
        free_type(*t);
        *t=nt;
        linearized=true;
        if(rr)*rr=true;
    }
    if(basic_pointer_p(variable_basic(type_variable(*t))))
        return do_linearize_type(&basic_pointer(variable_basic(type_variable(*t))),rr) || linearized;
    return linearized;
}

static void do_array_to_pointer_type_aux(type *t) {
    variable v = type_variable(*t);
    if(basic_pointer_p(variable_basic(v)))
        do_array_to_pointer_type_aux(&basic_pointer(variable_basic(v)));
    list dimensions = variable_dimensions(v);
    variable_dimensions(v)=NIL;
    FOREACH(DIMENSION,d,dimensions) {
        *t=make_type_variable(
                make_variable(
                    make_basic_pointer(*t),
                    NIL,NIL
                    )
                );
    }
}
/* returns true if a dereferencment has been supressed */
static bool do_array_to_pointer_type(type *t) {
    bool remove = false;
    if(pointer_type_p(*t)){
        variable vt = type_variable(*t);
        basic bt = variable_basic(vt);
        type t2 = basic_pointer(bt);
        if(array_type_p(t2)) {
            basic_pointer(bt) = type_undefined;
            free_type(*t);
            *t=t2;
            remove=true;
        }
    }
    do_array_to_pointer_type_aux(t);
    return remove;
}


static void do_linearize_array_manage_callers(entity m,set linearized_param) {
    list callers = callees_callees((callees)db_get_memory_resource(DBR_CALLERS,module_local_name(m), true));
    list callers_statement = callers_to_statements(callers);
    list call_sites = callers_to_call_sites(callers_statement);

    /* we may have to change the call sites, prepare iterators over call sites arguments here */
    FOREACH(CALL,c,call_sites) {
        list args = call_arguments(c);
        FOREACH(PARAMETER,p,functional_parameters(type_functional(entity_type(m)))) {
            if(set_belong_p(linearized_param,p)) {
                expression * arg = (expression*)REFCAR(args);
                type t = expression_to_type(*arg);
                type t2 = parameter_type(p);
                if(!pointer_type_p(t)) {
                    type t = make_type_variable(
                            make_variable(
                                make_basic_pointer(
                                    copy_type(parameter_type(p))
                                    ),
                                NIL,NIL)
                            );
                    *arg = 
                        MakeUnaryCall(
                                entity_intrinsic(DEREFERENCING_OPERATOR_NAME),
                                make_expression(
                                    make_syntax_cast(
                                        make_cast(
                                            t,
                                            MakeUnaryCall(
                                                entity_intrinsic(ADDRESS_OF_OPERATOR_NAME),
                                                *arg
                                                )
                                            )
                                        ),
                                    normalized_undefined
                                    )
                                );
                }
                else if(!type_equal_p(t,t2)) {
                    *arg =
                        MakeUnaryCall(
                                entity_intrinsic(DEREFERENCING_OPERATOR_NAME),*arg);
                }
                free_type(t);
            }
            POP(args);
        }
    }
    for(list citer=callers,siter=callers_statement;!ENDP(citer);POP(citer),POP(siter))
        DB_PUT_MEMORY_RESOURCE(DBR_CODE, STRING(CAR(citer)),STATEMENT(CAR(siter)));

}
static void do_linearize_array_cast(cast c) {
    do_linearize_type(&cast_type(c),NULL);
}
static void do_linearize_array_walker(void* obj) {
    gen_multi_recurse(obj,
            reference_domain,gen_true,do_linearize_array_reference,
            subscript_domain,gen_true,do_linearize_array_subscript,
            cast_domain,gen_true,do_linearize_array_cast,
            NULL);
}

static void do_linearize_expression_is_pointer(expression exp, hash_table ht) {
    basic b = basic_of_expression(exp);
    hash_put(ht,exp,(void*)(intptr_t)basic_pointer_p(b));
    free_basic(b);
}
static void do_linearize_pointer_is_expression(expression exp, hash_table ht) {
    intptr_t t = (intptr_t)hash_get(ht,exp);
    if(t != (intptr_t)HASH_UNDEFINED_VALUE ) {
        basic b = basic_of_expression(exp);
        /*SG: let us hope that by fixing only references, it will be enough */
        if(t && !basic_pointer_p(b) && expression_reference_p(exp)){
            syntax syn = expression_syntax(exp);
            expression_syntax(exp) = syntax_undefined;
            update_expression_syntax(exp,
                    make_syntax_call(
                        make_call(
                            entity_intrinsic(ADDRESS_OF_OPERATOR_NAME),
                            CONS(EXPRESSION,
                                make_expression(syn,normalized_undefined),
                                NIL)
                            )
                        )
                    );
        }
        free_basic(b);
    }
}

static hash_table init_expression_is_pointer(void* obj) {
    hash_table ht = hash_table_make(hash_int,HASH_DEFAULT_SIZE);
    gen_context_recurse(obj,ht,expression_domain,gen_true,do_linearize_expression_is_pointer);
    FOREACH(ENTITY,e,entity_declarations(get_current_module_entity())) {
        if(entity_variable_p(e))
            gen_context_recurse(entity_initial(e),ht,expression_domain,gen_true,do_linearize_expression_is_pointer);
    }
    return ht;
}

static void do_linearize_patch_expressions(void* obj, hash_table ht) {
    gen_context_recurse(obj,ht,expression_domain,gen_true,do_linearize_pointer_is_expression);
    FOREACH(ENTITY,e,entity_declarations(get_current_module_entity())) {
        if(entity_variable_p(e))
            gen_context_recurse(entity_initial(e),ht,expression_domain,gen_true,do_linearize_pointer_is_expression);
    }
}

static void do_linearize_array_init(value v) {
    if(value_expression_p(v)) {
        expression exp = value_expression(v);
        if(expression_call_p(exp)) {
            call c = expression_call(exp);
            entity op = call_function(c);
            if(ENTITY_BRACE_INTRINSIC_P(op)) {
                list inits = NIL;
                for(list iter = call_arguments(c); !ENDP(iter) ; POP(iter)) {
                    expression *eiter = (expression*)REFCAR(iter);
                    if(expression_call_p(*eiter)) {
                        call c2 = expression_call(*eiter);
                        if(ENTITY_BRACE_INTRINSIC_P(call_function(c2))) {
                            iter=gen_append(iter,call_arguments(c2));
                            call_arguments(c2)=NIL;
                            continue;
                        }
                    }
                    inits=CONS(EXPRESSION,copy_expression(*eiter),inits);
                }
                inits=gen_nreverse(inits);
                gen_full_free_list(call_arguments(c));
                call_arguments(c)=inits;
            }
        }
    }
}

static void do_linearize_remove_dereferencment_walker(expression exp, entity e) {
    if(expression_call_p(exp)) {
        call c = expression_call(exp);
        if(ENTITY_DEREFERENCING_P(call_function(c))) {
            expression arg = EXPRESSION(CAR(call_arguments(c)));
            if(expression_reference_p(arg)) {
                reference r = expression_reference(arg);
                if(same_entity_p(reference_variable(r),e)) {
                    syntax syn = expression_syntax(arg);
                    expression_syntax(arg)=syntax_undefined;
                    update_expression_syntax(exp,syn);
                }
            }
            else if(expression_call_p(arg)) {
                call c2 = expression_call(arg);
                if(ENTITY_PLUS_C_P(call_function(c2))) {
                    bool remove =false;
                    FOREACH(EXPRESSION,exp2,call_arguments(c2)) {
                        if(expression_reference_p(exp2))
                        remove|=same_entity_p(reference_variable(expression_reference(exp2)),e);
                    }
                    if(remove) {
                        syntax syn = expression_syntax(arg);
                        expression_syntax(arg)=syntax_undefined;
                        update_expression_syntax(exp,syn);
                    }
                }

            }
        }
    }
}
static void do_linearize_remove_dereferencment(statement s, entity e) {
    gen_context_recurse(s,e,expression_domain,gen_true,do_linearize_remove_dereferencment_walker);
    FOREACH(ENTITY,e,entity_declarations(get_current_module_entity()))
        if(entity_variable_p(e))
            gen_context_recurse(entity_initial(e),e,expression_domain,gen_true,do_linearize_remove_dereferencment_walker);
}

static void do_linearize_prepatch_type(type t) {
    if(pointer_type_p(t)) {
        type t2 = basic_pointer(variable_basic(type_variable(t)));
        type t3 = ultimate_type(t2);
        if(array_type_p(t2)) {
            variable v = type_variable(t2);
            variable_dimensions(v)=CONS(DIMENSION,
                    make_dimension(int_to_expression(0),int_to_expression(0)),
                    variable_dimensions(v));
            basic_pointer(variable_basic(type_variable(t)))=type_undefined;
            free_variable(type_variable(t));
            type_variable(t)=v;
        }
        else if(array_type_p(t3)) {
            variable v = copy_variable(type_variable(t2));
            variable_dimensions(v)=CONS(DIMENSION,
                    make_dimension(int_to_expression(0),int_to_expression(0)),
                    variable_dimensions(v));
            free_variable(type_variable(t));
            type_variable(t)=v;
        }
    }
}

static void do_linearize_prepatch(entity m,statement s) {
    FOREACH(ENTITY,e,entity_declarations(m))
        if(entity_variable_p(e)&&formal_parameter_p(e))
            do_linearize_prepatch_type(entity_type(e));
    FOREACH(PARAMETER,p,functional_parameters(type_functional(entity_type(m)))) {
        dummy d = parameter_dummy(p);
        if(dummy_identifier_p(d))
        {
            entity di = dummy_identifier(d);
            do_linearize_prepatch_type(entity_type(di));
        }
        do_linearize_prepatch_type(parameter_type(p));
        pips_assert("everything went well",parameter_consistent_p(p));
    }
}

static void do_linearize_array(entity m, statement s) {
    /* step 0: remind all expressions types */
    hash_table e2t = init_expression_is_pointer(s);

    /* step 0.5: transform int (*a) [3] into int a[*][3] */
    do_linearize_prepatch(m,s);

    /* step1: the statements */
    do_linearize_array_walker(s);
    FOREACH(ENTITY,e,entity_declarations(m))
        if(entity_variable_p(e))
            do_linearize_array_walker(entity_initial(e));

    /* step2: the declarations */
    FOREACH(ENTITY,e,entity_declarations(m))
        if(entity_variable_p(e)) {
            bool rr;
            do_linearize_type(&entity_type(e),&rr);
            if(rr) do_linearize_remove_dereferencment(s,e);
            do_linearize_array_init(entity_initial(e));
        }

    /* pips bonus step: the consistency */
    set linearized_param = set_make(set_pointer);
    FOREACH(PARAMETER,p,functional_parameters(type_functional(entity_type(m)))) {
        dummy d = parameter_dummy(p);
        if(dummy_identifier_p(d))
        {
            entity di = dummy_identifier(d);
            do_linearize_type(&entity_type(di),NULL);
        }
        if(do_linearize_type(&parameter_type(p),NULL))
            set_add_element(linearized_param,linearized_param,p);
        pips_assert("everything went well",parameter_consistent_p(p));
    }

    /* step3: change the caller to reflect the new types accordinlgy */
    if(!get_bool_property("LINEARIZE_ARRAY_USE_POINTERS"))
        do_linearize_array_manage_callers(m,linearized_param);
    set_free(linearized_param);

    /* final step: fix expressions if we have disturbed typing in the process */
    do_linearize_patch_expressions(s,e2t);
    hash_table_free(e2t);
}

static void do_array_to_pointer_walk_expression(expression exp) {
    if(expression_reference_p(exp)) {
        reference r = expression_reference(exp);
        entity e =reference_variable(r);
        list indices = reference_indices(r);
        if(!ENDP(indices)) {
            expression new_expression = expression_undefined;
            reference_indices(r)=NIL;
            indices=gen_nreverse(indices);
            FOREACH(EXPRESSION,index,indices) {
                new_expression=MakeUnaryCall(
                        entity_intrinsic(DEREFERENCING_OPERATOR_NAME),
                        MakeBinaryCall(
                            entity_intrinsic(PLUS_C_OPERATOR_NAME),
                            expression_undefined_p(new_expression)?
                            entity_to_expression(e):
                            new_expression,
                            index
                            )
                        );
            }
            syntax syn = expression_syntax(new_expression);
            expression_syntax(new_expression)=syntax_undefined;
            update_expression_syntax(exp,syn);
        }
    }
    else if(syntax_subscript(expression_syntax(exp))) {
        pips_user_warning("subscript are not well handled (yet)!\n");
    }
}

static void do_array_to_pointer_patch_call_expression(expression exp) {
    if(expression_call_p(exp)) {
        call c = expression_call(exp);
        entity op = call_function(c);
        if(ENTITY_ADDRESS_OF_P(op)) {
            expression arg = EXPRESSION(CAR(call_arguments(c)));
            if(expression_call_p(arg)) {
                call c2 = expression_call(arg);
                if(ENTITY_DEREFERENCING_P(call_function(c2))) {
                    syntax syn = expression_syntax( EXPRESSION(CAR(call_arguments(c2))) );
                    expression_syntax( EXPRESSION(CAR(call_arguments(c2))) )=syntax_undefined;
                    update_expression_syntax(exp,syn);
                }
            }
        }
    }
}

static void do_array_to_pointer_walk_cast(cast ct){
    do_array_to_pointer_type(&cast_type(ct));
}

static void do_array_to_pointer_walker(void *obj) {
    gen_multi_recurse(obj,
            expression_domain,gen_true,do_array_to_pointer_walk_expression,
            cast_domain,gen_true,do_array_to_pointer_walk_cast,
            NULL);
    gen_recurse(obj,expression_domain,gen_true,do_array_to_pointer_patch_call_expression);

}

static
list initialization_list_to_statements(entity e) {
    list stats = NIL;
    if(entity_array_p(e)) {
        value v = entity_initial(e);
        if(value_expression_p(v)) {
            expression exp = value_expression(v);
            if(expression_call_p(exp)) {
                call c = expression_call(exp);
                entity op = call_function(c);
                /* we assume that we only have one level of braces, linearize_array should have done the previous job 
                 * incomplete type are not handled ...
                 * */
                if(ENTITY_BRACE_INTRINSIC_P(op)) {
                    expression i = copy_expression(dimension_lower(DIMENSION(CAR(variable_dimensions(type_variable(ultimate_type(entity_type(e))))))));
                    FOREACH(EXPRESSION,exp,call_arguments(c)) {
                        stats=CONS(STATEMENT,
                                make_assign_statement(
                                    reference_to_expression(
                                    make_reference(
                                        e,
                                        CONS(EXPRESSION,i,NIL)
                                        )
                                    ),
                                    make_expression(
                                        expression_syntax(exp),
                                        normalized_undefined
                                        )
                                    ),
                                stats);
                        expression_syntax(exp)=syntax_undefined;
                        i=make_op_exp(PLUS_OPERATOR_NAME,
                                copy_expression(i),
                                int_to_expression(1)
                                );
                    }
                }
            }
        }
        /* use alloca when converting array to pointers, to make sure everything is initialized correctly */
        free_value(entity_initial(e));
        entity_initial(e) = make_value_expression(
                MakeUnaryCall(
                    entity_intrinsic(ALLOCA_FUNCTION_NAME),
                    make_expression(
                        make_syntax_sizeofexpression(
                            make_sizeofexpression_type(
                                copy_type(entity_type(e))
                                )
                            ),
                        normalized_undefined
                        )
                    )
                );
    }
    return gen_nreverse(stats);
}
static void insert_statements_after_declarations(statement st, list stats) {
    if(!ENDP(stats)) {
        if(ENDP(statement_declarations(st))) {
            insert_statement(st,make_block_statement(stats),true);
        }
        else {
            for(list iter=statement_block(st),prev=NIL;!ENDP(iter);POP(iter)) {
                if(declaration_statement_p(STATEMENT(CAR(iter)))) prev=iter;
                else {
                    CDR(prev)=stats;
                    while(!ENDP(CDR(stats))) POP(stats);
                    CDR(stats)=iter;
                    break;
                }
            }
        }

    }
}

/* transform each array type in module @p m with statement @p s */
static void do_array_to_pointer(entity m, statement s) {
    /* step1: the statements */
    do_array_to_pointer_walker(s);
    FOREACH(ENTITY,e,entity_declarations(m))
        if(entity_variable_p(e))
            do_array_to_pointer_walker(entity_initial(e));

    /* step2: the declarations */
    list inits = NIL;
    FOREACH(ENTITY,e,entity_declarations(m))
        if(entity_variable_p(e)) {
            // must do this before the type conversion
            inits=gen_append(inits,initialization_list_to_statements(e));
            if(do_array_to_pointer_type(&entity_type(e)))
                do_linearize_remove_dereferencment(s,e);
        }
    /* step3: insert the intialization statement just after declarations */
    insert_statements_after_declarations(get_current_module_statement(),inits);

    /* pips bonus step: the consistency */
    FOREACH(PARAMETER,p,functional_parameters(type_functional(entity_type(m)))) {
        dummy d = parameter_dummy(p);
        if(dummy_identifier_p(d))
        {
            entity di = dummy_identifier(d);
            do_array_to_pointer_type(&entity_type(di));
        }
        do_array_to_pointer_type(&parameter_type(p));
        pips_assert("everything went well",parameter_consistent_p(p));
    }

}


/* linearize cceeses to an array, and use pointers if asked to */
bool linearize_array(char *module_name)
{
    debug_on("LINEARIZE_ARRAY_DEBUG_LEVEL");
    /* prelude */
    set_current_module_entity(module_name_to_entity( module_name ));
    if(!compilation_unit_entity_p(get_current_module_entity())) {
        set_current_module_statement((statement) db_get_memory_resource(DBR_CODE, module_name, true) );

        /* just linearize accesses and change signature from n-D arrays to 1-D arrays */
        do_linearize_array(get_current_module_entity(),get_current_module_statement());
        if(get_bool_property("LINEARIZE_ARRAY_USE_POINTERS"))
            do_array_to_pointer(get_current_module_entity(),get_current_module_statement());
        cleanup_subscripts(get_current_module_statement());
        pips_assert("everything went well",statement_consistent_p(get_current_module_statement()));
        module_reorder(get_current_module_statement());
        DB_PUT_MEMORY_RESOURCE(DBR_CODE, module_name,get_current_module_statement());
        db_touch_resource(DBR_CODE,compilation_unit_of_module(module_name));


        /*postlude*/
        reset_current_module_statement();
    }
    reset_current_module_entity();
    debug_off();
    return true;
}

