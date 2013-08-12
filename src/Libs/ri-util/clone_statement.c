/*

  $Id$

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
#ifdef HAVE_CONFIG_H
    #include "pips_config.h"
#endif

#include "linear.h"

#include "genC.h"
#include "misc.h"
#include "ri.h"
#include "cloning.h"

#include "ri-util.h"


/* forward declarations */
static expression do_clone_expression(expression e, clone_context cc, hash_table ht);
static call do_clone_call(call c, clone_context cc, hash_table ht);
static statement do_clone_statement(statement s, clone_context cc, hash_table ht);
static entity do_clone_label(entity l, clone_context cc/*, hash_table ht*/);
/*end formard declarations */

static gen_chunk*
do_clone_reference(reference r, clone_context cc, hash_table ht)
{
    entity new_entity = entity_undefined;
    if( (new_entity = hash_get(ht,entity_name(reference_variable(r)))) == HASH_UNDEFINED_VALUE )
    {
        new_entity = reference_variable(r);
        if( entity_constant_p(new_entity) )
        {
            return (gen_chunk*)make_call(new_entity,NIL);
        }
    }
    list new_indices = NIL;
    FOREACH(EXPRESSION,e,reference_indices(r))
        new_indices=CONS(EXPRESSION,do_clone_expression(e,cc, ht),new_indices);
    return (gen_chunk*)make_reference(new_entity,gen_nreverse(new_indices));
}

static range
do_clone_range(range r, clone_context cc, hash_table ht)
{
    return make_range(
            do_clone_expression(range_lower(r),cc, ht),
            do_clone_expression(range_upper(r),cc, ht),
            do_clone_expression(range_increment(r),cc, ht));
}
static cast
do_clone_cast(cast c, clone_context cc, hash_table ht)
{
    return make_cast(
            copy_type(cast_type(c)),
            do_clone_expression(cast_expression(c),cc, ht));
}

static sizeofexpression
do_clone_sizeofexpression(sizeofexpression s, clone_context cc, hash_table ht)
{
    switch(sizeofexpression_tag(s))
    {
        case is_sizeofexpression_type:
            return copy_sizeofexpression(s);
        case is_sizeofexpression_expression:
            return make_sizeofexpression_expression(do_clone_expression(sizeofexpression_expression(s),cc, ht));
    };
    return sizeofexpression_undefined;
}

static subscript
do_clone_subscript(subscript s, clone_context cc, hash_table ht)
{
    list new_indices = NIL;
    FOREACH(EXPRESSION,e,subscript_indices(s))
        new_indices=CONS(EXPRESSION,do_clone_expression(e,cc, ht),new_indices);
    return make_subscript(do_clone_expression(subscript_array(s),cc, ht),gen_nreverse(new_indices));
}

static application
do_clone_application(application a, clone_context cc, hash_table ht)
{
    list new_arguments = NIL;
    FOREACH(EXPRESSION,e,application_arguments(a))
        new_arguments=CONS(EXPRESSION,do_clone_expression(e,cc, ht),new_arguments);
    return make_application(do_clone_expression(application_function(a),cc, ht),gen_nreverse(new_arguments));
}

static syntax
do_clone_syntax(syntax s, clone_context cc, hash_table ht)
{
    switch(syntax_tag(s))
    {
        case is_syntax_reference:
            {
                gen_chunk* chunk = do_clone_reference(syntax_reference(s),cc, ht);
                if(INSTANCE_OF(reference,chunk))
                {
                    pips_assert("result of cloning is a reference", check_reference((reference)chunk));
                    return make_syntax_reference((reference)chunk);
                }
                else if(INSTANCE_OF(call,chunk))
                {
                    pips_assert("result of cloning is a call", check_call((call)chunk));
                    return make_syntax_call((call)chunk);
                }
                else
                    pips_internal_error("expecting a call or a reference as result of cloning");
            }
        case is_syntax_range:
            return make_syntax_range(do_clone_range(syntax_range(s),cc, ht));
        case is_syntax_call:
            return make_syntax_call(do_clone_call(syntax_call(s),cc, ht));
        case is_syntax_cast:
            return make_syntax_cast(do_clone_cast(syntax_cast(s),cc, ht));
        case is_syntax_sizeofexpression:
            return make_syntax_sizeofexpression(do_clone_sizeofexpression(syntax_sizeofexpression(s),cc, ht));
        case is_syntax_subscript:
            return make_syntax_subscript(do_clone_subscript(syntax_subscript(s),cc, ht));
        case is_syntax_application:
            return make_syntax_application(do_clone_application(syntax_application(s),cc, ht));
        case is_syntax_va_arg:
            {
                list new_va_args = NIL;
                FOREACH(SIZEOFEXPRESSION,soe,syntax_va_arg(s))
                    new_va_args=CONS(SIZEOFEXPRESSION,do_clone_sizeofexpression(soe,cc, ht),new_va_args);
                return make_syntax_va_arg(gen_nreverse(new_va_args));
            }

    };
    return syntax_undefined;
}


static expression
do_clone_expression(expression e, clone_context cc, hash_table ht)
{
    return make_expression(
            do_clone_syntax(expression_syntax(e),cc, ht),
            normalized_undefined);
}


static entity
do_clone_entity(entity e, clone_context cc, hash_table ht)
{
    pips_assert("entity is fine",entity_consistent_p(e));
    entity new_entity = entity_undefined;
    if( (new_entity=hash_get(ht,entity_name(e))) == HASH_UNDEFINED_VALUE)
    {
        if(entity_scalar_p(e))
            new_entity = make_new_scalar_variable_with_prefix(
                    entity_user_name(e),
                    clone_context_new_module(cc),
                    copy_basic(entity_basic(e))
                    );
        else
            new_entity = make_new_array_variable_with_prefix(
                    entity_user_name(e),
                    clone_context_new_module(cc),
                    copy_basic(entity_basic(e)),
                    gen_full_copy_list(variable_dimensions(type_variable(entity_type(e))))
                    );
        AddLocalEntityToDeclarations(new_entity,clone_context_new_module(cc),clone_context_new_module_statement(cc));
        hash_put(ht,entity_name(e),new_entity);
    }
    return new_entity;
}

static sequence
do_clone_sequence(sequence s, clone_context cc, hash_table ht)
{
    list new_statements = NIL;
    FOREACH(STATEMENT,st,sequence_statements(s))
    {
        new_statements=CONS(STATEMENT,do_clone_statement(st,cc, ht),new_statements);
    }
    new_statements=gen_nreverse(new_statements);
    return make_sequence(new_statements);
}

static test
do_clone_test(test t, clone_context cc, hash_table ht)
{
    if(test_undefined_p(t)) return t;
    return make_test(
            do_clone_expression(test_condition(t),cc, ht),
            do_clone_statement(test_true(t),cc, ht),
            do_clone_statement(test_false(t),cc, ht));

}

static loop
do_clone_loop(loop l, clone_context cc, hash_table ht)
{
    entity new_entity = 
         gen_chunk_undefined_p(gen_find_eq(loop_index(l),loop_locals(l))) ?
            do_clone_entity(loop_index(l),cc, ht):
            loop_index(l);
    return make_loop(
            new_entity,
            do_clone_range(loop_range(l),cc, ht),
            do_clone_statement(loop_body(l),cc, ht),
            do_clone_label(loop_label(l),cc/*, ht*/),
            make_execution_sequential(),
            NIL /* unsure ...*/);
}

static entity
do_clone_label(entity l, clone_context cc/*, hash_table ht*/)
{
    if(entity_empty_label_p(l)) return entity_empty_label();

    /* if the label was cloned in the past, we get the same clone this function
       returned before instead of creating a new one */
    entity replacement = entity_undefined;

    /* Checking if the entity is in the list of cloned labels */
    for (list iter = clone_context_labels(cc); !ENDP(iter); POP(iter)) {
        if (same_entity_p(ENTITY(CAR(iter)), l)) {
            POP(iter);
            replacement = ENTITY(CAR(iter));
        } else {
            POP(iter);
        }
    }

    if(entity_undefined_p(replacement)) {
        replacement=make_new_label(clone_context_new_module(cc));
        /* Insert those two values at beginning of the list (reverse inserting order
           as it's insterting before instead of inserting at the end) */
        clone_context_labels(cc) = CONS(ENTITY, replacement, clone_context_labels(cc));
        clone_context_labels(cc) = CONS(ENTITY, l, clone_context_labels(cc));
    }

    return replacement;
}

static whileloop
do_clone_whileloop(whileloop w, clone_context cc, hash_table ht)
{
    return make_whileloop(
            do_clone_expression(whileloop_condition(w),cc, ht),
            do_clone_statement(whileloop_body(w),cc, ht),
            do_clone_label(whileloop_label(w),cc/*, ht*/),
            copy_evaluation(whileloop_evaluation(w)));
}

static call
do_clone_call(call c, clone_context cc, hash_table ht)
{
    list new_arguments = NIL;
    FOREACH(EXPRESSION,e,call_arguments(c))
    {
        new_arguments=CONS(EXPRESSION,do_clone_expression(e,cc, ht),new_arguments);
    }
    return make_call(
            call_function(c),
            gen_nreverse(new_arguments));
}

static multitest
do_clone_multitest(multitest m, clone_context cc, hash_table ht)
{
    return make_multitest(
            do_clone_expression(multitest_controller(m),cc, ht),
            do_clone_statement(multitest_body(m),cc, ht));
}

static forloop
do_clone_forloop(forloop f, clone_context cc, hash_table ht)
{
    return make_forloop(
            do_clone_expression(forloop_initialization(f),cc, ht),
            do_clone_expression(forloop_condition(f),cc, ht),
            do_clone_expression(forloop_increment(f),cc, ht),
            do_clone_statement(forloop_body(f),cc, ht));
}

static instruction
do_clone_instruction(instruction i, clone_context cc, hash_table ht)
{
    if(instruction_undefined_p(i)) return i;

    switch(instruction_tag(i))
    {
        case is_instruction_sequence:
            return make_instruction_sequence(do_clone_sequence(instruction_sequence(i),cc, ht));
        case is_instruction_test:
            return make_instruction_test(do_clone_test(instruction_test(i),cc, ht));
        case is_instruction_loop:
            return make_instruction_loop(do_clone_loop(instruction_loop(i),cc, ht));
        case is_instruction_whileloop:
            return make_instruction_whileloop(do_clone_whileloop(instruction_whileloop(i),cc, ht));
        case is_instruction_goto:
            pips_user_error("don't know how to clone a goto");
        case is_instruction_call:
            return make_instruction_call(do_clone_call(instruction_call(i),cc, ht));
        case is_instruction_unstructured:
            pips_user_error("don't know how to clone an unstructured");
        case is_instruction_multitest:
            return make_instruction_multitest(do_clone_multitest(instruction_multitest(i),cc, ht));
        case is_instruction_forloop:
            return make_instruction_forloop(do_clone_forloop(instruction_forloop(i),cc, ht));
        case is_instruction_expression:
            return make_instruction_expression(do_clone_expression(instruction_expression(i),cc, ht));
    };
    return instruction_undefined;
}

static statement
do_clone_statement(statement s, clone_context cc, hash_table ht)
{
    if (statement_undefined_p(s)) return s;

    entity new_label = do_clone_label(statement_label(s),cc/*, ht*/);
    /* add new declarations to top level statement
     * this prevents difficult scope renaming in C
     */
    list new_declarations_initialization = NIL;
    FOREACH(ENTITY, e, statement_declarations(s))
    {
        entity new_entity = do_clone_entity(e,cc, ht);
        if(! value_unknown_p(entity_initial(e)) &&
                !expression_undefined_p( value_expression(entity_initial(e)) ) )
        {
            statement ns = make_assign_statement(
                    entity_to_expression(new_entity),
                    do_clone_expression(value_expression(entity_initial(e)),cc, ht) );
            new_declarations_initialization=CONS(STATEMENT,ns,new_declarations_initialization);
        }
    }
    instruction new_instruction = do_clone_instruction(statement_instruction(s),cc, ht);
    instruction new_instruction_with_decl = ENDP(new_declarations_initialization)?
        new_instruction:
        make_instruction_sequence(
            make_sequence(
                gen_nconc(
                    gen_nreverse(new_declarations_initialization),
                    CONS(STATEMENT,instruction_to_statement(new_instruction),NIL))
                ));

    return make_statement(
            new_label,
            STATEMENT_NUMBER_UNDEFINED,
            STATEMENT_ORDERING_UNDEFINED,
            empty_comments,
            new_instruction_with_decl,
            NIL,
            NULL,
	    empty_extensions (), make_synchronization_none());

}

statement clone_statement(statement s, clone_context cc)
{
    hash_table ht = hash_table_make(hash_string,0);
    statement sout = do_clone_statement(s,cc,ht);
    hash_table_free(ht);
    return sout;
}

/* end of cloning */
