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
/**
 * @file inlining.c
 * @brief add inlining support to pips, with two flavors
 *  - inlining(char* module) to inline all calls to a module
 *  - unfolding(char* module) to inline all call in a module
 *
 * @author Serge Guelton <serge.guelton@enst-bretagne.fr>
 * @date 2009-01-07
 */
#ifdef HAVE_CONFIG_H
    #include "pips_config.h"
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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
#include "callgraph.h"
#include "effects-generic.h"
#include "effects-convex.h"
#include "preprocessor.h"
#include "text-util.h"
#include "transformations.h"
#include "parser_private.h"
#include "syntax.h"
#include "c_syntax.h"
#include "alias-classes.h"
#include "locality.h"
#include "conversion.h"
#include "pipsmake.h"
#include "accel-util.h"


/**
 * @name inlining
 * @{ */

/**
 * structure containing all the parameters needed by inlining.
 * It avoids using globals
 * newgen like macros are defined
 */
typedef struct {
    entity _inlined_module_;
    statement _inlined_module_statement_;
    statement _new_statements_;
    bool _has_static_declaration_;
    bool _has_inlinable_calls_;
    statement _laststmt_;
    instruction _tail_ins_;
    entity _returned_entity_;
    bool _use_effects_;
} iparam, * inlining_parameters;
#define IPARAM_INIT { \
    ._inlined_module_=entity_undefined,\
    ._inlined_module_statement_=statement_undefined,\
    ._new_statements_=statement_undefined,\
    ._has_static_declaration_=false,\
    ._has_inlinable_calls_=false,\
    ._laststmt_=statement_undefined,\
    ._tail_ins_=instruction_undefined,\
    ._returned_entity_=entity_undefined,\
    ._use_effects_=true\
}
#define inlined_module(p)           (p)->_inlined_module_
#define inlined_module_statement(p) (p)->_inlined_module_statement_
#define new_statements(p)           (p)->_new_statements_
#define has_static_declaration(p)   (p)->_has_static_declaration_
#define has_inlinable_calls(p)      (p)->_has_inlinable_calls_
#define laststmt(p)                 (p)->_laststmt_
#define tail_ins(p)                 (p)->_tail_ins_
#define returned_entity(p)          (p)->_returned_entity_
#define use_effects(p)              (p)->_use_effects_


/* replace return instruction by a goto
 */
static
void inline_return_remover(statement s,inlining_parameters p)
{
    if( return_statement_p( s ) )
        update_statement_instruction(s,make_instruction_goto(copy_statement(laststmt(p))));
}

/* replace return instruction by an assignment and a goto
 */
static
void inline_return_crawler(statement s,inlining_parameters p)
{
    if( return_statement_p( s ) )
    {
        // create the goto
        list l= (statement_instruction(s) == tail_ins(p)) ?
            NIL :
            make_statement_list( instruction_to_statement( make_instruction_goto( copy_statement(laststmt(p)) ) ) ) ;
        // create the assign and push it if needed
        call ic = statement_call(s);
        if( !ENDP(call_arguments(ic)) )
        {
            pips_assert("return is called with one argument",ENDP(CDR(call_arguments(ic))));
            statement assign = make_assign_statement(
                    entity_to_expression( returned_entity(p) ),
                    copy_expression(EXPRESSION(CAR(call_arguments(ic)))));
            l = CONS( STATEMENT, assign , l );
        }
        update_statement_instruction(s,make_instruction_sequence(make_sequence(l)));
    }
}

/* helper function to check if a call is a call to the inlined function
 */
static
bool inline_should_inline(entity inlined_module,call callee)
{
    return same_entity_lname_p(call_function(callee),inlined_module) ;
}

/* find effects on entity `e' in statement `s'
 * cumulated effects for these statements must have been loaded
 */
bool find_write_effect_on_entity(statement s, entity e)
{
	list cummulated_effects = load_cumulated_rw_effects_list( s );
	FOREACH(EFFECT, eff,cummulated_effects)
	{
		reference r = effect_any_reference(eff);
		entity re = reference_variable(r);
		if( entities_may_conflict_p(e,re) )
		{
			if( ENDP( reference_indices(r) ) && action_write_p(effect_action(eff) ) )
                return true;
		}
	}
	return false;
}

static bool has_similar_entity(entity e,set se)
{
    SET_FOREACH(entity,ee,se)
        if( same_string_p(entity_user_name(e),entity_user_name(ee)))
            return true;
    return false;
}



/* look for entity locally named has `new' in statements `s'
 * when found, find a new name and perform substitution
 */
static void
solve_name_clashes(statement s, entity new)
{
    list l = statement_declarations(s);
    set re = get_referenced_entities(s);
    for(;!ENDP(l);POP(l))
    {
        entity decl_ent = ENTITY(CAR(l));
        if( same_string_p(entity_user_name(decl_ent), entity_user_name(new)))
        {
            entity solve_clash = copy_entity(decl_ent);
            string ename = strdup(entity_name(solve_clash));
            do {
                string tmp;
                asprintf(&tmp,"%s_",ename);
                free(ename);
                ename=tmp;
                entity_name(solve_clash)=ename;
            } while( has_similar_entity(solve_clash,re));
            CAR(l).p = (void*)solve_clash;
            replace_entity(s,decl_ent,solve_clash);
        }
    }
    set_free(re);
}

/* return true if an entity declared in `iter' is static to `module'
 */
static
bool inline_has_static_declaration(list iter)
{
    FOREACH(ENTITY, e ,iter)
    {
        if ( variable_static_p(e))
            return true;
    }
    return false;
}

/* return true if an entity declared in the statement `s' from
   `p->inlined_module'
 */
static
void statement_with_static_declarations_p(statement s,inlining_parameters p )
{
    has_static_declaration(p)|=inline_has_static_declaration(statement_declarations(s) );
}

/* create a scalar similar to `efrom' initialized with expression `from'
 */
static
entity make_temporary_scalar_entity(expression from,statement * assign)
{
    pips_assert("expression is valid",expression_consistent_p(from)&&!expression_undefined_p(from));
    /* create the scalar */
	entity new = make_new_scalar_variable(
			get_current_module_entity(),
			basic_of_expression(from)
			);
    /* set intial */
    if(!expression_undefined_p(from))
    {
        *assign=make_assign_statement(entity_to_expression(new),copy_expression(from));
    }
	return new;
}

/* regenerate the label of each statement with a label.
 * it avoid duplications due to copy_statement
 */
static
void inlining_regenerate_labels(statement s, string new_module)
{
    entity lbl = statement_label(s);
    if(!entity_empty_label_p(lbl))
    {
        if( !entity_undefined_p(find_label_entity(new_module,label_local_name(lbl))))
        {
            statement_label(s)=lbl=make_new_label(new_module);
        }
        else
            FindOrCreateEntity(new_module,entity_local_name(lbl));
        if( statement_loop_p(s) )
            loop_label(statement_loop(s))=lbl;
    }
    else if( statement_loop_p(s) ) {
        if( !entity_undefined_p( find_label_entity(new_module,label_local_name(lbl)) ) )
            loop_label(statement_loop(s))=make_new_label(new_module);
    }
}

bool has_entity_with_same_name(entity e, list l) {
    FOREACH(ENTITY,ent,l)
        if(same_entity_name_p(e,ent)) return true;
    return false;
}


/* this should inline the call callee
 * calling module inlined_module
 */
static
statement inline_expression_call(inlining_parameters p, expression modified_expression, call callee)
{
    /* only inline the right call */
    pips_assert("inline the right call",inline_should_inline(inlined_module(p),callee));

    string modified_module_name = entity_local_name(get_current_module_entity());

    value inlined_value = entity_initial(inlined_module(p));
    pips_assert("is a code", value_code_p(inlined_value));
    code inlined_code = value_code(inlined_value);

    /* stop if the function has static declaration */
    {
        has_static_declaration(p)=false;
        if( c_module_p(inlined_module(p)) )
            gen_context_recurse(inlined_module_statement(p),p, statement_domain, gen_true, statement_with_static_declarations_p);
        else
            has_static_declaration(p)= inline_has_static_declaration( code_declarations(inlined_code) );

        if( has_static_declaration(p))
        {
            pips_user_warning("cannot inline function with static declarations\n");
            return statement_undefined;
        }
    }

    /* create the new instruction sequence
     * no need to change all entities in the new statements, because we build a new text ressource latter
     */
    statement expanded = copy_statement(inlined_module_statement(p));
    statement declaration_holder = make_empty_block_statement();
    //statement_declarations(expanded) = gen_full_copy_list( statement_declarations(expanded) ); // simple copy != deep copy

    /* add external declartions for all extern referenced entities it
     * is needed because inlined module and current module may not
     * share the same compilation unit.
     * Not relevant for fortran
     *
     * FI: However, it would be nice to check first if the entity is not
     * already in the scope for the function or in the scope of its
     * compilation unit (OK, the later is difficult because the order
     * of declarations has to be taken into account).
     */
    if(c_module_p(get_current_module_entity()))
    {
        string cu_name = compilation_unit_of_module(get_current_module_name());
        set inlined_referenced_entities = get_referenced_entities(expanded);
        list lire = set_to_sorted_list(inlined_referenced_entities,(gen_cmp_func_t)compare_entities);
        set_free(inlined_referenced_entities);
        //list new_externs = NIL;
        FOREACH(ENTITY,ref_ent,lire)
        {
            if( entity_field_p(ref_ent) ) /* special hook for struct member : consider their structure instead of the field */
            {
                ref_ent=entity_field_to_entity_struct(ref_ent);
            }

            if(!entity_enum_member_p(ref_ent) && /* enum member cannot be added to declarations */
                    !entity_formal_p(ref_ent) ) /* formal parameters are not considered */
            {
                string emn = entity_module_name(ref_ent);
                if(extern_entity_p(get_current_module_entity(),ref_ent) &&
                        !has_entity_with_same_name(ref_ent,entity_declarations(module_name_to_entity(cu_name)) ) )
                {
                    AddEntityToModuleCompilationUnit(ref_ent,get_current_module_entity());
                    gen_append(code_externs(entity_code(module_name_to_entity(cu_name))),CONS(ENTITY,ref_ent,NIL));
                }
                else if(variable_static_p(ref_ent) &&
                        !has_entity_with_same_name(ref_ent,entity_declarations(module_name_to_entity(cu_name)) ) )
                {
                    pips_user_warning("replacing static variable \"%s\" by a global one, this may lead to incorrect code\n", entity_user_name(ref_ent));
                    entity add = make_global_entity_from_local(ref_ent);
                    replace_entity(expanded,ref_ent,add);
                    replace_entity(inlined_module_statement(p),ref_ent,add);
                    AddEntityToModuleCompilationUnit(add,get_current_module_entity());
                    gen_append(code_externs(entity_code(module_name_to_entity(cu_name))),CONS(ENTITY,add,NIL));
                }
                else if(!variable_entity_p(ref_ent) && !same_string_p(emn,cu_name) &&
                        !has_entity_with_same_name(ref_ent,entity_declarations(module_name_to_entity(cu_name)) ))
                {
                    AddEntityToModuleCompilationUnit(ref_ent,get_current_module_entity());
                }
            }
        }
        gen_free_list(lire);
    }
    /* we have anothe rproblem with fortran, where declaration are  not handled like in C
     * another side effect of 'declarations as statements' */
    else
    {
        bool did_something = false;
        FOREACH(ENTITY,e,entity_declarations(inlined_module(p)))
        {
            if(!entity_area_p(e))
            {
                entity new;
                if(entity_variable_p(e)) {
                    if(entity_scalar_p(e)) {
                        new = make_new_scalar_variable_with_prefix(entity_user_name(e),get_current_module_entity(),copy_basic(entity_basic(e)));
                    }
                    else {
                        new = make_new_array_variable_with_prefix(entity_user_name(e),get_current_module_entity(),
                                copy_basic(entity_basic(e)), gen_full_copy_list(variable_dimensions(type_variable(entity_type(e)))));
                    }
                }
                else
                {
                    /*sg: unsafe*/
                    bool regenerate = entity_undefined_p(FindEntity(get_current_module_name(),entity_local_name(e)));
                    new=FindOrCreateEntity(get_current_module_name(),entity_local_name(e));
                    if(regenerate)
                    {
                        entity_storage(new)=copy_storage(entity_storage(e));
                        entity_initial(new)=copy_value(entity_initial(e));
                        entity_type(new)=copy_type(entity_type(e));
                    }
                }
                gen_context_recurse(expanded, new, statement_domain, gen_true, &solve_name_clashes);
                AddEntityToDeclarations(new,get_current_module_entity());
                replace_entity(expanded,e,new);
                did_something=true;
            }
        }
        if(did_something)
        {
            string decls = code_decls_text(entity_code(get_current_module_entity()));
            if(decls && !empty_string_p(decls)){
                free(decls);
                code_decls_text(entity_code(get_current_module_entity()))=strdup("");
            }
        }
    }


    /* ensure block status */
    if( ! statement_block_p( expanded ) )
    {
        instruction i = make_instruction_sequence( make_sequence( CONS(STATEMENT,expanded,NIL) ) );
        expanded = instruction_to_statement( i );
    }


    /* avoid duplicated label due to copy_statement */
    gen_context_recurse(expanded,modified_module_name,statement_domain,gen_true,inlining_regenerate_labels);

    /* add label at the end of the statement */
    laststmt(p)=make_continue_statement(make_new_label( modified_module_name ) );
    insert_statement(expanded,laststmt(p),false);

    /* fix `return' calls
     * in case a goto is immediatly followed by its targeted label
     * the goto is not needed (SG: seems difficult to do in the previous gen_recurse)
     */
    {
        list tail = sequence_statements(instruction_sequence(statement_instruction(expanded)));
        pips_assert("inlinined statement is not empty",!ENDP(tail) && !ENDP(CDR(tail)));
        {
            tail_ins(p)= statement_instruction(STATEMENT(CAR(gen_last(tail))));

            type treturn = ultimate_type(functional_result(type_functional(entity_type(inlined_module(p)))));
            if( type_void_p(treturn) ) /* only replace return statement by gotos */
            {
                gen_context_recurse(expanded, p,statement_domain, gen_true, &inline_return_remover);
            }
            else /* replace by affectation + goto */
            {
                /* create new variable to receive computation result */
                pips_assert("returned value is a variable", type_variable_p(treturn));
                do {
                    returned_entity(p)= make_new_scalar_variable_with_prefix(
                            "_return",
                            get_current_module_entity(),
                            copy_basic(variable_basic(type_variable(treturn)))
                            );
                    /* make_new_scalar_variable does not ensure the entity is not defined in enclosing statement, we check this */
                    FOREACH(ENTITY,ent,statement_declarations(expanded))
                    {
                        if(same_string_p(entity_user_name(ent),entity_user_name(returned_entity(p))))
                        {
                            returned_entity(p)=entity_undefined;
                            break;
                        }
                    }
                } while(entity_undefined_p(returned_entity(p)));

                AddEntityToCurrentModule(returned_entity(p));

                /* do the replacement */
                gen_context_recurse(expanded, p, statement_domain, gen_true, &inline_return_crawler);

                /* change the caller from an expression call to a call to a constant */
                if( entity_constant_p(returned_entity(p)) )
                {
                    expression_syntax(modified_expression) = make_syntax_call(make_call(returned_entity(p),NIL));
                }
                /* ... or to a reference */
                else
                {
                    reference r = make_reference( returned_entity(p), NIL);
                    expression_syntax(modified_expression) = make_syntax_reference(r);
                }
            }
        }
    }

    /* fix declarations */
    {
        /* retreive formal parameters*/
        list formal_parameters = NIL;
        FOREACH(ENTITY,cd,code_declarations(inlined_code))
            if( entity_formal_p(cd)) formal_parameters=CONS(ENTITY,cd,formal_parameters);
        formal_parameters = gen_nreverse(formal_parameters);

        { /* some basic checks */
            size_t n1 = gen_length(formal_parameters), n2 = gen_length(call_arguments(callee));
            pips_assert("function call has enough arguments",n1 >= n2);
        }
        /* iterate over the parameters and perform substitution between formal and actual parameters */
        for(list iter = formal_parameters,c_iter = call_arguments(callee) ; !ENDP(c_iter); POP(iter),POP(c_iter) )
        {
            entity e = ENTITY(CAR(iter));
            expression from = EXPRESSION(CAR(c_iter));

            /* check if there is a write effect on this parameter */
            bool need_copy = (!use_effects(p)) || find_write_effect_on_entity(inlined_module_statement(p),e);

            /* generate a copy for this parameter */
            entity new = entity_undefined;
            if(need_copy)
            {
                if(entity_scalar_p(e)) {
                    new = make_new_scalar_variable_with_prefix(entity_user_name(e),get_current_module_entity(),copy_basic(entity_basic(e)));
                }
                else {
                    new = make_new_array_variable_with_prefix(entity_user_name(e),get_current_module_entity(),
                            copy_basic(entity_basic(e)), gen_full_copy_list(variable_dimensions(type_variable(entity_type(e)))));
                }

                /* fix value */
                entity_initial(new) = make_value_expression( copy_expression( from ) );


                /* add the entity to our list */
                //statement_declarations(declaration_holder)=CONS(ENTITY,new,statement_declarations(declaration_holder));
                gen_context_recurse(expanded, new, statement_domain, gen_true, &solve_name_clashes);
                AddLocalEntityToDeclarations(new,get_current_module_entity(),declaration_holder);
                replace_entity(expanded,e,new);
                pips_debug(2,"replace %s by %s",entity_user_name(e),entity_user_name(new));
            }
            /* substitute variables */
            else
            {
                /* get new reference */

                bool add_dereferencment = false;
reget:
                switch(syntax_tag(expression_syntax(from)))
                {
                    case is_syntax_reference:
                        {
                            reference r = syntax_reference(expression_syntax(from));
                            size_t nb_indices = gen_length(reference_indices(r));
                            if( nb_indices == 0 )
                            {
                                new = reference_variable(r);
                            }
                            else /* need a temporary variable */
                            {
                                if( ENDP(variable_dimensions(type_variable(entity_type(e)))) )
                                {
                                    statement st=statement_undefined;
                                    new = make_temporary_scalar_entity(from,&st);
                                    if(!statement_undefined_p(st))
                                        insert_statement(declaration_holder,st,false);
                                }
                                else
                                {
                                    new = make_temporary_pointer_to_array_entity(e,MakeUnaryCall(entity_intrinsic(ADDRESS_OF_OPERATOR_NAME),from));
                                    add_dereferencment=true;
                                }
                                AddLocalEntityToDeclarations(new,get_current_module_entity(),declaration_holder);

                            }
                        } break;
                        /* this one is more complicated than I thought,
                         * what of the side effect of the call ?
                         * we must create a new variable holding the call result before
                         */
                    case is_syntax_call:
                        if( expression_constant_p(from) )
                            new = call_function(expression_call(from));
                        else
                        {
                                if( ENDP(variable_dimensions(type_variable(entity_type(e)))) )
                                {
                                    statement st=statement_undefined;
                                    new = make_temporary_scalar_entity(from,&st);
                                    if(!statement_undefined_p(st))
                                        insert_statement(declaration_holder,st,false);
                                }
                                else
                                {
                                    statement st=statement_undefined;
                                    new = make_temporary_scalar_entity(from,&st);
                                    if(!statement_undefined_p(st))
                                        insert_statement(declaration_holder,st,false);
                                }
                                AddLocalEntityToDeclarations(new,get_current_module_entity(),declaration_holder);
                        } break;
                    case is_syntax_subscript:
                        /* need a temporary variable */
                        {
                            if( ENDP(variable_dimensions(type_variable(entity_type(e)))) )
                            {
                                    statement st=statement_undefined;
                                    new = make_temporary_scalar_entity(from,&st);
                                    if(!statement_undefined_p(st))
                                        insert_statement(declaration_holder,st,false);
                            }
                            else
                            {
                                new = make_temporary_pointer_to_array_entity(e,MakeUnaryCall(entity_intrinsic(ADDRESS_OF_OPERATOR_NAME),from));
                                add_dereferencment=true;
                            }
                            AddLocalEntityToDeclarations(new,get_current_module_entity(),declaration_holder);

                        } break;

                    case is_syntax_cast:
                        pips_user_warning("ignoring cast\n");
                        from = cast_expression(syntax_cast(expression_syntax(from)));
                        goto reget;
                    default:
                        pips_internal_error("unhandled tag %d\n", syntax_tag(expression_syntax(from)) );
                };

                /* check wether the substitution will cause naming clashes
                 * then perform the substitution
                 */
                    gen_context_recurse(expanded , new, statement_domain, gen_true, &solve_name_clashes);
                    if(add_dereferencment) replace_entity_by_expression(expanded ,e,MakeUnaryCall(entity_intrinsic(DEREFERENCING_OPERATOR_NAME),entity_to_expression(new)));
                    else replace_entity(expanded ,e,new);
                    pips_debug(3,"replace %s by %s\n",entity_user_name(e),entity_user_name(new));

            }

        }
        gen_free_list(formal_parameters);
    }

    /* add declaration at the beginning of the statement */
    insert_statement(declaration_holder,expanded,false);

    /* final cleanings
     */
    gen_recurse(expanded,statement_domain,gen_true,fix_statement_attributes_if_sequence);
    unnormalize_expression(expanded);
    ifdebug(1) statement_consistent_p(declaration_holder);
    ifdebug(2) {
        pips_debug(2,"inlined statement after substitution\n");
        print_statement(declaration_holder);
    }
    return declaration_holder;
}


/* recursievly inline an expression if needed
 */
static
void inline_expression(expression expr, inlining_parameters  p)
{
    if( expression_call_p(expr) )
    {
        call callee = syntax_call(expression_syntax(expr));
        if( inline_should_inline( inlined_module(p), callee ) )
        {
                statement s = inline_expression_call(p, expr, callee );
                if( !statement_undefined_p(s) )
                {
                    insert_statement(new_statements(p),s,true);
                }
                ifdebug(1) statement_consistent_p(s);
                ifdebug(2) {
                    pips_debug(2,"inserted inline statement\n");
                    print_statement(new_statements(p));
                }
        }
    }
}

/* check if a call has inlinable calls
 */
static
void inline_has_inlinable_calls_crawler(call callee,inlining_parameters p)
{
    if( has_inlinable_calls(p)|=inline_should_inline(inlined_module(p),callee) ) gen_recurse_stop(0);
}
static
bool inline_has_inlinable_calls(entity inlined_module,void* elem)
{
    iparam p = { ._inlined_module_=inlined_module,._has_inlinable_calls_=false};
    gen_context_recurse(elem, &p, call_domain, gen_true,&inline_has_inlinable_calls_crawler);
    return has_inlinable_calls(&p);
}


/* this is in charge of replacing instruction by new ones
 * only apply if this instruction does not contain other instructions
 */
static
void inline_statement_crawler(statement stmt, inlining_parameters p)
{
    instruction sti = statement_instruction(stmt);
    if( instruction_call_p(sti) && inline_has_inlinable_calls(inlined_module(p),sti) )
    {
        /* the gen_recurse can only handle expressions, so we turn this call into an expression */
        update_statement_instruction(stmt,
                sti=make_instruction_expression(call_to_expression(copy_call(instruction_call(sti)))));
    }

    new_statements(p)=make_block_statement(NIL);
    gen_context_recurse(sti,p,expression_domain,gen_true, inline_expression);

    if( !ENDP(statement_block(new_statements(p)))  ) /* something happens on the way to heaven */
    {
        type t= functional_result(type_functional(entity_type(inlined_module(p))));
        if( ! type_void_p(t) )
        {
            //pips_assert("inlining instruction modification is ok", instruction_consistent_p(sti));
            insert_statement(new_statements(p),instruction_to_statement(copy_instruction(sti)),false);
        }
        if(statement_block_p(stmt))
        {
            list iter=statement_block(stmt);
            for(stmt=STATEMENT(CAR(iter));continue_statement_p(stmt);POP(iter))
                stmt=STATEMENT(CAR(iter));
        }
        update_statement_instruction(stmt,statement_instruction(new_statements(p)));
        ifdebug(2) {
            pips_debug(2,"updated statement instruction\n");
            print_statement(stmt);
        }
        //pips_assert("inlining statement generation is ok",statement_consistent_p(stmt));
    }
    ifdebug(1) statement_consistent_p(stmt);
}

/* split the declarations from s from their initialization if they contain a call to inlined_module
 */
static
void inline_split_declarations(statement s, entity inlined_module)
{
    if(statement_block_p(s))
    {
        list prelude = NIL;
        set selected_entities = set_make(set_pointer);
        FOREACH(ENTITY,e,statement_declarations(s))
        {
            value v = entity_initial(e);
            if(!value_undefined_p(v) && value_expression_p(v))
            {
                /* the first condition is a bit tricky :
                 * check int a = foo(); int b=bar();
		 * once we decide to inline foo(), we must split b=bar()
                 * because foo may touch a global variable used in bar()
                 */
                if( !ENDP(prelude) ||
                    inline_has_inlinable_calls(inlined_module,value_expression(v)) )
                {
                    set_add_element(selected_entities,selected_entities,e);
                    prelude=CONS(STATEMENT,make_assign_statement(entity_to_expression(e),copy_expression(value_expression(v))),prelude);
                    free_value(entity_initial(e));
                    entity_initial(e)=make_value_unknown();
                }
            }
        }
        set_free(selected_entities);
        FOREACH(STATEMENT,st,prelude)
            insert_statement(s,st,true);
        gen_free_list(prelude);
    }
    else if(!declaration_statement_p(s) && !ENDP(statement_declarations(s)))
        pips_user_warning("only blocks and declaration statements should have declarations\n");
}

/* this should replace all call to `inlined' in `module'
 * by the expansion of `inlined'
 */
static void
inline_calls(inlining_parameters p ,char * module)
{
    entity modified_module = module_name_to_entity(module);
    /* get target module's ressources */
    statement modified_module_statement =
        (statement) db_get_memory_resource(DBR_CODE, module, TRUE);
    pips_assert("statements found", !statement_undefined_p(modified_module_statement) );
    pips_debug(2,"inlining %s in %s\n",entity_user_name(inlined_module(p)),module);

    set_current_module_entity( modified_module );
    set_current_module_statement( modified_module_statement );

    /* first pass : convert some declaration with assignment to declarations + statements, if needed */
    gen_context_recurse(modified_module_statement, inlined_module(p), statement_domain, gen_true, inline_split_declarations);
    /* inline all calls to inlined_module */
    gen_context_recurse(modified_module_statement, p, statement_domain, gen_true, inline_statement_crawler);
    ifdebug(1) statement_consistent_p(modified_module_statement);
    ifdebug(2) {
        pips_debug(2,"in inline_calls for %s\n",module);
        print_statement(modified_module_statement);
    }

    DB_PUT_MEMORY_RESOURCE(DBR_CODE, module, modified_module_statement);
    DB_PUT_MEMORY_RESOURCE(DBR_CALLEES, module, compute_callees(modified_module_statement));
    reset_current_module_entity();
    reset_current_module_statement();
}

/**
 * this should inline all calls to module `module_name'
 * in calling modules, if possible ...
 *
 * @param module_name name of the module to inline
 *
 * @return true if we did something
 */
static
bool do_inlining(inlining_parameters p,char *module_name)
{
    /* Get the module ressource */
    inlined_module (p)= module_name_to_entity( module_name );
    inlined_module_statement (p)= (statement) db_get_memory_resource(DBR_CODE, module_name, TRUE);

    if(use_effects(p)) set_cumulated_rw_effects((statement_effects)db_get_memory_resource(DBR_CUMULATED_EFFECTS,module_name,TRUE));

    /* check them */
    pips_assert("is a functionnal",entity_function_p(inlined_module(p)) || entity_subroutine_p(inlined_module(p)) );
    pips_assert("statements found", !statement_undefined_p(inlined_module_statement(p)) );
    debug_on("INLINING_DEBUG_LEVEL");


    /* parse filter property */
    string inlining_callers_name = strdup(get_string_property("INLINING_CALLERS"));
    list callers_l = NIL;

    string c_name= NULL;
    for(c_name = strtok(inlining_callers_name," ") ; c_name ; c_name=strtok(NULL," ") )
    {
        callers_l = CONS(STRING, c_name, callers_l);
    }
    /*  or get module's callers */
    if(ENDP(callers_l))
    {
        callees callers = (callees)db_get_memory_resource(DBR_CALLERS, module_name, TRUE);
        callers_l = callees_callees(callers);
    }

    /* inline call in each caller */
    FOREACH(STRING, caller_name,callers_l)
    {
        inline_calls(p, caller_name );
        recompile_module(caller_name);
        /* we can try to remove some labels now*/
        if( get_bool_property("INLINING_PURGE_LABELS"))
            if(!remove_useless_label(caller_name))
                pips_user_warning("failed to remove useless labels after restructure_control in inlining");
    }

    if(use_effects(p)) reset_cumulated_rw_effects();

    pips_debug(2, "inlining done for %s\n", module_name);
    debug_off();
    /* Should have worked: */
    return TRUE;
}

/**
 * perform inlining using effects
 *
 * @param module_name name of the module to inline
 *
 * @return
 */
bool inlining(char *module_name)
{
    iparam p =IPARAM_INIT;
    use_effects(&p)=true;
	return do_inlining(&p,module_name);
}

/**
 * perform inlining without using effects
 *
 * @param module_name name of the module to inline
 *
 * @return
 */
bool inlining_simple(char *module_name)
{
    iparam p =IPARAM_INIT;
    use_effects(&p)=false;
	return do_inlining(&p,module_name);
}

/**  @} */

/**
 * @name unfolding
 * @{ */

/**
 * get ressources for the call to inline and call
 * apropriate inlining function
 *
 * @param caller_name calling module name
 * @param module_name called module name
 */
static bool
run_inlining(string caller_name, string module_name, inlining_parameters p)
{
    /* Get the module ressource */
    inlined_module (p)= module_name_to_entity( module_name );
    inlined_module_statement (p)= (statement) db_get_memory_resource(DBR_CODE, module_name, TRUE);
    if(statement_block_p(inlined_module_statement (p)) && ENDP(statement_block(inlined_module_statement (p))))
    {
        pips_user_warning("not inlining empty function, this should be a generated skeleton ...\n");
        return false;
    }
    else {

        if(use_effects(p)) set_cumulated_rw_effects((statement_effects)db_get_memory_resource(DBR_CUMULATED_EFFECTS,module_name,TRUE));

        /* check them */
        pips_assert("is a functionnal",entity_function_p(inlined_module(p)) || entity_subroutine_p(inlined_module(p)) );
        pips_assert("statements found", !statement_undefined_p(inlined_module_statement(p)) );

        /* inline call */
        inline_calls( p, caller_name );
        if(use_effects(p)) reset_cumulated_rw_effects();
        return true;
    }
}


/**
 * this should inline all call in module `module_name'
 * it does not works recursievly, so multiple pass may be needed
 * returns true if at least one function has been inlined
 *
 * @param module_name name of the module to unfold
 *
 * @return true if we did something
 */
static
bool do_unfolding(inlining_parameters p, char* module_name)
{
    debug_on("UNFOLDING_DEBUG_LEVEL");

    /* parse filter property */
    string unfolding_filter_names = strdup(get_string_property("UNFOLDING_FILTER"));
    set unfolding_filters = set_make(set_string);

    string filter_name= NULL;
    for(filter_name = strtok(unfolding_filter_names," ") ; filter_name ; filter_name=strtok(NULL," ") )
    {
        set_add_element(unfolding_filters, unfolding_filters, filter_name);
        recompile_module(module_name);
        /* we can try to remove some labels now*/
        if( get_bool_property("INLINING_PURGE_LABELS"))
            if(!remove_useless_label(module_name))
                pips_user_warning("failed to remove useless labels after restructure_control in inlining");
    }

    /* parse callee property */
    string unfolding_callees_names = strdup(get_string_property("UNFOLDING_CALLEES"));
    set unfolding_callees = set_make(set_string);

    string callee_name= NULL;
    for(callee_name = strtok(unfolding_callees_names," ") ; callee_name ; callee_name=strtok(NULL," ") )
    {
        set_add_element(unfolding_callees, unfolding_callees, callee_name);
    }

    /* gather all referenced calls as long as there are some */
    bool statement_has_callee = false;
    do {
        statement_has_callee = false;
        statement unfolded_module_statement =
            (statement) db_get_memory_resource(DBR_CODE, module_name, TRUE);
        /* gather all referenced calls */
        callees cc =compute_callees(unfolded_module_statement);
        set calls_name = set_make(set_string);
        set_assign_list(calls_name,callees_callees(cc));


        /* maybe the user put a restriction on the calls to inline ?*/
        if(!set_empty_p(unfolding_callees))
            calls_name=set_intersection(calls_name,calls_name,unfolding_callees);

        /* maybe the user used a filter ?*/
        calls_name=set_difference(calls_name,calls_name,unfolding_filters);



        /* there is something to inline */
        if( (statement_has_callee=!set_empty_p(calls_name)) )
        {
            SET_FOREACH(string,call_name,calls_name) {
                if(!run_inlining(module_name,call_name,p))
                    set_add_element(unfolding_filters,unfolding_filters,call_name);
            }
            recompile_module(module_name);
            /* we can try to remove some labels now*/
            if( get_bool_property("INLINING_PURGE_LABELS"))
                if(!remove_useless_label(module_name))
                    pips_user_warning("failed to remove useless labels after restructure_control in inlining");
        }
        set_free(calls_name);
        free_callees(cc);
    } while(statement_has_callee);

    set_free(unfolding_filters);
    free(unfolding_filter_names);


    pips_debug(2, "unfolding done for %s\n", module_name);

    debug_off();
    return true;
}


/**
 * perform unfolding using effect
 *
 * @param module_name name of the module to unfold
 *
 * @return
 */
bool unfolding(char* module_name)
{
    iparam p = IPARAM_INIT;
    use_effects(&p)=true;
	return do_unfolding(&p,module_name);
}


/**
 * perform unfolding without using effects
 *
 * @param module_name name of the module to unfold
 *
 * @return true upon success
 */
bool unfolding_simple(char* module_name)
{
    iparam p = IPARAM_INIT;
    use_effects(&p)=false;
	return do_unfolding(&p,module_name);
}
/**  @} */
