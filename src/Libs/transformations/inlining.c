/*

  $Id$

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
 * @file inlining.c
 * @brief add inlining support to pips, with two flavors
 *  - inlining(char* module) to inline all calls to a module
 *  - unfolding(char* module) to inline all call in a module
 *  - outlining(char* module) to outline statements from a module
 *
 * @author Serge Guelton <serge.guelton@enst-bretagne.fr>
 * @date 2009-01-07
 */

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
#include "effects-generic.h"
#include "preprocessor.h"
#include "text-util.h"
#include "transformations.h"
#include "parser_private.h"
#include "syntax.h"
#include "c_syntax.h"


/** 
 * @name inlining
 * @{ */
static entity       inlined_module;
static statement    inlined_module_statement;
static statement    laststmt;
static entity       returned_entity;
static bool 		use_effects;

static void reset_expression_normalized(expression e)
{
    if(!normalized_undefined_p(expression_normalized(e)))
        free_normalized(expression_normalized(e));
    expression_normalized(e)=normalized_undefined;
}

/* replace return instruction by a goto
 */
static
void inline_return_remover(instruction ins,instruction tail_ins)
{
    if( return_instruction_p( ins ) && ins !=tail_ins )
    {
        free_call(instruction_call(ins));
        instruction_tag(ins)=is_instruction_goto;
        instruction_goto(ins)=copy_statement(laststmt);
    }
}

/* replace return instruction by an assignment and a goto
 */
static
void inline_return_switcher(instruction ins,instruction tail_ins)
{
    if( return_instruction_p( ins ) )
    {
        // create the goto
        list l= (ins == tail_ins ) ? NIL : CONS( STATEMENT, instruction_to_statement( make_instruction_goto( copy_statement(laststmt) ) ) , NIL ) ;
        // create the assign and push it if needed
        call ic = instruction_call(ins);
        if( !ENDP(call_arguments(ic)) )
        {
            call c = make_call(
                    CreateIntrinsic(ASSIGN_OPERATOR_NAME),
                    CONS(
                        EXPRESSION,
                        entity_to_expression( returned_entity ),
                        gen_full_copy_list(call_arguments(ic))
                        )
                    );
            l = CONS( STATEMENT, instruction_to_statement(  make_instruction_call(c) ), l );
        }

        free_call( instruction_call(ins));
        instruction_tag(ins) = is_instruction_sequence;
        instruction_sequence(ins)=make_sequence( l );
    }
}

/* helper function to check if a call is a call to the inlined function
 */
static
bool inline_should_inline(call callee)
{
    return call_function(callee) == inlined_module ;
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
		if( same_entity_name_p(e,re) )
		{
			cell c = effect_cell(eff);
			if( ENDP( reference_indices( cell_preference_p(c) ? preference_reference(cell_preference(c)) : cell_reference(c) ) ) )
				if( action_write_p(effect_action(eff) ) )
					return true;
		}
	}
	return false;
}



/* look for entity locally named has `new' in statements `s'
 * when found, find a new name and perform substitution
 */
static void
solve_name_clashes(statement s, entity new)
{
    list l = statement_declarations(s);
    for(;!ENDP(l);POP(l))
    {
        entity decl_ent = ENTITY(CAR(l));
        if( same_string_p(local_name(entity_name_without_scope(decl_ent)),
                    local_name(entity_name_without_scope(new))) )
        {
            entity solve_clash = copy_entity(decl_ent);
            string ename = strdup(entity_name(solve_clash));
            do {
                string tmp =strdup( concatenate( ename, "_" , NULL ) );
                free(ename);
                ename=tmp;
            } while( gen_find_tabulated( ename, entity_domain) != entity_undefined );
            entity_name(solve_clash)=ename;
            CAR(l).p = (void*)solve_clash;
            replace_entity(s,decl_ent,solve_clash);
        }
    }
}
static
bool inline_has_static_declaration(list iter)
{
    FOREACH(ENTITY, e ,iter)
    {
        storage s = entity_storage(e);
        if ( storage_ram_p(s) && ENTITY_NAME_P( ram_section(storage_ram(s)), STATIC_AREA_LOCAL_NAME) )
            return true;
    }
    return false;
}

static
void statement_with_static_declarations_p(statement s,bool* has_static_declaration)
{
    *has_static_declaration|=inline_has_static_declaration( statement_declarations(s) );
}

static
entity make_temporary_array_entity(entity efrom, expression from)
{
	basic pointers =copy_basic(variable_basic(type_variable(entity_type(efrom))));
	list dims = gen_copy_seq(variable_dimensions(type_variable(entity_type(efrom))));
	if(ENDP(CDR(dims))) dims=NIL;
	else {

		list iter = dims;
		while(!ENDP(CDR(CDR(iter)))) POP(iter);
		//free_dimension(DIMENSION(CAR(CDR(iter))));
		CDR(iter)=NIL;
	}
	pointers = make_basic_pointer(make_type_variable(make_variable(pointers,dims,NIL)));
	entity new = make_new_scalar_variable(
			get_current_module_entity(),
			pointers
			);
	entity_initial(new) = expression_undefined_p(from)?value_undefined:
		make_value_expression(make_expression(make_syntax_cast(make_cast(make_type_variable(make_variable(pointers,NIL,NIL)),from)),normalized_undefined));
	AddLocalEntityToDeclarations(new, get_current_module_entity(),
			c_module_p(get_current_module_entity())?get_current_module_statement():statement_undefined);
	return new;
}
static
entity make_temporary_scalar_entity(entity efrom, expression from)
{
	entity new = make_new_scalar_variable(
			get_current_module_entity(),
			copy_basic(variable_basic(type_variable(entity_type(efrom))))
			);
	entity_initial(new) = expression_undefined_p(from)?value_undefined:make_value_expression(from);
	AddLocalEntityToDeclarations(new, get_current_module_entity(),
			c_module_p(get_current_module_entity())?get_current_module_statement():statement_undefined);
	return new;
}

static
void inlining_regenerate_labels(statement s, string new_module)
{
    entity lbl = statement_label(s);
    if(!entity_empty_label_p(lbl))
    {
        if( !entity_undefined_p(find_label_entity(new_module,label_local_name(lbl))))
            statement_label(s)=lbl=make_new_label(new_module);
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

/* this should inline the call callee
 * calling module inlied_module
 */
static
statement inline_expression_call(expression modified_expression, call callee)
{

    /* only inline the right call */
    pips_assert("inline the right call",inline_should_inline(callee));

    string modified_module_name = entity_local_name(get_current_module_entity());

    value inlined_value = entity_initial(inlined_module);
    pips_assert("is a code", value_code_p(inlined_value));
    code inlined_code = value_code(inlined_value);

    /* stop if the function has static declaration */
    {
        bool has_static_declarations = false;
        if( c_module_p(inlined_module) )
        {
            gen_context_recurse(inlined_module_statement,&has_static_declarations, statement_domain, gen_true, statement_with_static_declarations_p);
        }
        else
        {
            has_static_declarations = inline_has_static_declaration( code_declarations(inlined_code) );
        }
        if( has_static_declarations )
        {
            pips_user_warning("cannot inline function with static declarations\n");
            return statement_undefined;
        }
    }

    /* should add them to modified_module's*/
    list inlined_extern_declaration = code_externs(inlined_code);
    pips_assert("no external declaration",inlined_extern_declaration == NIL);


    /* create the new instruction sequence
     * no need to change all entities in the new statements, because we build a new text ressource latter
     */
    statement expanded = copy_statement(inlined_module_statement);
    statement declaration_holder = expanded;
    statement_declarations(expanded) = gen_full_copy_list( statement_declarations(expanded) ); // simple copy != deep copy

    /* fix block status */
    if( ! statement_block_p( expanded ) )
    {
        instruction i = make_instruction_sequence( make_sequence( CONS(STATEMENT,expanded,NIL) ) );
        expanded = instruction_to_statement( i );
    }

    /* avoid duplicated label due to copy_statement */
    gen_context_recurse(expanded,modified_module_name,statement_domain,gen_true,inlining_regenerate_labels);

    /* add label at the end of the statement */
    laststmt=make_continue_statement(make_new_label( modified_module_name ) );
    gen_nconc( sequence_statements(instruction_sequence(statement_instruction(expanded))), CONS( STATEMENT, laststmt, NIL) );

    /* fix `return' calls
     * in case a goto is immediatly followed by its targeted label
     * the goto is not needed (SG: seems difficult to do in the previous gen_recurse)
     */
    list tail = sequence_statements(instruction_sequence(statement_instruction(expanded)));
    if( ENDP(tail) || ENDP(CDR(tail)) )
    {
        pips_user_warning("expanded sequence_statements seems empty to me\n");
    }
    else
    {
        while( CDR(CDR(tail)) != NIL ) POP(tail);
        instruction tail_ins = statement_instruction(STATEMENT(CAR(tail)));

        type treturn = ultimate_type(functional_result(type_functional(entity_type(inlined_module))));
        if( type_void_p(treturn) ) /* only replace return statement by gotos */
        {
            gen_context_recurse(expanded, tail_ins, instruction_domain, gen_true, &inline_return_remover);
        }
        else /* replace by affectation + goto */
        {
            pips_assert("returned value is a variable", type_variable_p(treturn));
            /* create new variable to receive computation result */
            returned_entity = make_new_scalar_variable(
                    get_current_module_entity(),
                    copy_basic(variable_basic(type_variable(treturn)))
            );
            AddLocalEntityToDeclarations(returned_entity, get_current_module_entity(),
                    c_module_p(get_current_module_entity())?get_current_module_statement():statement_undefined);

            gen_context_recurse(expanded, tail_ins, instruction_domain, gen_true, &inline_return_switcher);
        }
        if( !type_void_p(treturn) )
        {
            if( entity_constant_p(returned_entity) )
            {
                expression_syntax(modified_expression) = make_syntax_call(make_call(returned_entity,NIL));
            }
            else
            {
                reference r = make_reference( returned_entity, NIL);
                expression_syntax(modified_expression) = make_syntax_reference(r);
            }
        }
    }

    /* fix declarations */
    string block_level = "0";

    /* this is the only I found to recover the inlined entities' declaration */
    list formal_parameters = NIL;
    FOREACH(ENTITY,cd,code_declarations(inlined_code))
        if( entity_formal_p(cd)) formal_parameters=CONS(ENTITY,cd,formal_parameters);

    list c_iter = call_arguments(callee);
    list iter = gen_nreverse(formal_parameters);
	size_t n1 = gen_length(iter), n2 = gen_length(c_iter);
	pips_assert("function call has enough arguments",n1 >= n2);
    for( ; !ENDP(c_iter); POP(iter),POP(c_iter) )
    {
        entity e = ENTITY(CAR(iter));
        expression from = EXPRESSION(CAR(c_iter));

        /* check if there is a write effect on this parameter */
        bool need_copy = (!use_effects) || find_write_effect_on_entity(inlined_module_statement,e);

        /* generate a copy for this parameter */
        entity new = entity_undefined;
        if(need_copy)
        {
            string emn = entity_module_name(e);
            new = copy_entity(e);

            /* fix name */
            string tname = strdup(concatenate(
                        emn,
                        MODULE_SEP_STRING,
                        block_level,
                        BLOCK_SEP_STRING,
                        entity_local_name(e),
                        NULL
                        ));
            entity_name(new)=tname;

            /* fix storage */
            entity dynamic_area = global_name_to_entity(emn, DYNAMIC_AREA_LOCAL_NAME);
            entity_storage(new)= make_storage_ram(
                    make_ram(
                        get_current_module_entity(),
                        dynamic_area,
						CurrentOffsetOfArea(dynamic_area, new),
                        NIL)
            );

            /* fix value */
            entity_initial(new) = make_value_expression( copy_expression( from ) );


            /* add the entity to our list */
			statement_declarations(declaration_holder)=gen_nconc(CONS(ENTITY,new,NIL), statement_declarations(declaration_holder));
            gen_context_recurse(expanded, new, statement_domain, gen_true, &solve_name_clashes);
            replace_entity(expanded,e,new);
        }
        /* substitute variables */
        else
        {
            /* get new reference */
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
								new = make_temporary_scalar_entity(e,from);
							else
							{
								new = make_temporary_array_entity(e,from);
							}

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
						new = make_temporary_scalar_entity(e,from);
                    break;
#if 0
				case is_syntax_subscript:
					/* need a temporary variable */
					{
						if( ENDP(variable_dimensions(type_variable(entity_type(e)))) )
							new = make_temporary_scalar_entity(e,from);
						else
						{
							new = make_temporary_array_entity(e,from);
						}

					} break;
#endif

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
            gen_context_recurse(expanded, new, statement_domain, gen_true, &solve_name_clashes);
            replace_entity(expanded,e,new);

        }

    }
    gen_free_list(formal_parameters);

    /* final packing
     * we (may have) generated goto so unstructured is mandatory
     * because of the unstructuration, we must purge statement number
     */
    gen_recurse(expanded,statement_domain,gen_true,fix_sequence_statement_attributes_if_sequence);
    gen_recurse(expanded,expression_domain,gen_true,reset_expression_normalized);
    return expanded;
}


/* recursievly inline an expression if needed
 */
static
void inline_expression(expression expr, list * new_instructions)
{
    if( expression_call_p(expr) )
    {
        call callee = syntax_call(expression_syntax(expr));
        if( inline_should_inline( callee ) )
        {
                statement s = inline_expression_call( expr, callee );
                if( !statement_undefined_p(s) )
                {
                    *new_instructions = CONS(STATEMENT, s, *new_instructions);;
                }
        }
    }
}

/* check if a call has inlinable calls
 */
static
void inline_has_inlinable_calls_crawler(call callee,bool* has_inlinable_calls)
{
    (*has_inlinable_calls)|=inline_should_inline(callee);
}
static
bool inline_has_inlinable_calls(call callee)
{
    bool has_inlinable_calls=false;
    gen_context_recurse(callee, &has_inlinable_calls, call_domain, gen_true,&inline_has_inlinable_calls_crawler);
    return has_inlinable_calls;
}

/* this is in charge of replacing instruction by new ones
 * only apply if this instruction does not contain other instructions
 */
static
void inline_statement_switcher(statement stmt)
{
    instruction *ins=&statement_instruction(stmt);
    switch( instruction_tag(*ins) )
    {
        /* handle this with the expression handler */
        case is_instruction_call:
            {
                call callee =instruction_call(*ins);
                if( inline_has_inlinable_calls( callee ) )
                {
                    instruction_tag(*ins) = is_instruction_expression;
                    instruction_expression(*ins)=call_to_expression(callee);
                    inline_statement_switcher(stmt);
                }
            } break;
        /* handle those with a gen_recurse */
        case is_instruction_expression:
            {
                list new_instructions=NIL;
                gen_context_recurse(*ins,&new_instructions,expression_domain,gen_true,&inline_expression);
                if( !ENDP(new_instructions) ) /* something happens on the way to heaven */
                {
                    type t= functional_result(type_functional(entity_type(inlined_module)));
                    if( ! type_void_p(t) )
                    {
                        instruction tmp = instruction_undefined;
                        if( expression_call_p(instruction_expression(*ins)) )
                        {
                            tmp = make_instruction_call(
                                    expression_call(instruction_expression(copy_instruction(*ins)))
                            );
                        }
                        else
                        {
                            tmp = copy_instruction(*ins);
                        }
                        new_instructions=CONS(STATEMENT,
                                instruction_to_statement(tmp),
                                new_instructions
                        );
                    }
                    //free_instruction(*ins);
                    *ins=make_instruction_sequence( make_sequence( gen_nreverse(new_instructions) ) );
                    statement_number(stmt)=STATEMENT_NUMBER_UNDEFINED;
					if(!empty_comments_p(statement_comments(stmt))) free(statement_comments(stmt));
                    statement_comments(stmt)=empty_comments;
                }
            }
            break;

        default:
            break;
    };
}

/* this should replace all call to `inlined' in `module'
 * by the expansion of `inlined'
 */
static void
inline_calls(char * module)
{
    entity modified_module = module_name_to_entity(module);
    /* get target module's ressources */
    statement modified_module_statement =
        (statement) db_get_memory_resource(DBR_CODE, module, TRUE);
    pips_assert("statements found", !statement_undefined_p(modified_module_statement) );

    set_current_module_entity( modified_module );
    set_current_module_statement( modified_module_statement );

    /* inline all calls to inlined_module */
    gen_recurse(modified_module_statement, statement_domain, gen_true, &inline_statement_switcher);

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
bool do_inlining(char * module_name)
{
   /* Get the module ressource */
   inlined_module = module_name_to_entity( module_name );
   inlined_module_statement =
       (statement) db_get_memory_resource(DBR_CODE, module_name, TRUE);

   if(use_effects) set_cumulated_rw_effects((statement_effects)db_get_memory_resource(DBR_CUMULATED_EFFECTS,module_name,TRUE));

   /* check them */
   pips_assert("is a functionnal",entity_function_p(inlined_module) || entity_subroutine_p(inlined_module) );
   pips_assert("statements found", !statement_undefined_p(inlined_module_statement) );
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
       inline_calls( caller_name );
       recompile_module(caller_name);
       /* we can try to remove some labels now*/
       if( get_bool_property("INLINING_PURGE_LABELS"))
           if(!remove_useless_label(caller_name))
               pips_user_warning("failed to remove useless labels after restructure_control in inlining");
   }

   if(use_effects) reset_cumulated_rw_effects();
   inlined_module = entity_undefined;
   inlined_module_statement = statement_undefined;

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
	use_effects=true;
	return do_inlining(module_name);
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
	use_effects=false;
	return do_inlining(module_name);
}

/**  @} */

/** 
 * @name unfolding
 * @{ */

/** 
 * put c into calls_name if c is not an intrinsic
 * 
 * @param c call to check
 * @param calls_name set containing registered entity names
 */
static void
call_selector(call c, set calls_name)
{
    entity e = call_function(c);
    if( !entity_constant_p(e) && !intrinsic_entity_p(e) && !entity_symbolic_p(e) )
    {
        string name = entity_local_name(e);
        set_add_element(calls_name,calls_name,name);
    }

}

static bool statement_has_callee = false; ///< set to true when a call is found

/** 
 * get ressources for the call to inline and call
 * apropriate inlining function
 * 
 * @param caller_name calling module name
 * @param module_name called module name
 */
static void
run_inlining(string caller_name, string module_name)
{
    statement_has_callee = true;
    /* Get the module ressource */
    inlined_module = module_name_to_entity( module_name );
    inlined_module_statement =
        (statement) db_get_memory_resource(DBR_CODE, module_name, TRUE);
    if(use_effects) set_cumulated_rw_effects((statement_effects)db_get_memory_resource(DBR_CUMULATED_EFFECTS,module_name,TRUE));

    /* check them */
    pips_assert("is a functionnal",entity_function_p(inlined_module) || entity_subroutine_p(inlined_module) );
    pips_assert("statements found", !statement_undefined_p(inlined_module_statement) );

    /* inline call */
    inline_calls( caller_name );
    if(use_effects) reset_cumulated_rw_effects();
    inlined_module = entity_undefined;
    inlined_module_statement=statement_undefined;
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
bool do_unfolding(char* module_name)
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
    do {
        statement_has_callee = false;
        statement unfolded_module_statement =
            (statement) db_get_memory_resource(DBR_CODE, module_name, TRUE);
        /* gather all referenced calls */
        set calls_name = set_make(set_string);
        gen_context_recurse(unfolded_module_statement, calls_name, call_domain, gen_true, &call_selector);

        /* maybe the user put a restriction on the calls to inline ?*/
        if(!set_empty_p(unfolding_callees))
            calls_name=set_intersection(calls_name,calls_name,unfolding_callees);

        /* maybe the user used a filter ?*/
        calls_name=set_difference(calls_name,calls_name,unfolding_filters);



        /* there is something to inline */
        if( !set_empty_p(calls_name) )
        {
            SET_MAP(call_name, run_inlining(module_name,(string)call_name) ,calls_name);
            recompile_module(module_name);
            /* we can try to remove some labels now*/
            if( get_bool_property("INLINING_PURGE_LABELS"))
                if(!remove_useless_label(module_name))
                    pips_user_warning("failed to remove useless labels after restructure_control in inlining");
        }
        set_free(calls_name);
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
	use_effects=true;
	return do_unfolding(module_name);
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
	use_effects=false;
	return do_unfolding(module_name);
}
/**  @} */


/** 
 * @name outlining
 * @{ */

#define STAT_ORDER "PRETTYPRINT_STATEMENT_NUMBER"

static
void patch_outlined_reference(expression x, entity e)
{
    if(expression_reference_p(x))
    {
        reference r =expression_reference(x);
        entity e1 = reference_variable(r);
        if(same_entity_p(e,e1))
        {
			variable v = type_variable(entity_type(e));
			if( (!basic_pointer_p(variable_basic(v))) && (ENDP(variable_dimensions(v))) ) /* not an array / pointer */
			{
				expression X = make_expression(expression_syntax(x),normalized_undefined);
				expression_syntax(x)=make_syntax_call(make_call(
							CreateIntrinsic(DEREFERENCING_OPERATOR_NAME),
							CONS(EXPRESSION,X,NIL)));
			}
        }
    }

}

/** 
 * outline the statements in statements_to_outline into a module named outline_module_name
 * the outlined statements are replaced by a call to the newly generated module
 * statements_to_outline is modified in place to represent that call
 * 
 * @param outline_module_name name of the new module
 * @param statements_to_outline statements to outline into outline_module_name
 * 
 * @return pointer to the newly generated statement (already inserted in statements_to_outline)
 */
statement outliner(string outline_module_name, list statements_to_outline)
{
    pips_assert("there are some statements to outline",!ENDP(statements_to_outline));

    /* retreive referenced and declared entities */
    list referenced_entities = NIL,
         declared_entities = NIL ;
    set sreferenced_entities = set_make(set_pointer);

    list skip_list = NIL;

    FOREACH(STATEMENT, s, statements_to_outline)
    {
        set tmp = statement_get_referenced_entities(s);
        set_union(sreferenced_entities,tmp,sreferenced_entities);
        set_free(tmp);
        declared_entities =gen_nconc(declared_entities, statement_to_declarations(s));
        /* look for entity to ignore in pragma */
        list pragmas = extensions_extension(statement_extensions(s));
        FOREACH(EXTENSION,ext,pragmas) {
            if(pragma_string_p(extension_pragma(ext))) {
                string str = pragma_string(extension_pragma(ext));
                if(strstr(str,OUTLINE_IGNORE)) {
                    str+=sizeof(OUTLINE_IGNORE);
                    entity ep= FindEntity(TOP_LEVEL_MODULE_NAME,str);
                    skip_list=CONS(ENTITY,ep,skip_list);
                }
            }
        }
    }
    /* set to list */
    referenced_entities=set_to_list(sreferenced_entities);
    set_free(sreferenced_entities);

    /* get the relative complements and create the parameter list*/
    gen_list_and_not(&referenced_entities,declared_entities);
    gen_free_list(declared_entities);
    gen_list_and_not(&referenced_entities,skip_list);
    gen_free_list(skip_list);

    /* purge the functions from the parameter list, we assume they are declared externally */
    list tmp_list=NIL;
    FOREACH(ENTITY,e,referenced_entities)
        if( ! entity_function_p(e) ) tmp_list=CONS(ENTITY,e,tmp_list);
    referenced_entities=tmp_list;

    gen_sort_list(referenced_entities,(int(*)(const void*,const void*))compare_entities);


    intptr_t i=0;
    entity new_fun = make_empty_subroutine(outline_module_name);
    statement body = instruction_to_statement(make_instruction_sequence(make_sequence(statements_to_outline)));

	/* all variables are promoted parameters */
    list effective_parameters = NIL;
    list formal_parameters = NIL;
    FOREACH(ENTITY,e,referenced_entities)
    {
        type t = entity_type(e);
        if( type_variable_p(t) ) {
            /* this create the dummy parameter */
            type new_type = copy_type(t);
            entity dummy_entity = FindOrCreateEntity(
                    outline_module_name,
                    entity_user_name(e)
                    );
            entity_type(dummy_entity)=new_type;
            entity_storage(dummy_entity)=make_storage_formal(make_formal(dummy_entity,++i));


            formal_parameters=CONS(PARAMETER,make_parameter(
                        copy_type(new_type),
                        make_mode_value(), /* to be changed */
                        make_dummy_identifier(dummy_entity)),formal_parameters);
            /* this adds the effective parameter */
            effective_parameters=CONS(EXPRESSION,entity_to_expression(e),effective_parameters);
        }
    }


    /* we need to patch parameters , effective parameters and body in C
     * because of by copy function call
	 * it's not needed if 
	 * - the parameter is only read 
	 * - it's an array / pointer
     */
    if(c_module_p(get_current_module_entity()))
    {
		list iter = effective_parameters;
        FOREACH(PARAMETER,p,formal_parameters)
        {
			expression x = EXPRESSION(CAR(iter));
            entity ex = reference_variable(expression_reference(x));
            entity e = dummy_identifier(parameter_dummy(p));
            if( type_variable_p(entity_type(ex)) ) {
                variable v = type_variable(entity_type(ex));
                bool entity_written=false;
                FOREACH(STATEMENT,stmt,statements_to_outline)
                    entity_written|=find_write_effect_on_entity(stmt,ex);

                if( (!basic_pointer_p(variable_basic(v))) && 
                        ENDP(variable_dimensions(v)) &&
                        entity_written
                  )
                {
                    type t = copy_type(entity_type(e));
                    entity_type(e)=make_type_variable(
                            make_variable(
                                make_basic_pointer(t),
                                NIL,
                                NIL
                                )
                            );
                    parameter_type(p)=t;
                    syntax s = expression_syntax(x);
                    expression X = make_expression(s,normalized_undefined);
                    expression_syntax(x)=make_syntax_call(make_call(CreateIntrinsic(ADDRESS_OF_OPERATOR_NAME),CONS(EXPRESSION,X,NIL)));
                    gen_context_recurse(body,ex,expression_domain,gen_true,patch_outlined_reference);
                }
            }
			POP(iter);
        }
		pips_assert("no effective parameter left", ENDP(iter));
    }

    /* prepare parameters and body*/
    functional_parameters(type_functional(entity_type(new_fun)))=formal_parameters;
	FOREACH(PARAMETER,p,formal_parameters) {
		code_declarations(value_code(entity_initial(new_fun))) =
			gen_nconc(
					code_declarations(value_code(entity_initial(new_fun))),
					CONS(ENTITY,dummy_identifier(parameter_dummy(p)),NIL));
	}

    /* we can now begin the outlining */
    bool saved = get_bool_property(STAT_ORDER);
    set_bool_property(STAT_ORDER,false);
    text t = text_named_module(new_fun, get_current_module_entity(), body);
    add_new_module_from_text(outline_module_name, t, fortran_module_p(get_current_module_entity()));
    set_bool_property(STAT_ORDER,saved);
	/* horrible hack to prevent declaration duplication 
	 * signed : Serge Guelton
	 */
	gen_free_list(code_declarations(EntityCode(new_fun)));
	code_declarations(EntityCode(new_fun))=NIL;


    /* and return the replacement statement */
    instruction new_inst =  make_instruction_call(make_call(new_fun,gen_nreverse(effective_parameters)));
    statement new_stmt = statement_undefined;

    /* perform substitution :
     * replace the original statements by a single call
     * and patch the remaining statement (yes it's ugly)
     */
    FOREACH(STATEMENT,old_statement,statements_to_outline)
    {
        free_instruction(statement_instruction(old_statement));
        if(statement_undefined_p(new_stmt))
        {
            statement_instruction(old_statement)=new_inst;
            new_stmt=old_statement;
        }
        else
            statement_instruction(old_statement)=make_continue_instruction();
        gen_free_list(statement_declarations(old_statement));
        statement_declarations(old_statement)=NIL;
    }
    return new_stmt;
}



/**
 * @brief entry point for outline module
 * outlining will be performed using either comment recognition
 * or interactively
 *
 * @param module_name name of the module containg the statements to outline
 */
bool
outline(char* module_name)
{
    /* prelude */
    set_current_module_entity(module_name_to_entity( module_name ));
    set_current_module_statement((statement) db_get_memory_resource(DBR_CODE, module_name, TRUE) );
 	set_cumulated_rw_effects((statement_effects)db_get_memory_resource(DBR_PROPER_EFFECTS, module_name, TRUE));

    /* retrieve name of the outiled module */
    string outline_module_name = get_string_property_or_ask("OUTLINE_MODULE_NAME","outline module name ?\n");

    /* retrieve statement to outline */
    list statements_to_outline = find_statements_with_pragma(get_current_module_statement(),OUTLINE_PRAGMA) ;
    if(ENDP(statements_to_outline)) {
        if( empty_string_p(get_string_property("OUTLINE_LABEL")) ) {
            statements_to_outline=find_statements_interactively(get_current_module_statement());
        }
        else  {
            string stmt_label=get_string_property("OUTLINE_LABEL");
            entity stmt_label_entity = find_label_entity(module_name,stmt_label);
            if(entity_undefined_p(stmt_label_entity))
                pips_user_error("label %s not found\n", stmt_label);
            statements_to_outline = find_statements_with_label(get_current_module_statement(),stmt_label_entity);
        }
    }

    /* apply outlining */
    (void)outliner(outline_module_name,statements_to_outline);


    /* validate */
    module_reorder(get_current_module_statement());
    DB_PUT_MEMORY_RESOURCE(DBR_CODE, module_name,
			   get_current_module_statement());
    DB_PUT_MEMORY_RESOURCE(DBR_CALLEES, module_name, compute_callees(get_current_module_statement()));

    /*postlude*/
	reset_cumulated_rw_effects();
    reset_current_module_entity();
    reset_current_module_statement();

    return true;
}
/**  @} */
