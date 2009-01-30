/** 
 * @file inlining.c
 * @brief add inlining support to pips, with two flavors
 *  - inlining(char* module) to inline all calls to a module
 *  - unfolding(char* module) to inline all call in a module
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
#include "misc.h"
#include "control.h"
#include "effects-generic.h"
#include "preprocessor.h"
#include "transformations.h"


/* FOREACH, similar to MAP but more gdb (and vim) friendly
 */

#define UNIQUE_NAME_1(prefix, x)   prefix##x
#define UNIQUE_NAME_2(prefix, x)   UNIQUE_NAME_1 (prefix, x)
#define UNIQUE_NAME  UNIQUE_NAME_2 (iter_, __LINE__)

#if __STDC_VERSION__ >= 199901L
#define FOREACH(_fe_CASTER, _fe_item, _fe_list) \
        list UNIQUE_NAME = (_fe_list);\
for( _fe_CASTER##_TYPE _fe_item;\
        !ENDP(UNIQUE_NAME) && (_fe_item= _fe_CASTER(CAR(UNIQUE_NAME) ));\
        POP(UNIQUE_NAME))
#else
#define FOREACH(_fe_CASTER, _fe_item, _fe_list) \
        list UNIQUE_NAME;\
        _fe_CASTER##_TYPE _fe_item;\
for( UNIQUE_NAME= (_fe_list);\
        !ENDP(UNIQUE_NAME) && (_fe_item= _fe_CASTER(CAR(UNIQUE_NAME) ));\
        POP(UNIQUE_NAME))
#endif


static entity       inlined_module;
static statement    inlined_module_statement;
static statement    laststmt;
static entity       returned_entity;

/* replace return instruction by a goto
 */
static
void inline_return_remover(instruction ins,instruction tail_ins)
{
    if( return_instruction_p( ins ) && ins !=tail_ins )
    {
        *ins = *make_instruction_goto( copy_statement(laststmt) );
    }
}

/* replace return instruction by an assignment and a goto
 */
static
void inline_return_switcher(instruction ins,instruction tail_ins)
{
    if( return_instruction_p( ins ) )
    {
        call c = make_call(
                    CreateIntrinsic(ASSIGN_OPERATOR_NAME),
                    CONS(
                        EXPRESSION,
                        entity_to_expression( returned_entity ),
                        call_arguments(instruction_call(ins))
                    )
                 );
        list l= (ins == tail_ins ) ? NIL : CONS( STATEMENT, instruction_to_statement( make_instruction_goto( copy_statement(laststmt) ) ) , NIL ) ;
        l = CONS( STATEMENT, instruction_to_statement(  make_instruction_call(c) ), l );

        sequence s = make_sequence( l );
        *ins = *make_instruction_sequence( s );
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
static effect
find_effect_on_entity(statement s, entity e)
{
    list cummulated_effects = load_cumulated_rw_effects_list( s );
    effect found = effect_undefined;
    MAP(EFFECT, eff,
    {
        reference r = effect_any_reference(eff);
        entity re = reference_variable(r);
        if( same_entity_p(e,re) )
        {
           cell c = effect_cell(eff);
           found = (reference_indices( cell_preference_p(c) ? preference_reference(cell_preference(c)) : cell_reference(c) ) == NIL) ? eff :found ;
        }
    }, cummulated_effects);
    return found;
}

struct entity_pair
{
    entity old;
    entity new;
};

/* substitute `thecouple->new' to `thecouple->old' in `exp'
 * if `exp' is a reference
 */
static void
do_substitute_entity(expression exp, struct entity_pair* thecouple)
{
    if( expression_reference_p(exp) )
    {
        reference ref = syntax_reference(expression_syntax(exp));
        entity referenced_entity = reference_variable(ref);
        if( same_entity_p(referenced_entity , thecouple->old) )
        {
             reference_variable(ref) = thecouple->new;
        }
    }
}

/* substitute `thecouple->new' to `thecouple->old' in `s'
 */
void
substitute_entity(statement s, entity old, entity new)
{
    struct entity_pair thecouple = { old, new };

    gen_context_recurse( s, &thecouple, expression_domain, gen_true, do_substitute_entity);

    MAP(ENTITY,decl_ent,
    {
        value v = entity_initial(decl_ent);
        if( !value_undefined_p(v) && value_expression_p( v ) )
            gen_context_recurse( value_expression(v), &thecouple, expression_domain, gen_true, do_substitute_entity);
    }, statement_declarations(s) );
}

/* look for entity locally named has `new' in statements `s'
 * when found, fidn a new name and perform substitution
 */
static void
solve_name_clashes(statement s, entity new)
{
    list l = statement_declarations(s);
    for(;l!=NIL;POP(l))
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
            substitute_entity(s,decl_ent,solve_clash);
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


/* this should inline the call callee
 * calling module inlied_module
 */
static 
instruction inline_expression_call(expression modified_expression, call callee)
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
            return instruction_undefined;
        }
    }

    /* should add them to modified_module's*/
    list inlined_extern_declaration = code_externs(inlined_code);
    pips_assert("no external declaration",inlined_extern_declaration == NIL);


    /* the new instruction sequence */
    statement expanded = copy_statement( inlined_module_statement );

    /* fix block status */
    if( ! statement_block_p( expanded ) )
    {
        instruction i = make_instruction_sequence( make_sequence( CONS(STATEMENT,expanded,NIL) ) );

        expanded = instruction_to_statement( i );
    }

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
            entity dynamic_area =
                global_name_to_entity(module_local_name(get_current_module_entity()),DYNAMIC_AREA_LOCAL_NAME);
            entity_storage(returned_entity) = make_storage_ram(
                    make_ram(get_current_module_entity(),
                       dynamic_area,
                      add_any_variable_to_area(dynamic_area,returned_entity, c_module_p(get_current_module_entity())),
                     NIL)); 
            AddLocalEntityToDeclarations(returned_entity, get_current_module_entity(),
                    c_module_p(get_current_module_entity())?get_current_module_statement():statement_undefined);

            gen_context_recurse(expanded, tail_ins, instruction_domain, gen_true, &inline_return_switcher);
        }
        if( !type_void_p(treturn) )
            *modified_expression = *entity_to_expression(returned_entity);
    }

    /* fix declarations */
    string block_level = "0";

    /* this is the only I found to recover the inlined entities' declaration */
    list iter = code_declarations(inlined_code); 
    while( !ENTITY_NAME_P( ENTITY(CAR(iter)), DYNAMIC_AREA_LOCAL_NAME ) ) POP(iter);
    POP(iter);/*pop the dynamic area label*/
    if( !ENDP(iter) && ENTITY_NAME_P( ENTITY(CAR(iter)), entity_user_name(inlined_module) ) )
            POP(iter); /* pop the first flag if needed */

    list adder = /*statement_declarations(expanded);*/NIL;
    list c_iter = call_arguments(callee);
    for( ; !ENDP(c_iter); POP(iter),POP(c_iter) )
    {
        entity e = ENTITY(CAR(iter));
        expression from = EXPRESSION(CAR(c_iter));

        /* check if there is a write effect on this parameter */
        effect eff = find_effect_on_entity(inlined_module_statement,e);
        bool need_copy = true;
        if(  ! effect_undefined_p(eff) )
        {
            /* skip if expression is complicated
             * it could handle the & or * operator, but some extra parenthesis would be needed
             * just as they are needed when writing macro functions in fact
             */
            if( expression_constant_p(from) || expression_reference_p(from))
            {
                action a = effect_action(eff);
                need_copy= action_write_p(a);
            }
        }
        else
        {
            need_copy =false;
        }
        /* generate a copy for this parameter */
        if(need_copy)
        {
            string emn = entity_module_name(e);
            entity new_ent = copy_entity(e);

            /* fix name */
            string tname = strdup(concatenate(
                        emn,
                        MODULE_SEP_STRING,
                        block_level,
                        BLOCK_SEP_STRING,
                        entity_local_name(e),
                        NULL
                        ));
            entity_name(new_ent)=tname;

            /* fix storage */
            entity dynamic_area = global_name_to_entity(emn, DYNAMIC_AREA_LOCAL_NAME);
            entity_storage(new_ent)= make_storage_ram(
                    make_ram(
                        get_current_module_entity(),
                        dynamic_area,
                        add_any_variable_to_area(dynamic_area,new_ent, c_module_p(get_current_module_entity())),
                        NIL)
            );

            /* fix value */
            entity_initial(new_ent) = make_value_expression( copy_expression( from ) );

            /* add the entity to our list */
            adder=CONS(ENTITY,new_ent,adder);
        }
        /* substitute variables */
        else
        {
            /* get new reference */
            entity new = entity_undefined;
            switch(syntax_tag(expression_syntax(from)))
            {
                case is_syntax_reference:
                    new = reference_variable(syntax_reference(expression_syntax(from)));
                    break;
                    /* this one is more complicated than I thought,
                     * what of the side effect of the call ?
                     * we must create a new variable holding the call result before
                     */
                case is_syntax_call:
                    if( expression_constant_p(from) )
                    {
                        new = call_function(expression_call(from));
                    }
                    else
                    {
                        new = make_new_scalar_variable(
                                get_current_module_entity(),
                                copy_basic(variable_basic(type_variable(entity_type(e))))
                                );
                        entity dynamic_area =
                            global_name_to_entity(module_local_name(get_current_module_entity()),DYNAMIC_AREA_LOCAL_NAME);
                        entity_storage(new) = make_storage_ram(
                                make_ram(get_current_module_entity(),
                                    dynamic_area,
                                    add_any_variable_to_area(dynamic_area,returned_entity, c_module_p(get_current_module_entity())),                     
                                    NIL));
                        entity_initial(new) = make_value_expression(from);
                        AddLocalEntityToDeclarations(new, get_current_module_entity(),
                                c_module_p(get_current_module_entity())?get_current_module_statement():statement_undefined);
                    } break;
                default:
                    pips_internal_error("unhandled tag %d\n", syntax_tag(expression_syntax(from)) );
            };

            /* check wether the substitution will cause naming clashes 
             * then perform the substitution
             */
            gen_context_recurse(expanded, new, statement_domain, gen_true, &solve_name_clashes);
            substitute_entity(expanded,e,new);

        }

    }
    if( adder !=NIL)
    {
        adder=gen_nreverse(adder);
        adder=gen_nconc(adder, statement_declarations(expanded));
        statement_declarations(expanded)=adder;
    }

    /* final packing
     * we (may have) generated goto so unstructured is mandatory
     * because of the unstructuration, we must purge statement number
     */
    gen_recurse(expanded,statement_domain,gen_true,fix_sequence_statement_attributes_if_sequence);
    unstructured u = control_graph(expanded);
    instruction ins = make_instruction_unstructured(u);

    return ins;

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
                instruction i = inline_expression_call( expr, callee );
                if( !instruction_undefined_p(i) )
                    *new_instructions = CONS(STATEMENT, instruction_to_statement(i), *new_instructions);
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
    instruction* ins=&statement_instruction(stmt);
    switch( instruction_tag(*ins) )
    {
        /* handle this with the expression handler */
        case is_instruction_call:
            {
                call callee =instruction_call(*ins);
                if( inline_has_inlinable_calls( callee ) )
                {
                    //free_instruction(*ins);
                    *ins= make_instruction_expression( call_to_expression(callee) );
                    inline_statement_switcher(stmt);
                }
            } break;
        /* handle those with a gen_recurse */
        case is_instruction_return:
        case is_instruction_expression:
            {
                list new_instructions=NIL;
                gen_context_recurse(*ins,&new_instructions,expression_domain,gen_true,&inline_expression);
                if( new_instructions != NIL ) /* something happens on the way to heaven */
                {
                    type t= functional_result(type_functional(entity_type(inlined_module)));
                    if( ! type_void_p(t) )
                    {
                        instruction tmp = *ins;
                        if( expression_call_p(instruction_expression(tmp)) )
                        {
                            tmp = make_instruction_call(
                                    expression_call(instruction_expression(tmp))
                            );
                        }
                        new_instructions=CONS(STATEMENT,
                                instruction_to_statement(copy_instruction(tmp)),
                                new_instructions
                        );
                    }
                    //free_instruction(*ins);
                    *ins = make_instruction_sequence( make_sequence( gen_nreverse(new_instructions) ) );
                    statement_number(stmt)=STATEMENT_NUMBER_UNDEFINED;
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

    /* restucture the generated unstructured statement */
    if(!restructure_control(module))
        pips_user_warning("failed to restructure after inlining");
    /* we can try to remove some labels now*/
    if(!remove_useless_label(module))
        pips_user_warning("failed to remove useless labels after restructure_control in inlining");

}

/* this should inline all calls to module `module_name'
 * in calling modules, if possible ...
 */
bool
inlining(char * module_name)
{
   /* Get the module ressource */
   inlined_module = module_name_to_entity( module_name );
   inlined_module_statement = 
       (statement) db_get_memory_resource(DBR_CODE, module_name, TRUE);

   set_cumulated_rw_effects((statement_effects)db_get_memory_resource(DBR_CUMULATED_EFFECTS,module_name,TRUE));

   /* check them */
   pips_assert("is a functionnal",entity_function_p(inlined_module) || entity_subroutine_p(inlined_module) );
   pips_assert("statements found", !statement_undefined_p(inlined_module_statement) );
   debug_on("INLINING_DEBUG_LEVEL");

   /* get module's callers */
   callees callers = (callees)db_get_memory_resource(DBR_CALLERS, module_name, TRUE);
   list callers_l = callees_callees(callers);

   /* inline call in each caller */
   MAP(STRING, caller_name, inline_calls( caller_name ) , callers_l );

   reset_cumulated_rw_effects();

   debug(2, "inlining", "done for %s\n", module_name);
   debug_off();
   /* Should have worked: */
   return TRUE;
}



/**************************************************************************************
 *
 *                  UNFOLD SECTION
 *
 **************************************************************************************/

/* select every call that is not an intrinsic
 */
static void 
call_selector(call c, list* calls_name)
{
    entity e = call_function(c);
    if( !entity_constant_p(e) && !intrinsic_entity_p(e) && !entity_symbolic_p(e) )
    {
        string name = entity_local_name(e);
        *calls_name= CONS(STRING,name,*calls_name);
    }

}

/* get ressources for the call to inline and call
 * apropriate inlining function
 */
static void
run_inlining(string caller_name, string module_name)
{
   /* Get the module ressource */
   inlined_module = module_name_to_entity( module_name );
   inlined_module_statement = 
       (statement) db_get_memory_resource(DBR_CODE, module_name, TRUE);
   set_cumulated_rw_effects((statement_effects)db_get_memory_resource(DBR_CUMULATED_EFFECTS,module_name,TRUE));

   /* check them */
   pips_assert("is a functionnal",entity_function_p(inlined_module) || entity_subroutine_p(inlined_module) );
   pips_assert("statements found", !statement_undefined_p(inlined_module_statement) );

   /* inline call */
   inline_calls( caller_name );
   reset_cumulated_rw_effects();
}

/* this should inline all call in module `module_name'
 * it does not works recursievly, so multiple pass may be needed
 * returns true if at least one function has been inlined
 */
bool
unfolding(char* module_name)
{
   /* Get the module ressource */
   entity unfolded_module = module_name_to_entity( module_name );
   statement unfolded_module_statement = 
       (statement) db_get_memory_resource(DBR_CODE, module_name, TRUE);

   /* check them */
   pips_assert("is a functionnal",entity_function_p(unfolded_module) || entity_subroutine_p(unfolded_module) );
   pips_assert("statements found", !statement_undefined_p(unfolded_module_statement) );
   debug_on("UNFOLDING_DEBUG_LEVEL");

   /* gather all referenced calls */
   list calls_name = NIL;
   gen_context_recurse(unfolded_module_statement, &calls_name, call_domain, gen_true, &call_selector);

   /* there is something to inline */
   if( !ENDP(calls_name) ) 
   {
       MAP(STRING, call_name,
       { run_inlining(module_name,call_name);
       },calls_name);
   }
   else
   {
       debug(1, "unfolding", "no function call in %s\n", module_name);
   }
   /* remove all comments from inlined statement,
    * because those comment could have a corrupted meaning
    */
   //gen_recurse(unfolded_module_statement,statement_domain,gen_true,remove_comments);
    /* Reorder the module, because new statements have been added */  
    module_reorder(unfolded_module_statement);
    DB_PUT_MEMORY_RESOURCE(DBR_CODE, module_name, unfolded_module_statement);

   debug(2, "unfolding", "done for %s\n", module_name);

   debug_off();
   return !ENDP(calls_name);
}
