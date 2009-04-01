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
        free_expression(instruction_return(ins));
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
static effect
find_effect_on_entity(statement s, entity e)
{
    list cummulated_effects = load_cumulated_rw_effects_list( s );
    effect found = effect_undefined;
    MAP(EFFECT, eff,
    {
        reference r = effect_any_reference(eff);
        entity re = reference_variable(r);
        if( same_entity_name_p(e,re) )
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
        if( same_entity_name_p(referenced_entity , thecouple->old) )
        {
            if( entity_constant_p(thecouple->new) )
            {
                expression_syntax(exp) = make_syntax_call(make_call(thecouple->new,NIL));
            }
            else
            {
                reference_variable(ref) = thecouple->new;
            }
        }
    }
}

static void reset_expression_normalized(expression e)
{
    if(!normalized_undefined_p(expression_normalized(e)))
        free_normalized(expression_normalized(e));
    expression_normalized(e)=normalized_undefined;
}

static void
do_substitute_all_entities(statement s, struct entity_pair* thecouple)
{
    FOREACH(ENTITY,decl_ent,statement_declarations(s))
    {
        value v = entity_initial(decl_ent);
        if( !value_undefined_p(v) && value_expression_p( v ) )
            gen_context_recurse( v, thecouple, expression_domain, gen_true, do_substitute_entity);
    }
}

/* substitute `thecouple->new' to `thecouple->old' in `s'
 */
void
substitute_entity(statement s, entity old, entity new)
{
    struct entity_pair thecouple = { old, new };

    gen_context_multi_recurse( s, &thecouple, expression_domain, gen_true, do_substitute_entity,
            statement_domain,gen_true, do_substitute_all_entities, NULL);

}


/* look for entity locally named has `new' in statements `s'
 * when found, fidn a new name and perform substitution
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


    /* create the new instruction sequence 
     * no need to change all entities in the new statements, because we build a new text ressource latter
     */
    statement expanded = copy_statement(inlined_module_statement);
    statement_declarations(expanded) = gen_full_copy_list( statement_declarations(expanded) ); // simple copy != deep copy

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
    gen_recurse(expanded,expression_domain,gen_true,reset_expression_normalized);
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
                {
                    *new_instructions = CONS(STATEMENT, instruction_to_statement(i), *new_instructions);
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
        case is_instruction_return:
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
                }
            }
            break;

        default:
            break;
    };
}

static void
clean_unused_entities(entity e)
{
    string s=entity_module_name(e);
    string ref = entity_local_name(get_current_module_entity());
    if( same_string_p(s,ref))
        gen_clear_tabulated_element((gen_chunk*)e);
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

    /* perform restructuring and cleaning
     * SG: this may not be needed ...
     */
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

/* build a textual representation of the modified module and update db
 * SG: this code should not be there
 */
void
recompile_module(char* module)
{
    entity modified_module = module_name_to_entity(module);
    statement modified_module_statement =
        (statement) db_get_memory_resource(DBR_CODE, module, TRUE);

    set_current_module_entity( modified_module );
    set_current_module_statement( modified_module_statement );

    /* build and register textual representation */
    text t = text_module(get_current_module_entity(), modified_module_statement);
    string dirname = db_get_current_workspace_directory();
    string res = fortran_module_p(modified_module)? DBR_INITIAL_FILE : DBR_C_SOURCE_FILE;
    string filename = db_get_file_resource(res,module,TRUE);
    string fullname = strdup(concatenate(dirname, "/",filename, NULL));
    FILE* f = safe_fopen(fullname,"w");
    print_text(f,t);
    fclose(f);
    DB_PUT_FILE_RESOURCE(res,module,filename);

    /* the module will be reparsed, so fix(=delete) all its previous entites */
    {
        list p = NIL;
        FOREACH(ENTITY, e, entity_declarations(modified_module))
        {
            if(! area_entity_p(e) )
                gen_clear_tabulated_element((gen_chunk*)e);

            else
                p = CONS(ENTITY,e,p);
        }
        entity_declarations(modified_module) = gen_nreverse(p);
        code_initializations(value_code(entity_initial(modified_module)))=make_sequence(NIL);
    }

    reset_current_module_entity();
    reset_current_module_statement();

    bool parsing_ok =(fortran_module_p(modified_module)) ? parser(module) : c_parser(module);
    if(!parsing_ok)
        pips_user_error("failed to recompile module");
    if(!controlizer(module))
        pips_user_warning("failed to apply controlizer after recompilation");

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
   }

   reset_cumulated_rw_effects();
   inlined_module = entity_undefined;
   inlined_module_statement = statement_undefined;

   pips_debug(2, "inlining done for %s\n", module_name);
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
call_selector(call c, set calls_name)
{
    entity e = call_function(c);
    if( !entity_constant_p(e) && !intrinsic_entity_p(e) && !entity_symbolic_p(e) )
    {
        string name = entity_local_name(e);
        if( !set_belong_p(calls_name,name) )
        {
            set_add_element(calls_name,calls_name,name);
        }
    }

}

static bool statement_has_callee = false;

/* get ressources for the call to inline and call
 * apropriate inlining function
 */
static void
run_inlining(string caller_name, string module_name)
{
    statement_has_callee = true;
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
    inlined_module = entity_undefined;
    inlined_module_statement=statement_undefined;
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

    debug_on("UNFOLDING_DEBUG_LEVEL");

    /* parse filter property */
    string unfolding_filter_names = strdup(get_string_property("UNFOLDING_FILTER"));
    set unfolding_filters = set_make(set_string);

    string filter_name= NULL;
    for(filter_name = strtok(unfolding_filter_names," ") ; filter_name ; filter_name=strtok(NULL," ") )
    {
        set_add_element(unfolding_filters, unfolding_filters, filter_name);
        recompile_module(module_name);
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
        }
        set_free(calls_name);
    } while(statement_has_callee);

    set_free(unfolding_filters);
    free(unfolding_filter_names);


    pips_debug(2, "unfolding done for %s\n", module_name);

    debug_off();
    return true;
}


/*********************************************************************************************
 *
 *  outlining part
 *
 */

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
            expression X = make_expression(expression_syntax(x),normalized_undefined);
            expression_syntax(x)=make_syntax_call(make_call(
                    CreateIntrinsic(DEREFERENCING_OPERATOR_NAME),
                    CONS(EXPRESSION,X,NIL)));
        }
    }

}

statement outliner(string outline_module_name, list statements_to_outline)
{
    pips_assert("there are some statements to outline",!ENDP(statements_to_outline));

    /* retreive referenced and declared entities */
    list referenced_entities = NIL,
         declared_entities = NIL ;
    FOREACH(STATEMENT, s, statements_to_outline)
    {
        referenced_entities=gen_nconc(referenced_entities, statement_to_referenced_entities(s));
        declared_entities =gen_nconc(declared_entities, statement_to_declarations(s));
    }

    /* get the relative complements and create the parameter list*/
    gen_list_and_not(&referenced_entities,declared_entities);

    list parameters = NIL;
    intptr_t i=0;
    set enames = set_make(set_string);
    entity new_fun = make_empty_subroutine(outline_module_name);
    statement body = instruction_to_statement(make_instruction_sequence(make_sequence(statements_to_outline)));

    list effective_parameters = NIL;
    FOREACH(ENTITY,e,referenced_entities)
    {
        entity dummy_entity = FindOrCreateEntity(
                outline_module_name,
                entity_user_name(e)
        );
        if(!set_belong_p(enames,entity_name(dummy_entity)))
        {
            set_add_element(enames,enames,entity_name(dummy_entity));
            entity_type(dummy_entity)=copy_type(entity_type(e));
            entity_storage(dummy_entity)=make_storage_formal(make_formal(dummy_entity,++i));


            parameters=CONS(PARAMETER,make_parameter(
                        copy_type(entity_type(e)),
                        make_mode_value(), /* to be changed */
                        make_dummy_identifier(dummy_entity)),parameters);
            AddEntityToDeclarations(dummy_entity,new_fun);

            effective_parameters=CONS(EXPRESSION,entity_to_expression(e),effective_parameters);

            /* we need to patch this fo C , see below*/
            if(c_module_p(get_current_module_entity()))
                gen_context_recurse(body,e,expression_domain,gen_true,patch_outlined_reference);
        }
    }
    set_free(enames);



    /* we need to patch parameters , effective parameters and body in C
     * because of by copy function call
     */
    if(c_module_p(get_current_module_entity()))
    {
        FOREACH(PARAMETER,p,parameters)
        {
            entity e = dummy_identifier(parameter_dummy(p));
            type t = copy_type(entity_type(e));
            entity_type(e)=make_type_variable(
                    make_variable(
                        make_basic_pointer(t),
                        NIL,
                        NIL
                        )
                    );
            parameter_type(p)=t;
        }
        FOREACH(EXPRESSION,x,effective_parameters)
        {
            syntax s = expression_syntax(x);
            expression X = make_expression(s,normalized_undefined);
            expression_syntax(x)=make_syntax_call(make_call(CreateIntrinsic(ADDRESS_OF_OPERATOR_NAME),CONS(EXPRESSION,X,NIL)));
        }
    }

    /* prepare paramters and body*/
    functional_parameters(type_functional(entity_type(new_fun)))=parameters;

    /* we can now begin the outlining */
    bool saved = get_bool_property(STAT_ORDER);
    set_bool_property(STAT_ORDER,false);
    text t = text_named_module(new_fun, get_current_module_entity(), body);
    add_new_module_from_text(outline_module_name, t, fortran_module_p(get_current_module_entity()));
    set_bool_property(STAT_ORDER,saved);

    /* and return the replacement statement */
    instruction new_inst =  make_instruction_call(make_call(new_fun,effective_parameters));
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


static
bool interactive_statement_picker(statement s,bool started)
{
    string statement_text = text_to_string(text_statement(get_current_module_entity(),1,s));
    string answer = string_undefined;
    do {
        while( string_undefined_p(answer) || empty_string_p(answer)  )
        {
            answer = user_request("%s\n%s\n%s\n%s\n%s\n",
                "Do you want to pick the following statement ?",
                "**********************",
                statement_text,
                "**********************",
                "[y/n] ?"
            );
        }
        if( answer[0]!='y' && answer[0]!='n' )
        {
            pips_user_warning("answer by 'y' or 'n' !\n");
            free(answer);
            answer=string_undefined;
        }
    } while(string_undefined_p(answer));

    if(!started &&  answer[0]=='y') started=true;

    return  answer[0]=='y';
}

static
void statement_walker(statement s, list* l, bool (*picker)(statement,bool), bool started )
{
    if( picker(s,started) ) 
    {
        *l=CONS(STATEMENT,s,*l);
        started=true;
    }
    else if( !started )
    {
        instruction i = statement_instruction(s);
        switch(instruction_tag(i))
        {
            case is_instruction_sequence:
                {
                    FOREACH(STATEMENT,S,sequence_statements(instruction_sequence(i)))
                        statement_walker(S,l,picker,started);
                } break;
            case is_instruction_test:
                {
                    statement_walker(test_true(instruction_test(i)),l,picker,started);
                    statement_walker(test_false(instruction_test(i)),l,picker,started);
                } break;
            case is_instruction_loop:
                statement_walker(loop_body(instruction_loop(i)),l,picker,started);
                break;
            case is_instruction_whileloop:
                statement_walker(whileloop_body(instruction_whileloop(i)),l,picker,started);
                break;
            case is_instruction_multitest:
                statement_walker(multitest_body(instruction_multitest(i)),l,picker,started);
                break;
            case is_instruction_forloop:
                statement_walker(forloop_body(instruction_forloop(i)),l,picker,started);
                break;
            case is_instruction_unstructured:
            case is_instruction_return:
            case is_instruction_expression:
            case is_instruction_goto:
            case is_instruction_call:
                break;
        };
    }
}


/** 
 * @brief entry point for outline module
 * outlining will be performed using either pragma recognition
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

    /* retrieve name of the outiled module */
    string outline_module_name = string_undefined;
    do {
        outline_module_name = user_request("outline module name ?\n");// should check
    } while( string_undefined_p(outline_module_name) || outline_module_name[0] == '\0' );
    

    /* retrieve statement to outline */
    list statements_to_outline = NIL;
    statement_walker(get_current_module_statement(),&statements_to_outline,&interactive_statement_picker,false);
    /* we may want to try another picker later ;-) */


    /* may need a sort */
    statements_to_outline=gen_nreverse(statements_to_outline);

    /* apply outlining */
    statement new_stmt = outliner(outline_module_name,statements_to_outline);


    /* validate */
    module_reorder(get_current_module_statement());
    DB_PUT_MEMORY_RESOURCE(DBR_CODE, module_name, get_current_module_statement());
    DB_PUT_MEMORY_RESOURCE(DBR_CALLEES, module_name, compute_callees(get_current_module_statement()));

    /*postlude*/
    reset_current_module_entity();
    reset_current_module_statement();

    return true;
}
