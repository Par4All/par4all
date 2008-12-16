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
#include "transformations.h"



static entity       inlined_module;
static statement    inlined_module_statement;
static statement    modified_module_statement;
static entity       modified_module;
static size_t       block_level;
static statement    laststmt;
static entity       returned_entity;
static instruction  tail_ins;

static list         new_instructions;

static
bool has_static_statements(list decl)
{
    decl=decl;
    return FALSE;
}

static
void inline_return_remover(instruction ins)
{
    if( instruction_call_p(ins) &&
        same_string_p( entity_local_name(call_function(instruction_call(ins))), RETURN_FUNCTION_NAME ) &&
        ins !=tail_ins    )
    {
        *ins = *make_instruction_goto( copy_statement(laststmt) );
    }
}

static
void inline_return_switcher(instruction ins)
{
    if( instruction_call_p(ins) &&
            same_string_p( entity_local_name(call_function(instruction_call(ins))), RETURN_FUNCTION_NAME ) )
    {
        expression e = MakeBinaryCall( CreateIntrinsic(ASSIGN_OPERATOR_NAME) , entity_to_expression( returned_entity ) , EXPRESSION(CAR(call_arguments(instruction_call(ins)))));

        list l= (ins == tail_ins ) ? NIL : CONS( STATEMENT, instruction_to_statement( make_instruction_goto( copy_statement(laststmt) ) ) , NIL ) ;
        l = CONS( STATEMENT, instruction_to_statement(  make_instruction_expression( e ) ), l );

        sequence s = make_sequence( l );
        *ins = *make_instruction_sequence( s );
    }
}

static
bool inline_should_inline(call callee)
{
    return call_function(callee) == inlined_module ;
}

/* this should inline the call callee
 * calling module inlied_module
 */
static 
instruction inline_expression_call(expression modified_expression, call callee)
{

    /* only inline the right call */
    pips_assert("inline the right call",inline_should_inline(callee));

    string modified_module_name = entity_local_name(modified_module);

    /* cannot inline if static variable are referenced
     * if the functionnal has static declaration, really impossible
     * if the module has static declarations, we could make this declaration
     *      global ...
     */
    value inlined_value = entity_initial(inlined_module);
    pips_assert("is a code", value_code_p(inlined_value));
    code inlined_code = value_code(inlined_value);

    /* should check for static declarations ... */
    list inlined_declaration = code_declarations(inlined_code);
    if( has_static_statements( inlined_declaration ) )
    {
        return NULL;
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
    laststmt=instruction_to_statement( make_instruction_expression( int_to_expression(0) ) );
    statement_label(laststmt) = make_new_label( modified_module_name ); 
    gen_nconc( sequence_statements(instruction_sequence(statement_instruction(expanded))), CONS( STATEMENT, laststmt, NIL) );

    /* fix `return' calls
     * in case a goto is immediatly followed by its targeted label
     * the goto is not needed (SG: seems difficult to do in the previous gen_recurse)
     */
    list tail = sequence_statements(instruction_sequence(statement_instruction(expanded)));
    pips_assert("expanded sequence_statements has at least a label and a statement",CDR(tail)!=NIL);
    while( CDR(CDR(tail)) != NIL ) POP(tail);
    tail_ins = statement_instruction(STATEMENT(CAR(tail)));

    type treturn = functional_result(type_functional(entity_type(inlined_module)));
    if( type_void_p(treturn) ) /* only replace return statement by gotos */
    {
        gen_recurse(expanded, instruction_domain, gen_true, &inline_return_remover);
    }
    else /* replace by affectation + goto */
    {
        /* create new variable to receive computation result */
        string mname = entity_local_name(inlined_module);
        size_t esize = strlen(mname) + 1 + 4 ;
        static unsigned int counter = 0;
        pips_assert("counter does not overflow",counter < 10000);
        string ename = malloc(sizeof(*ename)*esize);
        pips_assert("allocation ok",ename);
        snprintf(ename,esize,"%s%u",mname,counter++);

        string tname = strdup(concatenate(
                    modified_module_name,
                    MODULE_SEP_STRING,
                    ename,
                    NULL
                    ));
        free(ename);

        /* the entity */
        returned_entity=make_entity(
                tname,
                copy_type(treturn),
                storage_undefined,
                value_undefined
                );
        /* fix storage */
        entity dynamic_area = global_name_to_entity(modified_module_name, DYNAMIC_AREA_LOCAL_NAME);
        area aa = type_area(entity_type(dynamic_area));
        int OldOffset = area_size(aa);
        area_size(aa) = OldOffset + basic_type_size( variable_basic(type_variable(treturn)) );
        area_layout(aa) = gen_nconc(area_layout(aa), CONS(ENTITY, returned_entity, NIL));
        entity_storage(returned_entity) = make_storage_ram( make_ram( modified_module, dynamic_area, OldOffset, NIL ) );
        /* push it into the declaration list*/
        list adder = statement_declarations(modified_module_statement);
        adder=CONS(ENTITY, returned_entity,adder);
        statement_declarations(modified_module_statement) = adder;

        gen_recurse(expanded, instruction_domain, gen_true, &inline_return_switcher);
    }


    /* fix declarations */
    string block_level = "0";

    list iter = inlined_declaration; // << this is the only I found to recover the inlined entitties' declaration
    while( !same_string_p( entity_local_name(ENTITY(CAR(iter))), DYNAMIC_AREA_LOCAL_NAME ) ) POP(iter);

    list adder = statement_declarations(expanded);
    list c_iter = call_arguments(callee);
    for( POP(iter)/*to pop the dynamic area label*/ ; !ENDP(c_iter); POP(iter),POP(c_iter) )
    {
        entity e = ENTITY(CAR(iter));
        expression from = EXPRESSION(CAR(c_iter));

        string emn = entity_module_name(e);
        entity new_ent = copy_entity(e);

        /* fix name */
        string tname = strdup(concatenate(
                    modified_module_name,
                    MODULE_SEP_STRING,
                    block_level,
                    BLOCK_SEP_STRING,
                    entity_local_name(e),
                    NULL
        ));
        entity_name(new_ent)=tname;

        /* fix storage */
        entity dynamic_area = global_name_to_entity(emn, DYNAMIC_AREA_LOCAL_NAME);
        area aa = type_area(entity_type(dynamic_area));
        int OldOffset = area_size(aa);
        area_size(aa) = OldOffset + basic_type_size( basic_of_expression( from ) );
        area_layout(aa) = gen_nconc(area_layout(aa), CONS(ENTITY, new_ent, NIL));
        entity_storage(new_ent)= make_storage_ram( make_ram( modified_module, dynamic_area, OldOffset, NIL ) );

        /* fix value */
        entity_initial(new_ent) = make_value_expression( copy_expression( from ) );

        /* add the entity to our list */
        adder=CONS(ENTITY,new_ent,adder);
    }
    statement_declarations(expanded)=adder;

    /* final packing */
    sequence s = make_sequence( CONS(STATEMENT,expanded,NIL) );
    instruction ins = make_instruction_sequence( s );

    if( !type_void_p(treturn) )
        *modified_expression = *entity_to_expression(returned_entity);
    return ins;

}


/* recursievly inline an expression if needed
 */
static
void inline_expression(expression expr )
{
    if( expression_call_p(expr) )
    {
        call callee = syntax_call(expression_syntax(expr));
        if( inline_should_inline( callee ) )
        {
                new_instructions = CONS(STATEMENT, instruction_to_statement( inline_expression_call( expr, callee )), new_instructions);
#if 0
            type t= functional_result(type_functional(entity_type(inlined_module)));
            if( type_void_p(t) ) /* only replace return statement by gotos */
            {
            }
            else
            {
                /* create a new sequence to receive the new instruction */
                statement new_instr =  instruction_to_statement( inline_expression_call( expr, callee )) ;
                list l = CONS(STATEMENT, instruction_to_statement(copy_instruction(current_instruction)) , NIL );
                l = CONS(STATEMENT, new_instr ,l);
                *current_instruction=*make_instruction_sequence( make_sequence( l ) );
            }
#endif
        }
     /*   else
        {
            inline_all_expressions(current_instruction, call_arguments( callee ) );
        }*/

    }
}

/* check if a call has inlinable calls
 */
static bool has_inlinable_calls;
static
void inline_has_inlinable_calls_crawler(call callee)
{
    has_inlinable_calls|=inline_should_inline(callee);
}
static
bool inline_has_inlinable_calls(call callee)
{
    has_inlinable_calls=false;
    gen_recurse(callee, call_domain, gen_true,&inline_has_inlinable_calls_crawler);
    return has_inlinable_calls;
}

/* this is in charge of replacing instruction by new ones
 * only apply if this instruction does not contain other instructions
 */ 
static 
void inline_instruction_switcher(instruction ins)
{
    block_level=0;
    switch( instruction_tag(ins) )
    {
        /* handle this with the expression handler */
        case is_instruction_call:
            {
                call callee =instruction_call(ins);
                if( inline_has_inlinable_calls( callee ) )
                {
                    *ins= *make_instruction_expression( call_to_expression(callee) );
                    inline_instruction_switcher( ins );
                }
            } break;
        /* handle those with a gen_recurse */
        case is_instruction_return:
        case is_instruction_expression:
            {
                new_instructions=NIL;
                gen_recurse(ins,expression_domain,gen_true,&inline_expression);
                if( new_instructions != NIL ) /* something happens on the way to heaven */
                {
                    type t= functional_result(type_functional(entity_type(inlined_module)));
                    if( ! type_void_p(t) )
                    {
                        new_instructions=CONS(STATEMENT,instruction_to_statement(copy_instruction(ins)),new_instructions);
                    }
                    *ins = *make_instruction_sequence( make_sequence( gen_nreverse(new_instructions) ) );
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
    modified_module = module_name_to_entity(module);
    /* get target module's ressources */
    modified_module_statement =
        (statement) db_get_memory_resource(DBR_CODE, module, TRUE);
    pips_assert("statements found", !statement_undefined_p(modified_module_statement) );

    /* inline all calls to inlined_module */
    gen_recurse(modified_module_statement, instruction_domain, gen_true, &inline_instruction_switcher);

    /* validate changes */
    DB_PUT_MEMORY_RESOURCE(DBR_CODE, strdup(module), modified_module_statement);

    modified_module_statement = statement_undefined;
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

   /* check them */
   pips_assert("is a functionnal",entity_function_p(inlined_module) || entity_subroutine_p(inlined_module) );
   pips_assert("statements found", !statement_undefined_p(inlined_module_statement) );
   debug_on("INLINING_DEBUG_LEVEL");
   debug(2, "inlining", "done for %s\n", module_name);

   /* get module's callers */
   callees callers = (callees)db_get_memory_resource(DBR_CALLERS, module_name, TRUE);
   list callers_l = callees_callees(callers);

   /* inline call in each caller */
   MAP(STRING, caller_name, {
           inline_calls( caller_name );
           }, callers_l );


   debug_off();
   /* Should have worked: */
   return TRUE;
}
