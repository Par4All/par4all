#ifdef HAVE_CONFIG_H
    #include "pips_config.h"
#endif
#include "genC.h"
#include "constants.h"
#include "misc.h"
#include "linear.h"
#include "ri.h"
#include "ri-util.h"
#include "database.h"
#include "pipsdbm.h"
#include "properties.h"
#include "control.h"
#include "resources.h"
#include "transformations.h"
#include "effects-generic.h"

/** 
 * create a guard @a guard around statement @a s
 * 
 * @param s statement to guard
 * @param guard guard to apply
 */
static void guard_expanded_statement(statement s, expression guard)
{
    instruction ins = make_instruction_test(
            make_test(
                copy_expression(guard),
                copy_statement(s),/* update_instruction force us to copy */
                make_empty_statement()
                )
            );
    update_statement_instruction(s,ins);
}

/** 
 * create a guard @a guard around statement @a s if needed
 * it is needed when a not-private variable (that is not in @a parent_loop locals)
 * is written by the statement
 * @param s statement to guard
 * @param guard guard to apply
 * @param parent_loop set of enclosing loop locals
 */
static
void guard_expanded_statement_if_needed(statement s,expression guard, loop parent_loop)
{
    /* map guard on each block */
    if(statement_block_p(s))
    {
        FOREACH(STATEMENT,st,statement_block(s))
            guard_expanded_statement_if_needed(st,guard,parent_loop);
    }
    /* map on loop body if range independant from loop_index(parent_loop) */
    else if(statement_loop_p(s))
    {
        loop l =statement_loop(s);
        set s0 = get_referenced_entities(loop_range(l));
        if(set_belong_p(s0,loop_index(parent_loop)))
            guard_expanded_statement(s,guard);
        else
            guard_expanded_statement_if_needed(loop_body(l),guard,parent_loop);
        set_free(s0);
    }
    else {
        list effects = load_cumulated_rw_effects_list(s);
        list decls = statement_to_declarations(s);
        set sdecls = set_make(set_pointer);set_assign_list(sdecls,decls); gen_free_list(decls);
        list privates = loop_locals(parent_loop);
        set sprivates = set_make(set_pointer); set_assign_list(sprivates,privates);

        bool must_guard= false;
        FOREACH(EFFECT,eff,effects)
        {
            if((must_guard=(effect_write_p(eff) && 
                            !set_belong_p(sprivates,effect_entity(eff)) &&
                            !set_belong_p(sdecls,effect_entity(eff)) )))
                break;
            /* an io effect implies a guard too */
            if((must_guard=io_effect_p(eff)))
                break;
        }
        set_free(sdecls);set_free(sprivates);
        if(must_guard) guard_expanded_statement(s,guard);
    }
}

static
void do_loop_expansion(statement st, int size,int offset)
{
    loop l =statement_loop(st);
    range r = loop_range(l);
    if(expression_constant_p(range_increment(r)))
    {
        /* this gets (e-b)/i , that is the number of iterations in the loop */
        expression nb_iter = range_to_expression(r,range_to_nbiter);

        /* this gets the expanded nb_iter */
        expression expanded_nb_iter = 
            make_op_exp(MULTIPLY_OPERATOR_NAME,
                    int_to_expression(size),
                    make_op_exp(DIVIDE_OPERATOR_NAME,
                        make_op_exp(PLUS_OPERATOR_NAME,
                            nb_iter,
                            make_op_exp(MINUS_OPERATOR_NAME,
                                int_to_expression(size),
                                int_to_expression(1)
                                )
                            ),
                        int_to_expression(size)
                        )
                   );
        /* we must check for loop_index() in range_lower and atomize it if found */
        set ents = get_referenced_entities(range_lower(r));
        expression former_range_lower = range_lower(r);
        entity new_range_lower_value_entity = entity_undefined;
        if(set_belong_p(ents,loop_index(l)))
        {
            new_range_lower_value_entity=make_new_scalar_variable(get_current_module_entity(),basic_of_expression(range_lower(r)));
            AddEntityToCurrentModule(new_range_lower_value_entity);
            range_lower(r)=entity_to_expression(new_range_lower_value_entity);
        }
        /* then compute expanded range values */
        expression new_range_lower_value=
            make_op_exp(MINUS_OPERATOR_NAME,range_lower(r),int_to_expression(offset));
        set_free(ents);
        expression new_range_upper_value=
            make_op_exp(MINUS_OPERATOR_NAME,
                    make_op_exp(PLUS_OPERATOR_NAME,
                        entity_undefined_p(new_range_lower_value_entity)?copy_expression(new_range_lower_value):entity_to_expression(new_range_lower_value_entity),
                        make_op_exp(MULTIPLY_OPERATOR_NAME,expanded_nb_iter,copy_expression(range_increment(r)))
                        ),
                    int_to_expression(1)
                    );

        /* set the guard on all statement that need it */
        expression guard = 
            MakeBinaryCall(
                    entity_intrinsic(AND_OPERATOR_NAME),
                    MakeBinaryCall(
                        entity_intrinsic(GREATER_OR_EQUAL_OPERATOR_NAME),
                        entity_to_expression(loop_index(l)),
                        range_lower(r)
                        )
                    ,
                    MakeBinaryCall(
                        entity_intrinsic(LESS_OR_EQUAL_OPERATOR_NAME),
                        entity_to_expression(loop_index(l)),
                        range_upper(r)
                        )
                    );

        guard_expanded_statement_if_needed(loop_body(l),guard,l);
        free_expression(guard);

        /* update loop fields and add initialization statement if needed */
        range_upper(r)=new_range_upper_value;
        range_lower(r)=new_range_lower_value;
        if(!entity_undefined_p(new_range_lower_value_entity))
            insert_statement(st,make_assign_statement(entity_to_expression(new_range_lower_value_entity),copy_expression(former_range_lower)),true);
    }
    else
        pips_user_warning("cannot expand a loop with non constant increment\n");

}


bool loop_expansion(const string module_name)
{
    set_current_module_entity(module_name_to_entity(module_name));
    set_current_module_statement((statement) db_get_memory_resource(DBR_CODE, module_name, TRUE));
    set_cumulated_rw_effects((statement_effects)db_get_memory_resource(DBR_CUMULATED_EFFECTS, get_current_module_name(), TRUE));

    debug_on("LOOP_EXPANSION_DEBUG_LEVEL");


    string lp_label=get_string_property_or_ask(
            "LOOP_LABEL",
            "Which loop do you want to expand?\n(give its label):"
    );

    if( !empty_string_p(lp_label) )
    {
        entity lb_entity = find_label_entity(module_name,lp_label);
        if( !entity_undefined_p(lb_entity) )
        {
            statement loop_statement = find_loop_from_label(get_current_module_statement(),lb_entity);
            if(!statement_undefined_p(loop_statement))
            {
                int rate = get_int_property("LOOP_EXPANSION_SIZE");
                if( rate > 0)
                {
                    /* ok for the ui part, let's do something !*/
                    do_loop_expansion(loop_statement,rate,get_int_property("LOOP_EXPANSION_OFFSET"));
                    /* commit changes */
                    module_reorder(get_current_module_statement()); ///< we may have add statements
                    DB_PUT_MEMORY_RESOURCE(DBR_CODE, module_name, get_current_module_statement());

                }
                else pips_user_error("Please provide a positive loop expansion size\n");
            }
            else pips_user_error("label '%s' is not put on a loop\n",lp_label);


        }
        else pips_user_error("loop label `%s' does not exist\n", lp_label);
    }
    else pips_user_error("transformation cancelled \n", lp_label);

    debug_off();
    reset_current_module_entity();
    reset_current_module_statement();
    reset_cumulated_rw_effects();
    return true;;

}
