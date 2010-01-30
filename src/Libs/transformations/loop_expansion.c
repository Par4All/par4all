#include "genC.h"
#include "constants.h"
#include "misc.h"
#include "linear.h"
#include "ri.h"
#include "ri-util.h"
#include "database.h"
#include "pipsdbm.h"
#include "properties.h"
#include "resources.h"
#include "transformations.h"

static
void do_loop_expansion(statement st, int size)
{
    loop l =statement_loop(st);
    range r = loop_range(l);
    if(expression_constant_p(range_increment(r)))
    {
        /* this gets (e-b)/i , that is the number of iterations in the loop */
        expression nb_iter = range_to_expression(r,range_to_nbiter);
        /* this gets the remainder of nb_iter / size */
        expression remainder = make_op_exp(MODULO_OPERATOR_NAME, nb_iter, int_to_expression(size));
        /* this checks wether the remainder is 0 or not, that is do we need to expand the loop or not ? */
        expression not_expand_p = MakeBinaryCall(entity_intrinsic(EQUAL_OPERATOR_NAME),remainder,int_to_expression(0));
        /* this will hold the new value of range_upper */
        entity new_range_upper = make_new_scalar_variable(get_current_module_entity(),basic_of_expression(range_upper(r)));
        AddEntityToCurrentModule(new_range_upper);

        /* this makes the conditionnal assignment */
        statement assign_expand = make_assign_statement(
                entity_to_expression(new_range_upper),
                make_op_exp(PLUS_OPERATOR_NAME,
                    copy_expression(range_upper(r)),
                    make_op_exp(MINUS_OPERATOR_NAME,
                        int_to_expression(size),
                        copy_expression(remainder)))
                );
        statement assign_not_expand = make_assign_statement(
                entity_to_expression(new_range_upper),
                copy_expression(range_upper(r)));

        statement conditionnal_assignement = instruction_to_statement(
                make_instruction_test(
                    make_test(
                        not_expand_p,
                        assign_not_expand,
                        assign_expand)
                    )
                );
        statement_comments(conditionnal_assignement)=strdup("// PIPS test to adjust loop size to required loop expansion parameter");

        /* update fileds */
        loop_body(l)=instruction_to_statement(
                make_instruction_test(
                    make_test(
                        MakeBinaryCall(
                            entity_intrinsic(LESS_OR_EQUAL_OPERATOR_NAME),
                            entity_to_expression(loop_index(l)),
                            range_upper(r)
                            ),
                        loop_body(l),
                        make_empty_statement()
                        )
                    )
                );
        range_upper(r)=entity_to_expression(new_range_upper);
        insert_statement(st,conditionnal_assignement,true);
    }
    else
        pips_user_warning("cannot expand a loop with non constant increment\n");

}


bool loop_expansion(const string module_name)
{
    set_current_module_entity(module_name_to_entity(module_name));
    set_current_module_statement((statement) db_get_memory_resource(DBR_CODE, module_name, TRUE));
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
                    do_loop_expansion(loop_statement,rate);
                    /* commit changes */
                    module_reorder(get_current_module_statement()); ///< we may have had statements
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
    return true;;

}
