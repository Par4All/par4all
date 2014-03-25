#ifdef HAVE_CONFIG_H
    #include "pips_config.h"
#endif
#include "genC.h"
#include "constants.h"
#include "misc.h"
#include "linear.h"
#include "ri.h"
#include "effects.h"
#include "ri-util.h"
#include "effects-util.h"
#include "database.h"
#include "pipsdbm.h"
#include "properties.h"
#include "control.h"
#include "resources.h"
#include "effects-generic.h"
#include "transformations.h"
#include "accel-util.h"

#if 0
static void remove_guard(test t)
{
    statement s = (statement)gen_get_ancestor(statement_domain,t);
    list regions = load_cumulated_rw_effects_list(s);

    list read_regions = regions_read_regions(regions);
    list write_regions = regions_write_regions(regions);
    transformer tr = transformer_range(load_statement_precondition(s));
    /* skip read regions ... should be in a property */
    FOREACH(REGION,reg,write_regions)
    {
        reference r = region_any_reference(reg);
        entity e = reference_variable(r);
        /* so e is written by s, does the write exactly match
         * the dimension of e or not ?
         */
        list 
        if(region_to_minimal_dimensions(reg,tr
    }
}

bool remove_guards(const char *module_name)
{
    /* prelude */
    set_current_module_entity(module_name_to_entity( module_name ));
    set_current_module_statement((statement) db_get_memory_resource(DBR_CODE, module_name, true) );
    set_cumulated_rw_effects((statement_effects)db_get_memory_resource(DBR_REGIONS, module_name, true));
    module_to_value_mappings(get_current_module_entity());
    set_precondition_map( (statement_mapping) db_get_memory_resource(DBR_PRECONDITIONS, module_name, true) );

    /* main */
    gen_recurse(get_current_module_statement(),
            test_domain,gen_true,remove_guard);

    /* validate */
    module_reorder(get_current_module_statement());
    DB_PUT_MEMORY_RESOURCE(DBR_CODE, module_name,get_current_module_statement());

    /* postlude */
    reset_current_module_entity();
    reset_current_module_statement();
    reset_cumulated_rw_effects();
    reset_precondition_map();
    free_value_mappings();

    return true;

}
#endif

/** 
 * create a guard @a guard around statement @a s
 * 
 * @param s statement to guard
 * @param guard guard to apply
 */
static void guard_expanded_statement(statement s, expression guard)
{
    if(statement_test_p(s))
    {
        test t =statement_test(s);
        test_condition(t)=MakeBinaryCall(
                entity_intrinsic(AND_OPERATOR_NAME),
                copy_expression(guard),
                test_condition(t)
                );
    }
    else
    {
        statement false_branch = make_empty_statement();
        instruction ins = make_instruction_test(
                make_test(
                    copy_expression(guard),
                    copy_statement(s),/* update_instruction force us to copy */
                    false_branch
                    )
                );
        update_statement_instruction(s,ins);
    }
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
void do_loop_expansion(statement st, expression size,bool center,bool apply_guard)
{
    loop l =statement_loop(st);
    range r = loop_range(l);
    if(expression_constant_p(range_increment(r)))
    {
        /* this gets (e-b)/i , that is the number of iterations in the loop */
        expression nb_iter = range_to_expression(r,range_to_nbiter);
        /* the expanded nb_iter will be refered as efactor = factor* size */
        expression factor = 
                    make_op_exp(DIVIDE_OPERATOR_NAME,
                        make_op_exp(PLUS_OPERATOR_NAME,
                            copy_expression(nb_iter),
                            make_op_exp(MINUS_OPERATOR_NAME,
                                copy_expression(size),
                                int_to_expression(1)
                                )
                            ),
                        copy_expression(size)
                        );
        entity efactor = make_new_scalar_variable(
                get_current_module_entity(),
                basic_of_expression(factor)
                );
        AddEntityToCurrentModule(efactor);
        /* get_current_module_statement is inaccurate here, but if factor is store independant ... icm should work better ! st would be better */
        insert_statement(get_current_module_statement(),make_assign_statement(entity_to_expression(efactor),factor),true);

        expression expanded_nb_iter = 
            make_op_exp(MULTIPLY_OPERATOR_NAME,
                    copy_expression(size),
                    entity_to_expression(efactor)
                    );
        expression new_range_lower_value=expression_undefined;
        if(center)
          new_range_lower_value=make_op_exp(PLUS_OPERATOR_NAME,
              copy_expression(range_lower(r)),
              make_op_exp(
                DIVIDE_OPERATOR_NAME,
                make_op_exp(
                  MINUS_OPERATOR_NAME,
                  copy_expression(expanded_nb_iter),
                  copy_expression(nb_iter)
                  ),
                int_to_expression(2)
                )
              );
        else
          new_range_lower_value=copy_expression(range_lower(r));
        /* we must check for loop_index() in range_lower*/
        set ents = get_referenced_entities(new_range_lower_value);
        entity new_range_lower_value_entity = entity_undefined;
        statement assign_statement = statement_undefined;;
        if(set_belong_p(ents,loop_index(l)))
        {
            new_range_lower_value_entity=make_new_scalar_variable(get_current_module_entity(),basic_of_expression(new_range_lower_value));
            AddEntityToCurrentModule(new_range_lower_value_entity);
            assign_statement=make_assign_statement(entity_to_expression(new_range_lower_value_entity),copy_expression(new_range_lower_value));
        }
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
        if(apply_guard) {
            expression guard = MakeBinaryCall(
                    entity_intrinsic(AND_OPERATOR_NAME),
                    MakeBinaryCall(
                        entity_intrinsic(GREATER_OR_EQUAL_OPERATOR_NAME),
                        entity_to_expression(loop_index(l)),
                        entity_undefined_p(new_range_lower_value_entity)?range_lower(r):entity_to_expression(new_range_lower_value_entity)
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
        }

        /* update loop fields either by a constant, or by a new entity with appropriate init*/
        range_upper(r)=new_range_upper_value;
        range_lower(r)=new_range_lower_value;
        /* insert statement in the end, otherwise pips gets disturbed */
        if(!statement_undefined_p(assign_statement)) insert_statement(st,assign_statement,true);

    }
    else
        pips_user_warning("cannot expand a loop with non constant increment\n");

}


bool loop_expansion(const char* module_name)
{
    set_current_module_entity(module_name_to_entity(module_name));
    set_current_module_statement((statement) db_get_memory_resource(DBR_CODE, module_name, true));
    set_cumulated_rw_effects((statement_effects)db_get_memory_resource(DBR_CUMULATED_EFFECTS, get_current_module_name(), true));

    debug_on("LOOP_EXPANSION_DEBUG_LEVEL");
    /* first case: statement inserted by loop_expansion_init were illegal */
    list statements_to_clean  = find_statements_with_pragma(get_current_module_statement(),get_string_property("STATEMENT_INSERTION_FAILURE_PRAGMA"));
    list statements_to_merge  = find_statements_with_pragma(get_current_module_statement(),get_string_property("STATEMENT_INSERTION_SUCCESS_PRAGMA"));
    /* generate guard if no statement to merge or no statement to clean */
    bool apply_guard = ENDP(statements_to_merge) && ENDP(statements_to_clean);

    /* remove the test statement */
    FOREACH(STATEMENT,statement_to_clean,statements_to_clean)
    {
        update_statement_instruction(statement_to_clean,make_continue_instruction());
        free_extensions(statement_extensions(statement_to_clean));
        statement_extensions(statement_to_clean)=empty_extensions();
    }

    /* second case: statement inserted by loop_expansion_init were legal */
    /* remove the test statement and merge */
    FOREACH(STATEMENT,statement_to_clean,statements_to_merge)
    {
        update_statement_instruction(statement_to_clean,make_continue_instruction());
        free_extensions(statement_extensions(statement_to_clean));
        statement_extensions(statement_to_clean)=empty_extensions();

    }

    gen_free_list(statements_to_clean);
    gen_free_list(statements_to_merge);

    const char* lp_label=get_string_property_or_ask(
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
                const char* srate = get_string_property("LOOP_EXPANSION_SIZE");
                expression rate = string_to_expression(srate,get_current_module_entity());
                /* ok for the ui part, let's do something !*/
                do_loop_expansion(loop_statement,rate,get_bool_property("LOOP_EXPANSION_CENTER"),apply_guard);
                /* commit changes */
                module_reorder(get_current_module_statement()); ///< we may have add statements
                DB_PUT_MEMORY_RESOURCE(DBR_CODE, module_name, get_current_module_statement());

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

/* creates a new statement that perfom the expansion of the loop
 * this statement is flagged for further processing */
static
void do_loop_expansion_init(statement st, expression size)
{
    loop l =statement_loop(st);
    range r = loop_range(l);
    if(expression_constant_p(range_increment(r)))
    {
        /* compute the range of new loop
         * range lower will be next step of loop
         */

        /* this gets (e-b)/i , that is the number of iterations in the loop */
        expression nb_iter = range_to_expression(r,range_to_nbiter);
        /* this gets the expanded nb_iter */

        expression expanded_nb_iter = 
            make_op_exp(MULTIPLY_OPERATOR_NAME,
                    copy_expression(size),
                    make_op_exp(DIVIDE_OPERATOR_NAME,
                        make_op_exp(PLUS_OPERATOR_NAME,
                            copy_expression(nb_iter),
                            make_op_exp(MINUS_OPERATOR_NAME,
                                copy_expression(size),
                                int_to_expression(1)
                                )
                            ),
                        copy_expression(size)
                        )
                    );

        expression new_range_lower = make_op_exp(PLUS_OPERATOR_NAME,copy_expression(range_lower(r)),
                make_op_exp(MULTIPLY_OPERATOR_NAME,nb_iter,copy_expression(range_increment(r)))
                );
        expression new_range_upper = make_op_exp(MINUS_OPERATOR_NAME,
                make_op_exp(PLUS_OPERATOR_NAME,
                    copy_expression(range_lower(r)),
                    make_op_exp(MULTIPLY_OPERATOR_NAME,expanded_nb_iter,copy_expression(range_increment(r)))
                    ),
                int_to_expression(1)

                );

        clone_context cc = make_clone_context(get_current_module_entity(),get_current_module_entity(),NIL,get_current_module_statement());
        statement inserted_statement = 
            instruction_to_statement(
                    make_instruction_loop(
                        make_loop(
                            loop_index(l),
                            make_range(new_range_lower,new_range_upper,copy_expression(range_increment(r))),
                            clone_statement(loop_body(l),cc),
                            entity_empty_label(),
                            make_execution_sequential(),
                            NIL
                            )
                        )
                );
        free_clone_context(cc);
        add_pragma_str_to_statement(inserted_statement,(char*)get_string_property("STATEMENT_INSERTION_PRAGMA"),true);
        insert_statement(st,inserted_statement,false);
    }
    else
        pips_user_warning("cannot expand a loop with non constant increment\n");

}
/* first step of the loop expansion process:
 * create a statement to insert and flag it with a pragma
 */
bool loop_expansion_init(const char* module_name)
{
    set_current_module_entity(module_name_to_entity(module_name));
    set_current_module_statement((statement) db_get_memory_resource(DBR_CODE, module_name, true));

    debug_on("LOOP_EXPANSION_INIT_DEBUG_LEVEL");


    const char* lp_label=get_string_property_or_ask(
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
                const char* srate = get_string_property("LOOP_EXPANSION_SIZE");
                expression rate =string_to_expression(srate,get_current_module_entity());
                /* ok for the ui part, let's do something !*/
                do_loop_expansion_init(loop_statement,rate);
                /* commit changes */
                module_reorder(get_current_module_statement()); ///< we may have add statements
                DB_PUT_MEMORY_RESOURCE(DBR_CODE, module_name, get_current_module_statement());

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

