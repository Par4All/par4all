/* symbolic tiling
 * less general, but works for symbolics bounds
 */
#ifdef HAVE_CONFIG_H
    #include "pips_config.h"
#endif

#include "genC.h"
#include "linear.h"
#include "ri.h"

#include "ri-util.h"
#include "misc.h"
#include "pipsdbm.h"
#include "properties.h"
#include "control.h"

#include "effects-generic.h"
#include "effects-simple.h"
#include "effects-util.h"

#include "transformations.h"

bool fix_loop_index_sign(loop l) {
    set ent = get_referenced_entities(loop_range(l));
    set_add_element(ent,ent,loop_index(l));
    SET_FOREACH(entity,e,ent) {
        if(local_entity_of_module_p(e,get_current_module_entity())) {
            if(entity_formal_p(e)) {
                type t = ultimate_type(entity_type(e));
                if(unsigned_type_p(t)) {
                    entity newe=make_new_scalar_variable_with_prefix(entity_user_name(e),
                            get_current_module_entity(),make_basic_int(DEFAULT_INTEGER_TYPE_SIZE));
                    AddEntityToCurrentModule(newe);
                    free_value(entity_initial(newe));
                    entity_initial(newe)=make_value_expression(entity_to_expression(e));
                    replace_entity(l,e,newe);
                    pips_assert("type consistent",type_consistent_p(entity_type(e)));
                }
            }
            else {
                type t = ultimate_type(entity_type(e));
                if(unsigned_type_p(t)) {
                    free_type(entity_type(e));
                    entity_type(e)=
                        make_type_variable(
                                make_variable(
                                    make_basic_int(DEFAULT_INTEGER_TYPE_SIZE),
                                    NIL,
                                    NIL
                                    )
                                );
                    pips_assert("type consistent",type_consistent_p(entity_type(e)));
                }
            }
        }
        pips_assert("entity is fine",entity_consistent_p(e));
    }
    set_free(ent);
    return true;
}

void do_symbolic_tiling(statement base, list vector)
{
    list effects = load_cumulated_rw_effects_list(base);
    list tiled_loops_outer = NIL;
    list tiled_loops_inner = NIL;
    list prelude = NIL;
    statement sloop=copy_statement(base);
    FOREACH(EXPRESSION,tile_size,vector)
    {
        loop l = statement_loop(sloop);
        list tile_expression_effects = proper_effects_of_expression(tile_size);
        /* check if tile_size is modified by sloop and generate a temporary variable if needed */
        /* we should also check tile_size has no write effect
         * but currently, string_to_expression asserts this */
        FOREACH(EFFECT,teff,tile_expression_effects) {
            FOREACH(EFFECT,eff,effects) {
                if(effect_write_p(eff) && 
                        references_may_conflict_p(effect_any_reference(teff),effect_any_reference(eff)))
                {
                    entity etemp = make_new_scalar_variable(get_current_module_entity(),make_basic_int(DEFAULT_INTEGER_TYPE_SIZE));
                    AddEntityToCurrentModule(etemp);
                    statement ass = make_assign_statement(entity_to_expression(etemp),copy_expression(tile_size));
                    update_expression_syntax(tile_size,make_syntax_reference(make_reference(etemp,NIL)));
                    prelude=CONS(STATEMENT,ass,prelude);
                    goto generate_tile;
                }
            }
        }
generate_tile:;
        /* outer loop new index */
        entity index = make_new_scalar_variable(get_current_module_entity(),make_basic_int(DEFAULT_INTEGER_TYPE_SIZE));
        AddEntityToCurrentModule(index);
        expression lower_bound = 
            binary_intrinsic_expression(MULTIPLY_OPERATOR_NAME,
                    entity_to_expression(index),
                    copy_expression(tile_size)
                    );
        expression upperbound_lhs = copy_expression(range_upper(loop_range(l)));
        expression upperbound_rhs = 
            binary_intrinsic_expression(PLUS_OPERATOR_NAME,
                    copy_expression(lower_bound),
                    binary_intrinsic_expression(MINUS_OPERATOR_NAME,
                        copy_expression(tile_size),
                        int_to_expression(1)
                        )
                    );

        expression upper_bound = 
            binary_intrinsic_expression(MIN_OPERATOR_NAME,
                    upperbound_lhs,
                    upperbound_rhs
                    );

        /* inner loop */
        statement inner = instruction_to_statement(
                make_instruction_loop(
                    make_loop(
                        loop_index(l),
                        make_range(
                            lower_bound,
                            upper_bound,
                            int_to_expression(1)
                            ),
                        statement_undefined,
                        entity_empty_label(),
                        make_execution_parallel(),
                        NIL
                        )
                    )
                );
        /* outer loop */
        statement outer = sloop;
        loop_index(l)=index;
        //range_increment(loop_range(l))=copy_expression(tile_size);
        /* will help partial_eval */
        range_upper(loop_range(l))=
            binary_intrinsic_expression(DIVIDE_OPERATOR_NAME,
                    range_upper(loop_range(l)),
                    copy_expression(tile_size)
                    );
        /* save */
        tiled_loops_outer=CONS(STATEMENT,outer,tiled_loops_outer);
        tiled_loops_inner=CONS(STATEMENT,inner,tiled_loops_inner);

        /* go on for the next one */
        sloop=loop_body(l);
    }
    /* once we are done, regenerate the whole thing */

    /* prepare chain all */
    tiled_loops_inner=gen_append(tiled_loops_inner,tiled_loops_outer);
    statement last = STATEMENT(CAR(tiled_loops_inner));
    statement prev = last;
    POP(tiled_loops_inner);
    /* set tail */
    loop_body(statement_loop(last))=sloop;
    /* chain all */
    FOREACH(STATEMENT,curr,tiled_loops_inner) {
        loop_body(statement_loop(curr))=prev;
        prev=curr;
    }

    /* update */
    statement_label(base)=entity_empty_label();/*the label have been duplicated by copy_statement */
    statement_instruction(base)=instruction_undefined;
    update_statement_instruction(base,make_instruction_block(gen_nreverse(CONS(STATEMENT,prev,prelude))));
    clean_up_sequences(base);

    /* fix signed / unsigned types for further processing otherwise we could end with integer overflow */
    gen_recurse(base,loop_domain,gen_true, fix_loop_index_sign);
}


/* checks if sloop is a perfectly nested loop of depth @depth */
static bool symbolic_tiling_valid_p(statement sloop, size_t depth)
{
    intptr_t l;
    if(depth == 0 ) return true;
    else {
        if(statement_loop_p(sloop) && 
                ( execution_parallel_p(loop_execution(statement_loop(sloop))) || get_bool_property("SYMBOLIC_TILING_FORCE") ) &&
                ( expression_integer_value(range_increment(loop_range(statement_loop(sloop))),&l) && l == 1 )
          ){
            statement body = loop_body(statement_loop(sloop));
            return symbolic_tiling_valid_p(body,depth-1);
        }
        else return false;
    }
}


bool symbolic_tiling(const char *module_name)
{
    /* prelude */
    bool result = false;
    set_current_module_entity(module_name_to_entity( module_name ));
    set_current_module_statement((statement) db_get_memory_resource(DBR_CODE, module_name, true) );
    set_cumulated_rw_effects((statement_effects)db_get_memory_resource(DBR_CUMULATED_EFFECTS, module_name, true));

    // sometimes empty continues are put here and there and disturb me
    clean_up_sequences(get_current_module_statement());

    entity elabel = find_label_entity(
            get_current_module_name(),
            get_string_property("LOOP_LABEL")
            );
    statement sloop = find_loop_from_label(get_current_module_statement(),elabel);
    if( !statement_undefined_p(sloop) )
    {
        list vector = string_to_expressions(
                get_string_property("SYMBOLIC_TILING_VECTOR"),",",
                get_current_module_entity());
        if(ENDP(vector))
            pips_user_warning("must provide a non empty array with expressions valid in current context\n");
        else if((result=symbolic_tiling_valid_p(sloop,gen_length(vector)))) {
            do_symbolic_tiling(sloop,vector);
        }
        gen_full_free_list(vector);
    }
    else pips_user_warning("must provide a valid loop label\n");

    /* postlude */
    module_reorder(get_current_module_statement());
    DB_PUT_MEMORY_RESOURCE(DBR_CODE, module_name,get_current_module_statement());

    reset_current_module_entity();
    reset_current_module_statement();
    reset_cumulated_rw_effects();

    return result;
}

