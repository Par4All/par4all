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

static void convert_min_max_to_tests_in_loop(loop l)
{
    expression ub = range_upper(loop_range(l));
    if(expression_call_p(ub))
    {
        call c = expression_call(ub);
        entity op = call_function(c);
        if(ENTITY_MIN_P(op)) {
            expression lhs = copy_expression(binary_call_lhs(c));
            expression rhs = copy_expression(binary_call_rhs(c));
            free_expression(range_upper(loop_range(l)));
            range_upper(loop_range(l))=lhs;
            loop_body(l)=instruction_to_statement(
                    make_instruction_test(
                        make_test(
                            binary_intrinsic_expression(
                                LESS_OR_EQUAL_OPERATOR_NAME,
                                entity_to_expression(loop_index(l)),
                                rhs),
                            loop_body(l),
                            make_empty_block_statement()
                            )
                        )
                    );
        }
    }
}

static void convert_min_max_to_tests(statement s)
{
    gen_recurse(s,loop_domain, gen_true, convert_min_max_to_tests_in_loop);
}

#if 0
static statement do_symbolic_tiling_rec_junction(statement root, list tests) {
    if(ENDP(tests)) return copy_statement(root);
    else return instruction_to_statement(
            make_instruction_test(
                make_test(EXPRESSION(CAR(tests)),
                    do_symbolic_tiling_rec_junction(root,CDR(tests)),
                    do_symbolic_tiling_rec_junction(root,CDR(tests))
                    )
                )
            );
}
#endif

void do_symbolic_tiling(statement base, list vector)
{
    list effects = load_cumulated_rw_effects_list(base);
    list tiled_loops_outer = NIL;
    list tiled_loops_inner = NIL;
    list prelude = NIL;
    list allmins = NIL;
    statement sloop=copy_statement(base);
    FOREACH(EXPRESSION,tile_size,vector)
    {
        loop l = statement_loop(sloop);
        list tile_expression_effects = proper_effects_of_expression(tile_size);
        /* check if tile_size is modified by sloop and generate an a temporary variable if needed */
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
        entity index = make_new_index_entity(loop_index(l),"t");
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
        allmins=CONS(EXPRESSION,binary_intrinsic_expression(GREATER_THAN_OPERATOR_NAME,
                    copy_expression(upperbound_lhs),
                    copy_expression(upperbound_rhs)
                    ),allmins);

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
        if(!ENDP(allmins) && get_bool_property("SYMBOLIC_TILING_PERFECT_TILES")) {
          expression guard = EXPRESSION(CAR(allmins));
          set s = get_referenced_entities(guard);
          if(set_belong_p(s,loop_index(statement_loop(curr)))) {
            loop_body(statement_loop(curr))=
              instruction_to_statement(
                  make_instruction_test(
                    make_test(
                      guard,
                      copy_statement(loop_body(statement_loop(curr))),
                      copy_statement(loop_body(statement_loop(curr)))
                      )
                    )
                  );
            POP(allmins);
          }
          set_free(s);
        }
    }

    /* update */
    statement_label(base)=entity_empty_label();/*the label have been duplicated by copy_statement */
    statement_instruction(base)=instruction_undefined;
    update_statement_instruction(base,make_instruction_block(gen_nreverse(CONS(STATEMENT,prev,prelude))));

    if(get_bool_property("SYMBOLIC_TILING_NO_MIN"))
        convert_min_max_to_tests(base);/* << this one is here to wait for better preconditions computations */
}


/* checks if sloop is a perfectly nested loop of depth @depth */
static bool symbolic_tiling_valid_p(statement sloop, size_t depth)
{
    int l;
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

