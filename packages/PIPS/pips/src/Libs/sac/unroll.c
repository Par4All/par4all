/*

  $Id$

  Copyright 1989-2014 MINES ParisTech

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
#ifdef HAVE_CONFIG_H
    #include "pips_config.h"
#endif

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "effects.h"

#include "resources.h"

#include "misc.h"
#include "ri-util.h"
#include "effects-util.h"
#include "pipsdbm.h"

#include "effects-generic.h"
#include "transformations.h"

#include "control.h"


#include "sac.h"
#include "hyperplane.h"

#include "properties.h"

#include <limits.h>

static bool should_unroll_p(instruction i)
{
    switch(instruction_tag(i))
    {
        case is_instruction_call:
            return true;

        case is_instruction_sequence:
            {
                cons * j;

                for( j = sequence_statements(instruction_sequence(i));
                        j != NIL;
                        j = CDR(j) )
                {
                    statement s = STATEMENT(CAR(j));
                    if (!should_unroll_p(statement_instruction(s)))
                        return false;
                }
                return true;
            }

        case is_instruction_test:
        case is_instruction_loop:
        case is_instruction_whileloop:
        case is_instruction_goto:
        case is_instruction_unstructured:
        default:
            return false;
    }
}

typedef struct {
    int min;
    int max;
} MinMaxVar;

static void compute_variable_size(statement s, MinMaxVar* varwidth)
{
    int width = effective_variables_width(statement_instruction(s));

    if (width > varwidth->max)
        varwidth->max = width;

    if (width < varwidth->min)
        varwidth->min = width;
}

static void simd_loop_unroll(statement loop_statement, intptr_t rate)
{
  range r = loop_range(statement_loop(loop_statement));
  expression erange = range_to_expression(r,range_to_nbiter);
  intptr_t irange;
  if(expression_integer_value(erange,&irange) && irange <=rate) {
      bool saved[] = {
          get_bool_property("LOOP_NORMALIZE_ONE_INCREMENT"),
          get_bool_property("LOOP_NORMALIZE_SKIP_INDEX_SIDE_EFFECT")
      };
      set_bool_property("LOOP_NORMALIZE_ONE_INCREMENT",true);
      set_bool_property("LOOP_NORMALIZE_SKIP_INDEX_SIDE_EFFECT",true);
      loop_normalize_statement(loop_statement);
      range r = loop_range(statement_loop(loop_statement));
      simplify_expression(&range_upper(r));
      simplify_expression(&range_lower(r));
      set_bool_property("LOOP_NORMALIZE_ONE_INCREMENT",saved[0]);
      set_bool_property("LOOP_NORMALIZE_SKIP_INDEX_SIDE_EFFECT",saved[1]);
      full_loop_unroll(loop_statement);
  }
  else
  {  
	  expression lower = range_lower(r);
	  NORMALIZE_EXPRESSION(lower);
	  entity new = entity_undefined;
	  if (normalized_complex_p(expression_normalized(lower))){
			new = make_new_index_entity(loop_index(statement_loop(loop_statement)),"i");
			AddEntityToCurrentModule(new);
			range_lower(r) = entity_to_expression(new);
	  }	  
	  do_loop_unroll(loop_statement,rate,NULL);
	  if (! entity_undefined_p(new)){
			insert_statement(loop_statement,
					make_assign_statement(entity_to_expression(new),lower),
					true);
	  } 
  }
  free_expression(erange);
}

static int simple_simd_unroll_rate(loop l) {
    /* Compute variable size */
    MinMaxVar varwidths = { INT_MAX , 0 };
    int varwidth;
    gen_context_recurse(loop_body(l), &varwidths, statement_domain, gen_true, 
            compute_variable_size);

    /* Decide between min and max unroll factor */
    if (get_bool_property("SIMDIZER_AUTO_UNROLL_MINIMIZE_UNROLL"))
        varwidth = varwidths.max;
    else
        varwidth = varwidths.min;

    /* Round up varwidth to a power of 2 */
    int regWidth = get_int_property("SAC_SIMD_REGISTER_WIDTH");

    if ((varwidth > regWidth/2) || (varwidth <= 0)) 
        return 1;

    for(int j = 8; j <= regWidth/2; j*=2)
    {
        if (varwidth <= j)
        {
            varwidth = j; 
            break;
        }
    }

    /* Unroll as many times as needed by the variables width */
    return regWidth / varwidth;
}

static bool simple_simd_unroll_loop_filter(statement s)
{
    instruction i;
    loop l;
    instruction iBody;

    /* If this is not a loop, keep on recursing */
    i = statement_instruction(s);
    if (!instruction_loop_p(i))
        return true;
    l = instruction_loop(i);

    /* Can only simdize certain loops */
    iBody = statement_instruction(loop_body(l));
    if (!should_unroll_p(iBody))
        return true;  /* can't do anything */

    /* Unroll as many times as needed by the variables width */
    simd_loop_unroll(s, simple_simd_unroll_rate(l) );

    /* Do not recursively analyse the loop */
    return false;
}

static void compute_parallelism_factor(statement s, MinMaxVar* factor)
{
    int varwidth = effective_variables_width(statement_instruction(s));

    /* see if the statement can be SIMDized */
    FOREACH(MATCH, m,match_statement(s))
    {
        /* and if so, to what extent it may benefit from unrolling */
        FOREACH(OPCODE, o,opcodeClass_opcodes(match_type(m)))
        {
            if (get_subwordSize_from_opcode(o, 0) >= varwidth) //opcode may be used
            {
                if (opcode_vectorSize(o) > factor->max)
                    factor->max = opcode_vectorSize(o);
                if (opcode_vectorSize(o) < factor->min)
                    factor->min = opcode_vectorSize(o);
            }
        }
    }
}

static bool full_simd_unroll_loop_filter(statement s)
{
    MinMaxVar factor;
    instruction i;
    loop l;
    instruction iBody;

    /* If this is not a loop, keep on recursing */
    i = statement_instruction(s);
    if (!instruction_loop_p(i))
        return true;
    l = instruction_loop(i);

    /* Can only simdize certain loops */
    iBody = statement_instruction(loop_body(l));
    if (!should_unroll_p(iBody))
        return true;  /* can't do anything */

    /* look at each of the statements in the body */
    factor.min = INT_MAX;
    factor.max = 1;
    gen_context_recurse(loop_body(l), &factor, statement_domain, gen_true, compute_parallelism_factor);
    factor.min = factor.min > factor.max ? factor.max : factor.min;


    /* Decide between min and max unroll factor, and unroll */
    int unroll_rate = get_bool_property("SIMDIZER_AUTO_UNROLL_MINIMIZE_UNROLL") ? factor.min : factor.max;
    simd_loop_unroll(s, unroll_rate);

    /* Do not recursively analyse the loop */
    return false;
}

static void simd_unroll_as_needed(statement module_stmt)
{
    /* Choose algorithm to use, and use it */
    if (get_bool_property("SIMDIZER_AUTO_UNROLL_SIMPLE_CALCULATION"))
    {
        gen_recurse(module_stmt, statement_domain, 
                simple_simd_unroll_loop_filter, gen_null);
    }
    else
    {
        set_simd_treematch((matchTree)db_get_memory_resource(DBR_SIMD_TREEMATCH,"",true));
        set_simd_operator_mappings(db_get_memory_resource(DBR_SIMD_OPERATOR_MAPPINGS,"",true));

        gen_recurse(module_stmt, statement_domain, 
                full_simd_unroll_loop_filter, gen_null);
        reset_simd_treematch();
        reset_simd_operator_mappings();
    }
}
bool loop_auto_unroll(const char* mod_name) {
    // get the resources
    statement mod_stmt = (statement)
        db_get_memory_resource(DBR_CODE, mod_name, true);

    set_current_module_statement(mod_stmt);
    set_current_module_entity(module_name_to_entity(mod_name));

    debug_on("SIMDIZER_DEBUG_LEVEL");

    /* do the job */
    const char* slabel = get_string_property_or_ask("LOOP_LABEL","enter the label of a loop !");
    entity elabel = find_label_entity(mod_name,slabel);

    if(entity_undefined_p(elabel)) {
        pips_user_error("label %s does not exist !\n", slabel);
    }
    else {
        statement theloopstatement = find_loop_from_label(get_current_module_statement(),elabel);
        if(!statement_undefined_p(theloopstatement)) {
            simple_simd_unroll_loop_filter(theloopstatement);
        }
    }

    pips_assert("Statement is consistent after SIMDIZER_AUTO_UNROLL", 
            statement_consistent_p(mod_stmt));

    // Reorder the module, because new statements have been added
    module_reorder(mod_stmt);
    DB_PUT_MEMORY_RESOURCE(DBR_CODE, mod_name, mod_stmt);

    // update/release resources
    reset_current_module_statement();
    reset_current_module_entity();

    debug_off();

    return true;
}

bool simdizer_auto_unroll(char * mod_name)
{
    // get the resources
    statement mod_stmt = (statement)
        db_get_memory_resource(DBR_CODE, mod_name, true);

    set_current_module_statement(mod_stmt);
    set_current_module_entity(module_name_to_entity(mod_name));

    debug_on("SIMDIZER_DEBUG_LEVEL");

    simd_unroll_as_needed(mod_stmt);

    pips_assert("Statement is consistent after SIMDIZER_AUTO_UNROLL", 
            statement_consistent_p(mod_stmt));

    // Reorder the module, because new statements have been added
    module_reorder(mod_stmt);
    DB_PUT_MEMORY_RESOURCE(DBR_CODE, mod_name, mod_stmt);

    // update/release resources
    reset_current_module_statement();
    reset_current_module_entity();

    debug_off();

    return true;
}

static void gather_local_indices(reference r, set s) {
    list indices = reference_indices(r);
    if(!ENDP(indices)) {
        expression x = EXPRESSION(CAR(gen_last(indices)));
        set xs = get_referenced_entities(x);
        set_union(s,s,xs);
        set_free(xs);
    }
}

static void keep_loop_indices(statement s, list *L) {
	if(statement_loop_p(s))
		*L=CONS(STATEMENT,s,*L);
}

static list do_simdizer_auto_tile_int_to_list(int maxdepth, int path,loop l) {
    list out = NIL;
    int rw = simple_simd_unroll_rate(l);
    while(maxdepth--) {
        int a = 1;
        if(path & 1) a=rw;
        out=CONS(EXPRESSION,int_to_expression(a),out);
    }
    return out;
}

static statement do_simdizer_auto_tile_generate_all_tests(statement root, int maxdepth, int path, expression * tests) {
    if(expression_undefined_p(*tests)) {
        clone_context cc = make_clone_context(get_current_module_entity(),get_current_module_entity(),NIL,get_current_module_statement());
        statement cp = clone_statement(root, cc);
        /* duplicate effects for the copied statement */
        store_cumulated_rw_effects(cp,copy_effects(load_cumulated_rw_effects(root)));
        list l = do_simdizer_auto_tile_int_to_list(maxdepth, path,statement_loop(root));
        do_symbolic_tiling(cp,l);
        statement siter = cp;
        FOREACH(LOOP,li,l)
            siter = loop_body(statement_loop(siter));
        free_clone_context(cc);
        return cp;
    }
    else {
        int npath = path << 1 ;
        return do_simdizer_auto_tile_generate_all_tests(root, maxdepth, npath+1, tests+1);
        /*
        int npath = path << 1 ;
        statement trueb = do_simdizer_auto_tile_generate_all_tests(root, maxdepth, npath+1, tests+1);
        statement falseb = do_simdizer_auto_tile_generate_all_tests(root, maxdepth, npath,  tests+1);
        return
        instruction_to_statement(
                make_instruction_test(
                    make_test(
                        *tests,
                        trueb,
                        falseb
                        )
                    )
                );
                */
    }
}

static statement simdizer_auto_tile_generate_all_tests(statement root, int maxdepth, expression tests[1+maxdepth]) {
    return do_simdizer_auto_tile_generate_all_tests(root,maxdepth,0,&tests[0]);
}

bool simdizer_auto_tile(const char * module_name) {
    bool success = false;
    set_current_module_entity(module_name_to_entity( module_name ));
    set_current_module_statement((statement) db_get_memory_resource(DBR_CODE, module_name, true) );
    set_cumulated_rw_effects((statement_effects)db_get_memory_resource(DBR_CUMULATED_EFFECTS, module_name, true));

    /* do the job */
    const char* slabel = get_string_property_or_ask("LOOP_LABEL","enter the label of a loop !");
    entity elabel = find_label_entity(module_name,slabel);

    if(entity_undefined_p(elabel)) {
        pips_user_error("label %s does not exist !\n", slabel);
    }
    else {
        statement theloopstatement = find_loop_from_label(get_current_module_statement(),elabel);
        if(!statement_undefined_p(theloopstatement)) {
            if(perfectly_nested_loop_p(theloopstatement)) {
                /* retrieve loop indices that are referenced as last element */
                set indices = set_make(set_pointer);
                gen_context_recurse(theloopstatement,indices,reference_domain,gen_true,gather_local_indices);
                list tloops = NIL;
                gen_context_recurse(theloopstatement,&tloops, statement_domain,gen_true,keep_loop_indices);
                list allloops =gen_copy_seq(tloops);
                FOREACH(STATEMENT,l,allloops) {
                    if(!set_belong_p(indices,loop_index(statement_loop(l))))
                        gen_remove_once(&tloops,l);
                }
				theloopstatement=STATEMENT(CAR((tloops)));
				//allloops=gen_nreverse(allloops);
				while(theloopstatement!=STATEMENT(CAR(allloops))) POP(allloops);
                set_free(indices);
                set loops = set_make(set_pointer);set_assign_list(loops,tloops);gen_free_list(tloops);

                /* build tests */
                int max_unroll_rate = simple_simd_unroll_rate(statement_loop(theloopstatement));
                int nloops=gen_length(allloops);
                /* iterate over all possible combination of tests */
                expression alltests[1+nloops];
                alltests[nloops] = expression_undefined ;
                int j=0;
                FOREACH(STATEMENT,sl,allloops) {
                    alltests[j++]=MakeBinaryCall(
                            entity_intrinsic(GREATER_THAN_OPERATOR_NAME),
                            range_to_expression(loop_range(statement_loop(sl)),range_to_nbiter),
                            int_to_expression(2*max_unroll_rate)
                            );
                }
                /* create the if's recursively */
                statement root =
                    simdizer_auto_tile_generate_all_tests(
                        theloopstatement,
                        nloops,
                        alltests);
                gen_recurse(root, statement_domain, gen_true, statement_remove_useless_label);
                *theloopstatement=*root;
                loop_label(statement_loop(theloopstatement))=elabel;
                success=true;

                /* validate */
                module_reorder(get_current_module_statement());
                DB_PUT_MEMORY_RESOURCE(DBR_CODE, module_name,get_current_module_statement());
            }
            else pips_user_error("loop is not perfectly nested !\n");
        }
        else pips_user_error("label is not on a loop!\n");
    }

    reset_cumulated_rw_effects();
    reset_current_module_statement();
    reset_current_module_entity();
    return success;
}




