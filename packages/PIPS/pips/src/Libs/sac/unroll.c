/*

  $Id$

  Copyright 1989-2010 MINES ParisTech

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

#include "resources.h"

#include "misc.h"
#include "ri-util.h"
#include "pipsdbm.h"

#include "semantics.h"
#include "effects-generic.h"
#include "transformations.h"

#include "control.h"

#include "dg.h"

typedef dg_arc_label arc_label;
typedef dg_vertex_label vertex_label;

#include "graph.h"

#include "sac.h"

#include "properties.h"

#include <limits.h>

static bool should_unroll_p(instruction i)
{
    switch(instruction_tag(i))
    {
        case is_instruction_call:
            return TRUE;

        case is_instruction_sequence:
            {
                cons * j;

                for( j = sequence_statements(instruction_sequence(i));
                        j != NIL;
                        j = CDR(j) )
                {
                    statement s = STATEMENT(CAR(j));
                    if (!should_unroll_p(statement_instruction(s)))
                        return FALSE;
                }
                return TRUE;
            }

        case is_instruction_test:
        case is_instruction_loop:
        case is_instruction_whileloop:
        case is_instruction_goto:
        case is_instruction_unstructured:
        default:
            return FALSE;
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

static bool simple_simd_unroll_loop_filter(statement s)
{
    MinMaxVar varwidths;
    int varwidth;
    instruction i;
    loop l;
    instruction iBody;
    int regWidth;
    int j;

    /* If this is not a loop, keep on recursing */
    i = statement_instruction(s);
    if (!instruction_loop_p(i))
        return TRUE;
    l = instruction_loop(i);

    /* Can only simdize certain loops */
    iBody = statement_instruction(loop_body(l));
    if (!should_unroll_p(iBody))
        return TRUE;  /* can't do anything */

    /* Compute variable size */
    varwidths.min = INT_MAX;
    varwidths.max = 0;
    gen_context_recurse(loop_body(l), &varwidths, statement_domain, gen_true, 
            compute_variable_size);

    /* Decide between min and max unroll factor */
    if (get_bool_property("SIMDIZER_AUTO_UNROLL_MINIMIZE_UNROLL"))
        varwidth = varwidths.max;
    else
        varwidth = varwidths.min;

    /* Round up varwidth to a power of 2 */
    regWidth = get_int_property("SAC_SIMD_REGISTER_WIDTH");

    if ((varwidth > regWidth/2) || (varwidth <= 0)) 
        return FALSE;

    for(j = 8; j <= regWidth/2; j*=2)
    {
        if (varwidth <= j)
        {
            varwidth = j; 
            break;
        }
    }

    /* Unroll as many times as needed by the variables width */
    simd_loop_unroll(s, regWidth / varwidth);

    /* Do not recursively analyse the loop */
    return FALSE;
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
        return TRUE;
    l = instruction_loop(i);

    /* Can only simdize certain loops */
    iBody = statement_instruction(loop_body(l));
    if (!should_unroll_p(iBody))
        return TRUE;  /* can't do anything */

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

void simd_unroll_as_needed(statement module_stmt)
{
    /* Choose algorithm to use, and use it */
    if (get_bool_property("SIMDIZER_AUTO_UNROLL_SIMPLE_CALCULATION"))
    {
        gen_recurse(module_stmt, statement_domain, 
                simple_simd_unroll_loop_filter, gen_null);
    }
    else
    {
        set_simd_treematch((matchTree)db_get_memory_resource(DBR_SIMD_TREEMATCH,"",TRUE));
        set_simd_operator_mappings(db_get_memory_resource(DBR_SIMD_OPERATOR_MAPPINGS,"",TRUE));

        gen_recurse(module_stmt, statement_domain, 
                full_simd_unroll_loop_filter, gen_null);
        reset_simd_treematch();
        reset_simd_operator_mappings();
    }
}

bool simdizer_auto_unroll(char * mod_name)
{
    // get the resources
    statement mod_stmt = (statement)
        db_get_memory_resource(DBR_CODE, mod_name, TRUE);

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

    return TRUE;
}


static void addSimdCommentToStat(statement s, int num)
{

    char comment[sizeof("c " SIMD_COMMENT) + 8*sizeof(int) ];

    sprintf(comment, "c " SIMD_COMMENT "%d\n", num);

    insert_comments_to_statement(s, comment);
}

static void addSimdCommentToStats(statement s)
{
    if(instruction_sequence_p(statement_instruction(s)))
    {
        sequence seq = instruction_sequence(statement_instruction(s));
        int num = 0;
        FOREACH(STATEMENT, curStat, sequence_statements(seq) )
        {
            addSimdCommentToStat(curStat, num++);
        }
    }
    else
    {
        addSimdCommentToStat(s, 0);
    }
}

void simd_loop_unroll(statement loop_statement, int rate)
{
    do_loop_unroll(loop_statement,rate,addSimdCommentToStats);
}
