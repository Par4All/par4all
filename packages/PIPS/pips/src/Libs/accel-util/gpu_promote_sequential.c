/*
 Copyright 2011 MINES ParisTech
 Copyright 2011 HPC Project

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

/**
 * @file gpu_promote_sequential.c
 * Avoid useless data transfert by promoting sequential code on GPU
 * @author Mehdi Amini <mehdi.amini@hpc-project.com>
 */
#ifdef HAVE_CONFIG_H
#include "pips_config.h"
#endif
#include <ctype.h>

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "kernel_memory_mapping.h"
#include "effects.h"
#include "ri-util.h"
#include "semantics.h"
#include "effects-util.h"
#include "text.h"
#include "pipsdbm.h"
#include "resources.h"
#include "properties.h"
#include "misc.h"
#include "control.h"
#include "callgraph.h"
#include "effects-generic.h"
#include "effects-simple.h"
#include "effects-convex.h"
#include "text-util.h"
#include "parser_private.h"
#include "preprocessor.h"
#include "transformer.h"
#include "accel-util.h"


static statement promote_statement(list promoted_stmts, loop l) {
  promoted_stmts = gen_nreverse(promoted_stmts);
  entity loop_idx = make_new_scalar_variable(get_current_module_entity(),
                                             make_basic(is_basic_int, (void *) 4));
  statement promoted = make_new_loop_statement(loop_idx,
                                               int_to_expression(0),
                                               int_to_expression(0),
                                               int_to_expression(1),
                                               make_block_statement(promoted_stmts),
                                               make_execution_parallel());
  if(!loop_undefined_p(l)) {
    loop_locals(l) = CONS(entity,loop_idx,loop_locals(l));
  }
  return promoted;
}

/**
 * Operate on a sequence and promote all sequential code in a trivial parallel
 * loop. The heuristic that trigger the promotion is quite simple at that time,
 * all sequential code is promoted if a parallel loop is present in the sequence
 *
 * @param seq is the sequence on which to operate
 * @loop l is an optionnal param that give the enclosing loop
 */
void gpu_promote_sequential_on_sequence(sequence seq, loop l) {
  list stmts = sequence_statements(seq);

  // Try to find a parallel loop in the sequence
  bool found_parallel_loop_p = false;
  FOREACH(statement,s,stmts) {
    if(parallel_loop_statement_p(s)) {
      found_parallel_loop_p = true;
      break;
    }
  }

  if(found_parallel_loop_p) {
    list new_stmts = NIL;
    list promoted_stmts = NIL;
    FOREACH(statement,s,stmts) {
      if(!parallel_loop_statement_p(s)) {
        if(!empty_statement_or_continue_p(s)) {
          ifdebug(2) {
            pips_debug(2,"Promote statement :");
            print_statement(s);
          }

          promoted_stmts = CONS(statement,s,promoted_stmts);
        }
      } else if(promoted_stmts) {
        promoted_stmts = gen_nreverse(promoted_stmts);
        statement promoted = promote_statement(promoted_stmts, l);
        new_stmts = CONS(statement,promoted,CONS(statement,s,new_stmts));
        promoted_stmts = NIL;

        ifdebug(2) {
          pips_debug(2,"Promote statements in loop :");
          print_statement(promoted);
        }
      }
    }
    if(promoted_stmts) {
      promoted_stmts = gen_nreverse(promoted_stmts);
      statement promoted = promote_statement(promoted_stmts, l);
      new_stmts = CONS(statement,promoted,new_stmts);
    }
    sequence_statements(seq) = gen_nreverse(new_stmts); // FIXME Free old sequence
  }
}

static bool gpu_promote_sequential_walker_in(loop l) {
  if(loop_sequential_p(l)) {
    statement body = loop_body(l);
    if(statement_sequence_p(body)) {
      gpu_promote_sequential_on_sequence(statement_sequence(body),l);
    }
  }
  return true;
}

void gpu_promote_sequential_on_statement(statement s) {
  gen_recurse(s,loop_domain,gpu_promote_sequential_walker_in,gen_null);
}

bool gpu_promote_sequential(const char* module_name) {
  statement module_stat = (statement)db_get_memory_resource(DBR_CODE,
                                                            module_name,
                                                            true);
  set_current_module_statement(module_stat);
  set_current_module_entity(local_name_to_top_level_entity(module_name));

  debug_on("GPU_PROMOTE_SEQUENTIAL_DEBUG_LEVEL");

  /* Initialize set for each statement */
  gpu_promote_sequential_on_statement(module_stat);

  debug_off();

  module_reorder(module_stat);

  DB_PUT_MEMORY_RESOURCE(DBR_CODE,
      module_name,
      module_stat);


  reset_current_module_statement();
  reset_current_module_entity();

  return true;



}
