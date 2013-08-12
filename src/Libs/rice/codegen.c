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

#include "local.h"
#include "effects-generic.h"

/* FI: I did not know this was still used... */
extern entity current_module_entity;

/*
 this function checks if a successor su of a vertex is accessible
 through an arc whose level is less than 'level'

 dal is the arc label

 level is the minimum level
 */
static bool AK_ignore_this_level(dg_arc_label dal, int level) {
  bool true_dep = get_bool_property("RICE_DATAFLOW_DEPENDENCE_ONLY");
  FOREACH(CONFLICT, c, dg_arc_label_conflicts(dal)) {
    if(conflict_cone(c) != cone_undefined) {
      FOREACH(int, l, cone_levels(conflict_cone(c)))
      {
        if(l >= level) {
          if(true_dep) {
            action s = effect_action( conflict_source( c ));
            action k = effect_action( conflict_sink( c ));

            return (action_write_p( s ) && action_read_p( k ));
          } else {
            return (false);
          }
        }
      }
    }
  }

  return (true);
}

/*
 this function checks if a vertex v should be ignored, i.e. does not
 belong to region
 */
static bool AK_ignore_this_vertex(set region, vertex v) {
  dg_vertex_label dvl = (dg_vertex_label)vertex_vertex_label(v);
  statement st = ordering_to_statement(dg_vertex_label_statement(dvl));

  return (!set_belong_p(region, (char *)st));
}

/*
 this function checks if a successor su of a vertex should be
 ignored, i.e.  if it is linked through an arc whose level is less than
 'level' or if it does not belong to region
 */
static bool AK_ignore_this_successor(vertex __attribute__ ((unused)) v,
                                     set region,
                                     successor su,
                                     int level) {
  dg_arc_label al = (dg_arc_label)successor_arc_label(su);

  bool ignore_p = AK_ignore_this_vertex(region, successor_vertex(su));

  if(!ignore_p)
    ignore_p = AK_ignore_this_level(al, level);

  if(!ignore_p
      && get_bool_property("PARALLELIZATION_IGNORE_THREAD_SAFE_VARIABLES")) {
    list cl = dg_arc_label_conflicts(al);
    bool thread_safe_p = true;
    FOREACH(CONFLICT, c, cl) {
      effect e = conflict_source(c);
      reference r = effect_any_reference(e);
      entity v = reference_variable(r);

      if(!thread_safe_variable_p(v)) {
        thread_safe_p = false;
        break;
      }
    }
    ignore_p = thread_safe_p;
  }
  return ignore_p;
}

/* Check if a variable is private to loop nest */
static bool variable_private_to_loop_p(list /* of loop statement */loops,
                                       entity var) {
  FOREACH(STATEMENT, st, loops) {
    // We filter-out local declarations and loop locals
    list l = loop_locals(instruction_loop(statement_instruction(st)));
    list
        d =
            statement_declarations(loop_body(instruction_loop(statement_instruction(st))));

    // There is usually no declaration at this level,
    // but if we find some, warn the user
    if(statement_declarations(st) != NIL) {
      pips_user_warning("We don't expect declarations there... (sn : %d)\n",
          statement_number(st));
      print_entities(statement_declarations(st));
    }

    ifdebug(8) {
      print_statement(st);
      fprintf(stderr, "The list of privatized/private variables : \n");
      print_entities(l);
      fprintf(stderr, "\nThe list of locally declared variables : \n");
      print_entities(d);
      fprintf(stderr, "\n");
    }

    if(gen_find_eq(var, l) != entity_undefined || gen_find_eq(var, d)
        != entity_undefined) {
      return true;
    }
  }
  return false;
}

/* This function checks if conflict c between vertices v1 and v2 should
 be ignored at level l.

 A conflict is to be ignored if the variable that creates the conflict is
 local to one of the enclosing loops.

 FI: this should be extended to variables declared within the last loop
 body for C code?

 Note: The loops around every statement got by
 load_statement_loops(statement) here are just these after taking off
 the loops on which the Kennedy's algo. can't be applied. (YY)

 FI: I do not understand the note above.
 */

bool ignore_this_conflict(vertex v1, vertex v2, conflict c, int l) {
  extern int enclosing;
  effect e1 = conflict_source(c);
  reference r1 = effect_any_reference(e1);
  entity var1 = reference_variable(r1);
  statement s1 = vertex_to_statement(v1);
  list loops1 = load_statement_enclosing_loops(s1);

  effect e2 = conflict_sink(c);
  reference r2 = effect_any_reference( e2 );
  entity var2 = reference_variable(r2);
  statement s2 = vertex_to_statement(v2);
  list loops2 = load_statement_enclosing_loops(s2);
  register int i;

  if(var1 != var2) {
    /* equivalences do not deserve more cpu cycles */
    return (false);
  }
  for (i = 1; i < l - enclosing; i++) {
    if(!ENDP(loops1)) {
      loops1 = CDR(loops1);
    }
    if(!ENDP(loops2)) {
      loops2 = CDR(loops2);
    }
  }
  ifdebug(8) {
    pips_debug(8, "verifying the following conflit at level %d: \n",l);
    fprintf(stderr,
            "\t%02td --> %02td ",
            statement_number(s1),
            statement_number(s2));
    fprintf(stderr, "\t\tfrom ");
    print_effect(conflict_source(c));

    fprintf(stderr, " to ");
    print_effect(conflict_sink(c));
    fprintf(stderr, "\n");
  }

  return variable_private_to_loop_p(loops1, var1)
      || variable_private_to_loop_p(loops2, var2);
}

/* s is a strongly connected component which is analyzed at level
 l. Its vertices are enclosed in at least l loops. This gives us a
 solution to retrieve the level l loop enclosing a scc: to take its
 first vertex and retrieve the l-th loop around this vertex.
 */
statement find_level_l_loop_statement(scc s, int l) {
  vertex v = VERTEX(CAR(scc_vertices(s)));
  statement st = vertex_to_statement(v);
  list loops = load_statement_enclosing_loops(st);

  if(l > 0)
    MAPL(pl, {
          if (l-- == 1)
          return(STATEMENT(CAR(pl)));
        }, loops);
  return (statement_undefined);
}

set scc_region(scc s) {
  set region = set_make(set_pointer);
  MAPL(pv, {
        set_add_element(region, region,
            (char *) vertex_to_statement(VERTEX(CAR(pv))));
      }, scc_vertices(s));
  return (region);
}

/* s is a strongly connected component for which a DO loop is being
 produced.  this function returns false if s contains no dependences at
 level l. in this case, the loop will be a DOALL loop.
 */
bool contains_level_l_dependence(scc s, set region, int level) {
  FOREACH(VERTEX, v, scc_vertices(s)) {
    statement s1 = vertex_to_statement(v);
    FOREACH(SUCCESSOR, su, vertex_successors(v))
    {
      vertex vs = successor_vertex(su);
      statement s2 = vertex_to_statement(vs);

      if(!AK_ignore_this_vertex(region, vs)) {
        dg_arc_label dal = (dg_arc_label)successor_arc_label(su);
        FOREACH(CONFLICT, c, dg_arc_label_conflicts(dal))
        {
          if(!ignore_this_conflict(v, vs, c, level)) {
            if(conflict_cone(c) != cone_undefined) {
              FOREACH(int, l, cone_levels(conflict_cone(c)))
              {
                if(l == level) {
                  ifdebug(7) {
                    pips_debug(7, "containing conflit at level %d: ",level);
                    fprintf(stderr,
                            "\t%02td --> %02td ",
                            statement_number(s1),
                            statement_number(s2));
                    fprintf(stderr, "\t\tfrom ");
                    print_effect(conflict_source(c));
                    fprintf(stderr, " to ");
                    print_effect(conflict_sink(c));
                    fprintf(stderr, "\n");
                  }
                  return (true);
                }
              }
            }
          }
        }
      }
    }
  }

  return (false);
}

/* this function returns true if scc s is stronly connected at level l,
 i.e. dependence arcs at level l or greater form at least one cycle */

bool strongly_connected_p(scc s, int l) {
  cons *pv = scc_vertices(s);
  vertex v = VERTEX(CAR(pv));

  /* if s contains more than one vertex, it is strongly connected */
  if(CDR(pv) != NIL)
    return (true);
  /* if there is a dependence from v to v, s is strongly connected */
  FOREACH(SUCCESSOR, s, vertex_successors(v)) {
    if(!AK_ignore_this_level((dg_arc_label)successor_arc_label(s), l)
        && successor_vertex(s) == v)
      return (true);
  }

  /* s is not strongly connected */
  return (false);
}

/* this function creates a nest of parallel loops around an isolated
 statement whose iterations may execute in parallel.

 loops is the loop nest that was around body in the original program. l
 is the current level; it tells us how many loops have already been
 processed.

 */
statement MakeNestOfParallelLoops(int l,
                                  cons *loops,
                                  statement body,
                                  bool task_parallelize_p) {
  statement s;
  pips_debug(3, " at level %d ...\n",l);

  if(loops == NIL)
    s = body;
  else if(l > 0)
    s = MakeNestOfParallelLoops(l - 1, CDR(loops), body, task_parallelize_p);
  else {
    statement slo = STATEMENT(CAR(loops));
    loop lo = statement_loop(slo);
    tag seq_or_par = ((CDR(loops) == NIL || task_parallelize_p)
        && index_private_p(lo)) ? is_execution_parallel
                                : is_execution_sequential;

    /* At most one outer loop parallel */
    bool
        task_parallelize_inner =
            (seq_or_par == is_execution_parallel
                && !get_bool_property("GENERATE_NESTED_PARALLEL_LOOPS")) ? false
                                                                         : task_parallelize_p;

    s = MakeLoopAs(slo,
                   seq_or_par,
                   MakeNestOfParallelLoops(0,
                                           CDR(loops),
                                           body,
                                           task_parallelize_inner));
  }
  return (s);
}

int statement_imbrication_level(statement st) {
  list loops = load_statement_enclosing_loops(st);
  return (gen_length(loops));
}

statement MakeNestOfStatementList(int l,
                                  int nbl,
                                  list *lst,
                                  list loops,
                                  list * block,
                                  list * eblock,
                                  bool task_parallelize_p) {
  statement stat = statement_undefined;
  statement rst = statement_undefined;
  extern int enclosing;

  debug_on("RICE_DEBUG_LEVEL");

  if(*lst != NIL && nbl) {
    if(gen_length(*lst) == 1)
      rst = (STATEMENT(CAR(*lst)));
    else
      rst = make_block_statement(*lst);
    if(nbl >= l - 1)
      stat = MakeNestOfParallelLoops(l - 1 - enclosing,
                                     loops,
                                     rst,
                                     task_parallelize_p);
    else
      stat = rst;
    *lst = NIL;
    INSERT_AT_END(*block, *eblock, CONS(STATEMENT, stat, NIL));
  }

  debug_off();
  return (stat);
}

/* This function implements Allen & Kennedy's algorithm.
 *
 * BB (Bruno Baron): task_parallelize_p is true when we want to
 * parallelize the loop, false when we only want to vectorize
 * it. Probably called by "rice_cray", but there is no explicit
 * information about the vectorization facility in PIPS.
 *
 * This function is also used to perform loop invariant code motion
 * (Julien Zory).
 */
statement CodeGenerate(statement __attribute__ ((unused)) stat,
                       graph g,
                       set region,
                       int l,
                       bool task_parallelize_p) {
  list lst = NIL;
  cons *lsccs;
  // cons *ps; unused, but still present in commented out code
  list loops = NIL;

  cons *block = NIL, *eblock = NIL;
  statement stata = statement_undefined;
  statement rst = statement_undefined;
  int nbl = 0;

  debug_on("RICE_DEBUG_LEVEL");

  pips_debug(9, "Begin: starting at level %d ...\n", l);
  ifdebug(9)
    print_statement_set(stderr, region);

  pips_debug(9, "finding and top-sorting sccs ...\n");
  set_sccs_drivers(&AK_ignore_this_vertex, &AK_ignore_this_successor);
  lsccs = FindAndTopSortSccs(g, region, l);
  reset_sccs_drivers();

  pips_debug(9, "generating code ...\n");

  FOREACH(scc,s,lsccs) {
    stata = statement_undefined;
    if(strongly_connected_p(s, l))
      stata = ConnectedStatements(g, s, l, task_parallelize_p);
    else {
      if(!get_bool_property("PARTIAL_DISTRIBUTION"))
        /* if s contains a single vertex and if this vertex is not
         dependent upon itself, we generate a doall loop for it,
         unless it is a continue statement. */
        stata = IsolatedStatement(s, l, task_parallelize_p);
      else {
        /* statements that are independent are gathered
         into the same doall loop */
        stata = IsolatedStatement(s, l, task_parallelize_p);

        /* set inner_region = scc_region(s);
         if (contains_level_l_dependence(s,inner_region,l)) {
         stat = IsolatedStatement(s, l, task_parallelize_p);
         debug(9, "CodeGenerate",
         "isolated comp.that contains dep. at Level %d\n",
         l);
         }
         else  {
         vertex v = VERTEX(CAR(scc_vertices(s)));
         statement st = vertex_to_statement(v);
         instruction sbody = statement_instruction(st);
         nbl = statement_imbrication_level(st);
         if (instruction_call_p(sbody)
         && !instruction_continue_p(sbody))
         if (nbl>=l-1)
         stat=IsolatedStatement(s, l, task_parallelize_p);
         else {
         loops = load_statement_enclosing_loops(st);
         lst = gen_nconc(lst, CONS(STATEMENT, st, NIL));
         }
         }
         */
      }
    }

    /* In order to preserve the dependences, statements that have
     been collected should be generated before the isolated statement
     that has just been detected */

    if(stata != statement_undefined) {
      ifdebug(9) {
        pips_debug(9, "generated statement:\n");
        print_parallel_statement(stata);
      }
      (void)MakeNestOfStatementList(l,
                                    nbl,
                                    &lst,
                                    loops,
                                    &block,
                                    &eblock,
                                    task_parallelize_p);
      INSERT_AT_END(block, eblock, CONS(STATEMENT, stata, NIL));
    }
  }
  gen_free_list(lsccs);

  (void)MakeNestOfStatementList(l,
                                nbl,
                                &lst,
                                loops,
                                &block,
                                &eblock,
                                task_parallelize_p);

  switch(gen_length(block)) {
    case 0:
      rst = statement_undefined;
      break;
    default:
      rst = make_block_statement(block);
  }

  ifdebug(8) {
    pips_debug(8, "Result:\n");

    if(rst == statement_undefined)
      pips_debug(8, "No code to generate\n");
    else
      print_parallel_statement(rst);

  }
  debug_off();
  return (rst);
}

/* This function creates a new loop whose characteristics (index,
 * bounds, ...) are similar to those of old_loop. The body and the
 * execution type are different between the old and the new loop.
 *
 * fixed bug about private variable without effects, FC 22/09/93
 */
statement MakeLoopAs(statement old_loop_statement,
                     tag seq_or_par,
                     statement body) {
  loop old_loop = statement_loop(old_loop_statement);
  loop new_loop;
  statement new_loop_s;
  statement old_body = loop_body (old_loop);
  list new_locals = gen_copy_seq(loop_locals(old_loop));

  if(rice_distribute_only)
    seq_or_par = is_execution_sequential;

  // copy declaration from old body to new body
  if((statement_decls_text (old_body) != string_undefined)
      && (statement_decls_text (old_body) != NULL)) {
    if(!statement_block_p(body))
      body = make_block_statement(CONS(STATEMENT,body,NIL));
    statement_decls_text (body) = copy_string (statement_decls_text (old_body));
  }
  if((statement_declarations (old_body) != list_undefined)
      && (statement_declarations (old_body) != NULL)) {
    if(!statement_block_p(body))
      body = make_block_statement(CONS(STATEMENT,body,NIL));
    statement_declarations (body)
        = gen_copy_seq(statement_declarations (old_body));
  }

  new_loop = make_loop(loop_index(old_loop), copy_range(loop_range(old_loop)), /* avoid sharing */
  body, entity_empty_label(), make_execution(seq_or_par, UU), new_locals);

  new_loop_s
      = make_statement(entity_empty_label(),
                       statement_number(old_loop_statement),
                       STATEMENT_ORDERING_UNDEFINED,
                       string_undefined,
                       make_instruction(is_instruction_loop, new_loop),
                       NIL,
                       NULL,
                       copy_extensions(statement_extensions(old_loop_statement)), make_synchronization_none());

  ifdebug(8) {
    pips_assert("Execution is either parallel or sequential",
        seq_or_par==is_execution_sequential || seq_or_par==is_execution_parallel);
    pips_debug(8, "New %s loop\n",
        seq_or_par==is_execution_sequential? "sequential" : "parallel");
    print_parallel_statement(new_loop_s);
  }

  return (new_loop_s);
}

/* If the isolated statement is a CALL and is not a CONTINUE,
 regenerate the nested loops around it. Otherwise, returns an
 undefined statement. */
statement IsolatedStatement(scc s, int l, bool task_parallelize_p) {
  vertex v = VERTEX(CAR(scc_vertices(s)));
  statement st = vertex_to_statement(v);
  statement rst = statement_undefined;
  list loops = load_statement_enclosing_loops(st);
  instruction sbody = statement_instruction(st);
  extern int enclosing;

  pips_debug(8, "Input statement %" PRIdPTR "\n", statement_number(st));

  /* continue statements are ignored. */
  /*FI: But they should not be isolated statements if the contain
   declarations... */
  //if(declaration_statement_p(st))
  //pips_internal_error("Declaration statement is junked.");

  if(!instruction_call_p(sbody) || (continue_statement_p(st)
      && !declaration_statement_p(st)))
    ;
  /* FI: we are likely to end up in trouble here because C allows
   expressions as instructions... */
  /* FI: if the statement is any kind of loop or a test, do not go
   down.*/
  else
    rst = MakeNestOfParallelLoops(l - 1 - enclosing,
                                  loops,
                                  st,
                                  task_parallelize_p);

  ifdebug(8) {
    pips_debug(8, "Returned statement:\n");
    safe_print_statement(rst);
  }

  return (rst);
}

/* BB: ConnectedStatements() is called when s contains more than one
 vertex or one vertex dependent upon itself. Thus, vectorization can't
 occur.

 FI: it may not be true if one of the statements is a C declaration.
 */
statement ConnectedStatements(graph g, scc s, int l, bool task_parallelize_p) {
  extern int enclosing;
  statement slo = find_level_l_loop_statement(s, l - enclosing);
  loop lo = statement_loop(slo);
  statement inner_stat;
  set inner_region;
  tag seq_or_par;
  bool task_parallelize_inner;

  pips_debug(8, "at level %d:\n",l);
  ifdebug(8)
    PrintScc(s);

  inner_region = scc_region(s);
  seq_or_par = (!task_parallelize_p
      || contains_level_l_dependence(s, inner_region, l)
      || !index_private_p(lo)) ? is_execution_sequential
                               : is_execution_parallel;

  /* At most one outer loop parallel */
  task_parallelize_inner
      = (seq_or_par == is_execution_parallel
          && !get_bool_property("GENERATE_NESTED_PARALLEL_LOOPS")) ? false
                                                                   : task_parallelize_p;

  /* CodeGenerate does not use the first parameter... */
  inner_stat = CodeGenerate(/* big hack */statement_undefined,
                            g,
                            inner_region,
                            l + 1,
                            task_parallelize_inner);

  set_free(inner_region);

  if(statement_undefined_p(inner_stat))
    return inner_stat;
  else
    return MakeLoopAs(slo, seq_or_par, inner_stat);
}
