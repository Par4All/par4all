/* semantical analysis
  *
  * Accurate propagation of pre- and postconditions in unstructured
  *
  * Francois Irigoin, October 2000
  * $Id$
  *
  * $Log: unstructured.c,v $
  * Revision 1.1  2000/10/25 06:55:03  irigoin
  * Initial revision
  *
  *
  */

#include <stdio.h>
#include <string.h>
/* #include <stdlib.h> */

#include "genC.h"
#include "database.h"
#include "linear.h"
#include "ri.h"
#include "text.h"
#include "text-util.h"
#include "ri-util.h"
#include "constants.h"
#include "control.h"
#include "effects-generic.h"
#include "effects-simple.h"

#include "misc.h"

#include "properties.h"

#include "vecteur.h"
#include "contrainte.h"
#include "ray_dte.h"
#include "sommet.h"
#include "sg.h"
#include "polyedre.h"

#include "transformer.h"

#include "semantics.h"

/* These control nodes are successors of the entry node */
static list to_be_processed = list_undefined;

static bool to_be_processed_p(control c)
{
  return gen_find_eq(c, to_be_processed)!=gen_chunk_undefined;
}

/* These control nodes have a postcondition. They may not be successors of
   the entry node because some predecessors may not be reachable and still
   be processed. */
static list already_processed = list_undefined;

static bool already_processed_p(control c)
{
  return gen_find_eq(c, already_processed)!=gen_chunk_undefined;
}

/* These control nodes are successors of the entry node but they have not
   yet received a postcondition*/
static list still_to_be_processed = list_undefined;

static bool still_to_be_processed_p(control c)
{
  return gen_find_eq(c, still_to_be_processed)!=gen_chunk_undefined;
}

static void print_control_nodes(list l)
{
  MAP(CONTROL, c, {
    fprintf(stderr, "%x, %s", (unsigned int) c,
	    statement_identification(control_statement(c)));
  }, l);
  fprintf(stderr, "\n");
}

/* Node postconditions cannot be recomputed several time because
   preconditions are stored. Since they have to be used several time in an
   unknown order, it is necessary to store them. */
static statement_mapping statement_to_control_postcondition_map = NULL;

static transformer load_control_postcondition(statement stat)
{
  transformer post = (transformer)
    hash_get((hash_table) (statement_to_control_postcondition_map),
	     (char *) (stat));

  if(post == (transformer) HASH_UNDEFINED_VALUE)
    post = transformer_undefined;

  return post;
}

static void store_control_postcondition(statement stat, transformer post)
{
  pips_debug(8, "Store postcondition for statement %s\n",
	     statement_identification(stat));
  ifdebug(8) {
    print_transformer(post);
  }

  pips_assert("The postcondition is not defined yet", 
	      hash_get((hash_table) statement_to_control_postcondition_map, (char *) stat)
	      == HASH_UNDEFINED_VALUE);
  hash_put((hash_table) (statement_to_control_postcondition_map),
	   (char *)stat, (char *) post);
}

static void update_control_postcondition(statement stat, transformer post)
{
  pips_debug(8, "Update postcondition for statement %s\n",
	     statement_identification(stat));
  ifdebug(8) {
    print_transformer(post);
  }

  pips_assert("The postcondition is already defined", 
	      hash_get((hash_table) statement_to_control_postcondition_map, (char *) stat)
	      != HASH_UNDEFINED_VALUE);
  hash_update((hash_table) (statement_to_control_postcondition_map),
	      (char *)stat, (char *) post);
}

static void free_control_postcondition()
{
  /* all postconditions must be freed */
  HASH_MAP(k,v,{
    free_transformer((transformer) v);
  }, statement_to_control_postcondition_map);
  /* as well as the map */
  hash_table_free(statement_to_control_postcondition_map);
}

/* a control can be considered already processed if it has been processed
   already or if it can receive a trivial empty postcondition */
static bool considered_already_processed_p(control c)
{
  bool already = already_processed_p(c);

  if(!already) {
    if(!to_be_processed_p(c)) {
      /* unreachable node with empty precondition */
      transformer pre = transformer_empty();
      statement stmt = control_statement(c);
      transformer post = statement_to_postcondition(pre, stmt);
      store_control_postcondition(stmt, post);
      already = TRUE;
    }
  }

  return already;
}

static bool nodes_considered_already_processed_p(list l)
{
  bool ready = TRUE;

  MAP(CONTROL, c, {
    if(considered_already_processed_p(c)) {
      ready &= TRUE;
    }
    else {
      ready &= FALSE;
    }
  }, l);

  return ready;
}

/* a control is ready to be processed if all its predecessors have known
   postconditions or can receive a trivial empty postcondition */
static bool ready_to_be_processed_p(control c)
{
  bool ready = TRUE;
  MAP(CONTROL, pred, {
    if(already_processed_p(pred)) {
      /* useless, except for debugging */
      ready &= TRUE;
    }
    else if(!to_be_processed_p(pred)) {
      /* postcondition must be empty because pred is not reachable */
      transformer pre = transformer_empty();
      statement stmt = control_statement(pred);
      transformer post = statement_to_postcondition(pre, stmt);
      store_control_postcondition(stmt, post);
    }
    else if(still_to_be_processed_p(pred)) {
      ready &= FALSE;
    }
    else {
      pips_error("ready_to_be_processed_p", "node does not belong to any category\n");
    }
  }, control_predecessors(c));
  return ready;
}

static bool nodes_ready_to_be_processed_p(list l)
{
  bool ready = TRUE;

  MAP(CONTROL, c, {
    if(ready_to_be_processed_p(c)) {
      ready &= TRUE;
    }
    else {
      ready &= FALSE;
    }
  }, l);

  return ready;
}

/* A new postcondition may have to be allocated because a unique control
   may have to successors and hence two preconditions. To keep memory
   management tractable, a new postcondition always is allocated. */
static transformer load_predecessor_postcondition_or_test_condition(control pred, control c, bool postcondition_p)
{
  statement stmt = control_statement(pred);
  transformer post = postcondition_p? 
    transformer_dup(load_control_postcondition(stmt))
    :transformer_identity();

  pips_assert("The predecessor precondition is defined",
	      !transformer_undefined_p(post));
  pips_assert("The predecessor has no more than two successors",
	      gen_length(control_successors(pred))<=2);
  /* Let's hope that || is evaluated left to right... */
  pips_assert("The predecessor has c as successor",
	      c==CONTROL(CAR(control_successors(pred)))
	      ||c==CONTROL(CAR(CDR(control_successors(pred)))) );

  if(c==CONTROL(CAR(control_successors(pred)))) {
    if(ENDP(CDR(control_successors(pred)))) {
      /* stmt is not a test, post is OK */
      ;
    }
    else if(c==CONTROL(CAR(control_successors(pred)))
	    && c==CONTROL(CAR(CDR(control_successors(pred))))) {
      /* Sometimes, the two successors are really only one. To simplify a
	 useless convex hull, do not add condition information. */
      ;
    }
    else {
      /* True branch of a test */
      expression e = test_condition(statement_test(stmt));
      post = precondition_add_condition_information(post, e, TRUE);
    }
  }
  else {
    /* Must be the false branch of a test */
    expression e = test_condition(statement_test(stmt));
    post = precondition_add_condition_information(post, e, FALSE);
  }

  return post;
}

static transformer load_predecessor_postcondition(control pred, control c)
{
  transformer post = load_predecessor_postcondition_or_test_condition(pred, c, TRUE);
  return post;
}

static transformer load_predecessor_test_condition(control pred, control c)
{
  transformer post = load_predecessor_postcondition_or_test_condition(pred, c, FALSE);
  return post;
}

static void process_ready_node(control c, transformer pre_entry, unstructured u)
{
  transformer post = transformer_undefined;
  statement stmt = control_statement(c);
  list preds = control_predecessors(c);
  transformer pre = transformer_undefined;

  pips_debug(5, "Process node %s\n", statement_identification(control_statement(c))); 

  if(c==unstructured_control(u)) {
    /* Do not forget the unstructured precondition for the entry node */
    /* FI: I do not know why it has to be replicated. Probably because the
       statement containing the unstructured and the statement of the
       entry node may share the same precondition. */
    pre = copy_transformer(pre_entry);
  }
  else {
    pre = load_predecessor_postcondition(CONTROL(CAR(preds)), c);
    POP(preds);
  }

  MAP(CONTROL, pred, {
    transformer npre = load_predecessor_postcondition(pred, c);
    transformer lpre = pre;

    pre = transformer_convex_hull(npre, lpre);
    /* memory leak with lpre. pre_entry and postconditions cannot be
       freed: lpre cannot be freed during the first iteration */
    if(pred!=CONTROL(CAR(preds))) free_transformer(lpre);
    lpre = pre;
  }, preds);

  post = statement_to_postcondition(pre, stmt);
  store_control_postcondition(stmt, post);
}

static bool process_unreachable_node(control c)
{
  statement s = control_statement(c);

  if(transformer_undefined_p(load_control_postcondition(s))) {
    transformer pre = transformer_empty();
    statement stmt = control_statement(c);
    transformer post = statement_to_postcondition(pre, stmt);

    store_control_postcondition(s, post);
  }

  return TRUE;
}

static list find_scc_in_cfg()
{
  list to_be_searched = gen_copy_seq(still_to_be_processed);
  list scc = list_undefined;
  list l = list_undefined;

  pips_assert("List still_to_be_processed is not empty",
	      !ENDP(still_to_be_processed));

  for(l=to_be_searched; !ENDP(l); ) {
    control head = CONTROL(CAR(l));
    list succs = NIL;
    list preds = NIL;
    list scc_preds = list_undefined;

    forward_control_map_get_blocs(head, &succs);
    backward_control_map_get_blocs(head, &preds);
    gen_list_and(&succs, preds);
    scc = succs;
    gen_list_and_not(&preds, scc);
    scc_preds = preds;

    if(nodes_considered_already_processed_p(scc_preds)) {
      /* exit loop */
      l = NIL;
    }
    else {
      /* scc should be removed from to_be_searched to gain time! */
      gen_free_list(scc);
      POP(l);
    }
    gen_free_list(scc_preds);
  }

  gen_free_list(to_be_searched);
  return scc;
}

/* It is assumed that the scc is ready to be processed. */
static list find_scc_entry_nodes(list scc, unstructured u)
{
  control first = unstructured_control(u);
  list entry_nodes = (first==gen_find_eq(first, scc))?
    CONS(CONTROL, u, NIL):NIL;

  ifdebug(5) {
    pips_debug(5, "Begin for scc with %d nodes:\n", gen_length(scc));
    print_control_nodes(scc);
  }

  MAP(CONTROL, p, {
    pips_assert("An already-processed node cannot be in a ready-to-be-processed scc",
		gen_find_eq(p, scc)==gen_chunk_undefined);
    MAP(CONTROL, c, {
      if(c==gen_find_eq(c, scc)
	 && gen_find_eq(c, entry_nodes)==gen_chunk_undefined) {
	entry_nodes = CONS(CONTROL, c, entry_nodes);
      }
    }, control_successors(p));
  }, already_processed);


  ifdebug(5) {
    pips_debug(5, "End with %d entry nodes:\n", gen_length(entry_nodes));
    print_control_nodes(entry_nodes);
  }

  return entry_nodes;
}
static void process_cycle_in_scc(list scc, list cycle, transformer pre_entry,
				 unstructured u)
{
  transformer tfc = transformer_undefined;
  transformer tfp = transformer_identity();
  /* transformer post = transformer_undefined; */
  /* transformer pre = transformer_undefined; */
  transformer fwd_pre = transformer_undefined;
  transformer back_pre = transformer_undefined;
  transformer generic_pre = transformer_undefined;
  transformer real_pre = transformer_undefined;

  pips_debug(5, "Begin for cycle:\n");
  ifdebug(5) print_control_nodes(cycle);

  /* compute transformer along path, as in a sequence but with test
     conditions added*/
  MAPL(cc, {
    control c = CONTROL(CAR(cc));
    statement s = control_statement(c);
    transformer tfs = load_statement_transformer(s);

    tfp = transformer_combine(tfp, tfs);

    /* If c has two different successors, add condition information */
    if(gen_length(control_successors(c))==2
       && CONTROL(CAR(CDR(control_successors(c))))
       !=CONTROL(CAR(control_successors(c)))) {
      control cnext = control_undefined;
      expression e = test_condition(statement_test(s));

      if(!ENDP(CDR(cc))) {
	cnext = CONTROL(CAR(CDR(cc)));
      }
      else {
	cnext = CONTROL(CAR(cycle));
      }

      if(cnext==CONTROL(CAR(control_successors(c)))) {
	tfp = transformer_add_condition_information(tfp, e, TRUE);
      }
      else {
	tfp = transformer_add_condition_information(tfp, e, FALSE);
      }
    }
  }, cycle);

  ifdebug(5) {
    pips_debug(5, "Path transformer:\n");
    print_transformer(tfp);
  }

  /* Compute fix-point */

  tfc = (* transformer_fix_point_operator)(tfp);

  ifdebug(5) {
    pips_debug(5, "Cycle transformer:\n");
    print_transformer(tfc);
  }

  /* compute entry precondition with forward edges, backward edges and
     possibly the unstructured entry point */
  if(CONTROL(CAR(cycle))==unstructured_control(u)) {
    fwd_pre = copy_transformer(pre_entry);
  }
  else {
    fwd_pre = transformer_empty();
  }

  ifdebug(5) {
    pips_debug(5, "Precondition fwd_pre after unstructured entry node processing:\n");
    print_transformer(fwd_pre);
  }

  /* process forward edges first */
  MAP(CONTROL, p,{
    transformer new_pre = transformer_undefined;
    transformer prep = transformer_undefined;

    if(p!=gen_find_eq(p, scc)) {
      /* entry edge: postcondition of predecessor is available */
      prep = load_predecessor_postcondition(p, CONTROL(CAR(cycle)));
      new_pre = transformer_convex_hull(fwd_pre, prep);

      free_transformer(fwd_pre);
      free_transformer(prep);
      fwd_pre = new_pre;

      ifdebug(5) {
	pips_debug(5, "Precondition fwd_pre using predecessor %s:\n",
		   statement_identification(control_statement(p)));
	print_transformer(fwd_pre);
      }

    }
  }, control_predecessors(CONTROL(CAR(cycle))));

  ifdebug(5) {
    pips_debug(5, "Precondition fwd_pre:\n");
    print_transformer(fwd_pre);
  }

  /* Apply simple fix-point to approximate the entry precondition for this
     cycle. Although generic_pre is OK because the fixppoint operator is f^*,
     let's try to refine the precondition by running the cycle once to compute
     f^+ and by adding f^0 */
  generic_pre = transformer_apply(tfc, fwd_pre);
  back_pre = transformer_apply(tfp, generic_pre);
  real_pre = transformer_convex_hull(fwd_pre, back_pre);

  ifdebug(5) {
    pips_debug(5, "Precondition generic_pre:\n");
    print_transformer(generic_pre);
    pips_debug(5, "Precondition back_pre:\n");
    print_transformer(back_pre);
    pips_debug(5, "Precondition real_pre:\n");
    print_transformer(real_pre);
  }

  free_transformer(generic_pre);
  free_transformer(back_pre);
  free_transformer(fwd_pre);

  /* propagate the real_pre precondition down the cycle */
  MAPL(cc, {
    control c = CONTROL(CAR(cc));
    statement s = control_statement(c);
    transformer tfs = load_statement_transformer(s);
    transformer previous_pre = transformer_undefined;
    transformer composite_pre = transformer_undefined;

    real_pre = transformer_apply(tfs, real_pre);

    /* If c has two different successors, add condition information */
    if(gen_length(control_successors(c))==2
       && CONTROL(CAR(CDR(control_successors(c))))
       !=CONTROL(CAR(control_successors(c)))) {
      control cnext = control_undefined;
      expression e = test_condition(statement_test(s));

      if(!ENDP(CDR(cc))) {
	cnext = CONTROL(CAR(CDR(cc)));
      }
      else {
	cnext = CONTROL(CAR(cycle));
      }

      if(cnext==CONTROL(CAR(control_successors(c)))) {
	real_pre = precondition_add_condition_information(real_pre, e, TRUE);
      }
      else {
	real_pre = precondition_add_condition_information(real_pre, e, FALSE);
      }
    }

    /* convex hull with previous postcondition and storage */
    previous_pre = load_control_postcondition(s);
    composite_pre = transformer_convex_hull(previous_pre, real_pre);
    update_control_postcondition(s, composite_pre);

    ifdebug(5) {
      pips_debug(5, "Precondition previous_pre for statement %s:\n",
		 statement_identification(s));
      print_transformer(previous_pre);
      pips_debug(5, "New precondition composite_pre for statement %s:\n",
		 statement_identification(s));
      print_transformer(composite_pre);
    }

    free_transformer(real_pre);
    free_transformer(previous_pre);
  }, cycle);

  pips_debug(5, "End\n");
}

static void add_control_to_cycle(list scc, list cycle, control succ,
				 transformer pre_entry, unstructured u)
{
  static void build_control_cycles_in_scc(list, list, transformer,
					  unstructured);
  control e = CONTROL(CAR(cycle));

  ifdebug(5) {
    pips_debug(5, "Begin with cycle:\n");
    print_control_nodes(cycle);
  }

  if(succ==e) {
    process_cycle_in_scc(scc, cycle, pre_entry, u);
  }
  else {
    if(succ==gen_find_eq(succ, cycle)) {
      /* we have found a cycle in a cycle :-( */
      pips_user_error("Have you performed a full control restructuration?\n");
      pips_error("add_control_to_cycle", "Inner cycle: not implemented yet!");
    }
    else {
      cycle = gen_append(cycle, CONS(CONTROL, succ, NIL));
      build_control_cycles_in_scc(scc, cycle, pre_entry, u);
      /* restore the current cycle value */
      gen_remove(&cycle, succ);
    }
  }

  ifdebug(5) {
    pips_debug(5, "End with same cycle:\n");
    print_control_nodes(cycle);
  }
}

/* try to complete cycle and process it when it is completed */
static void build_control_cycles_in_scc(list scc, list cycle, transformer pre_entry,
					 unstructured u)
{
  control l = CONTROL(CAR(gen_last(cycle)));
  /* control e = CONTROL(CAR(cycle)); */

  ifdebug(5) {
    pips_debug(5, "Begin with cycle:\n");
    print_control_nodes(cycle);
  }

  pips_assert("The cycle is not empty", !ENDP(cycle));

  if(gen_length(control_successors(l))==1) {
    control succ = CONTROL(CAR(control_successors(l)));

    pips_assert("A unique successor must be in scc", succ==gen_find_eq(succ, scc));
    add_control_to_cycle(scc, cycle, succ, pre_entry, u);
  }
  else if(gen_length(control_successors(l))==2) {
    control succ_t = CONTROL(CAR(control_successors(l)));
    control succ_f = CONTROL(CAR(CDR(control_successors(l))));

    if(succ_t==succ_f) {
      /* they must be in the scc */
      pips_assert("A double unique successor must be in scc",
		  succ_t==gen_find_eq(succ_t, scc));
      add_control_to_cycle(scc, cycle, succ_t, pre_entry, u);
    }
    else{
      pips_assert("At least one successor must be in scc",
		  gen_find_eq(succ_t, scc) || gen_find_eq(succ_f, scc));
      if(succ_t==gen_find_eq(succ_t, scc)) {
	add_control_to_cycle(scc, cycle, succ_t, pre_entry, u);
      }
      if(succ_f==gen_find_eq(succ_f, scc)) {
	add_control_to_cycle(scc, cycle, succ_f, pre_entry, u);
      }
    }
  }
  else {
    pips_error("build_control_cycles_in_scc",
	       "A control node in a scc must have 1 or 2 successors: %d successors\n",
	       gen_length(control_successors(l)));
  }

  ifdebug(5) {
    pips_debug(5, "End with same cycle:\n");
    print_control_nodes(cycle);
  }
}

/* compute all cycles from e to e in scc as well as their associated
   transformers (too bad for the convex approximation). Beware of internal
   cycles. Compute the precondition for the entry point and propagate it
   along each cycle. Union preconditions between cycles. */
static void process_ready_scc_for_one_entry(list scc,
					    transformer pre_entry,
					    unstructured u,
					    control e)
{
  list cycle = CONS(CONTROL, e, NIL);

  ifdebug(5) {
    pips_debug(5, "Begin for scc:\n");
    print_control_nodes(scc);
    pips_debug(5, "with entry point: %s\n",
	  statement_identification(control_statement(e)));
  }

  build_control_cycles_in_scc(scc, cycle, pre_entry, u);

  pips_debug(5, "End\n");
}

/* Each entry node can be processed independently and the resulting
preconditions be unioned. For a given entry node, a transformer must be
computed for each cycle broken at the chosen entry node. Internal cycles
must be processed differently, without entry precondition. Preconditions
for each cycle must be unioned over the cycles. */

static void process_ready_scc(list scc, transformer pre_entry, unstructured u)
{
  list entry_nodes = find_scc_entry_nodes(scc, u);

  ifdebug(5) {
    pips_debug(5, "Begin for scc:\n");
    print_control_nodes(scc);
    pips_debug(5, "with entry nodes:\n");
    print_control_nodes(entry_nodes);
  }

  /* initialize the postconditions in the scc with a neutral value without
     propagating the conditions obtained downwards in the related
     statement */
  MAP(CONTROL, c, {
      transformer post = transformer_empty();
      statement stmt = control_statement(c);
      store_control_postcondition(stmt, post);
  }, scc);

  /* Process each entry node */
  MAP(CONTROL, e, {
    process_ready_scc_for_one_entry(scc, pre_entry, u, e);
  }, entry_nodes);

  /* Propagate the preconditions downwards in the underlying statements */
  MAP(CONTROL, c, {
      statement stmt = control_statement(c);
  }, scc);
  pips_debug(5, "End\n");
}

/* compute pre- and post-conditions in an unstructured from the entry
   precondition and return the exit postcondition */
transformer unstructured_to_accurate_postconditions(transformer pre_u,
						    transformer pre, unstructured u)
{
  transformer post = transformer_undefined;

  pips_debug(5, "Begin\n");

  wide_forward_control_map_get_blocs(unstructured_control(u), &to_be_processed);
  still_to_be_processed = gen_copy_seq(to_be_processed);
  already_processed = NIL;
  statement_to_control_postcondition_map = MAKE_STATEMENT_MAPPING();

  pips_assert("Node lists are defined", !list_undefined_p(to_be_processed)
	      && !list_undefined_p(still_to_be_processed) && ENDP(already_processed) );

  while(!ENDP(still_to_be_processed)) {
    int count = -1;
    do {
      list l = list_undefined;

      /* process forward */
      pips_debug(5, "Process forward for\n");
      ifdebug(5) print_control_nodes(still_to_be_processed);

      count = 0;
      for(l=still_to_be_processed; !ENDP(l); ) {
	control c = CONTROL(CAR(l));
	POP(l); /* right away because c's cdr might be modified */
	if(ready_to_be_processed_p(c)) {
	  process_ready_node(c, pre, u);
	  gen_remove(&still_to_be_processed, c);
	  already_processed = CONS(CONTROL, c, already_processed);
	  count++;
	}
      }
    } while(count!=0);
    if(!ENDP(still_to_be_processed)) {
      list scc = list_undefined;
      /* find a scc and process it */
      pips_debug(5, "Find a scc and process it\n");
      scc = find_scc_in_cfg();
      pips_assert("scc is defined\n", scc!=list_undefined);
      pips_assert("scc is not empty\n", !ENDP(scc));
      pips_debug(5, "scc found:\n");
      ifdebug(5) print_control_nodes(scc);
      process_ready_scc(scc, pre, u);
      pips_error("unstructured_to_accurate_postconditions", "Not implemented yet!");
    }
  }

  /* Make sure that all control nodes have been processed */
  gen_recurse(u, control_domain, process_unreachable_node, gen_null);

  post = copy_transformer
    (load_control_postcondition(control_statement(unstructured_exit(u))));
  free_control_postcondition();
  gen_free_list(to_be_processed);
  gen_free_list(still_to_be_processed);
  gen_free_list(already_processed);

  pips_debug(5, "End\n");

  return post;
}
