/* semantical analysis
  *
  * Accurate propagation of transformers and postconditions in
  * unstructured (i.e. CFG).
  *
  * It is assumed that CFG are small graphs because proper restructuring
  * has been applied before this analysis is performed. Large graphs with
  * up to hundreds of nodes as found in some industrial codes cannot be
  * effectively handled by this implementation. Arbitrary thresholds are
  * used to avoid expensive analysis and simple fix-point operators are
  * used instead.
  *
  * The control flow graph is decomposed into strongly connected
  * components. Each component is processed in order and preconditions or
  * transformers are propagated from the entry point.
  *
  * If a component has only one node, it is easy to process.
  *
  * If a component is a SCC, it is processed for each of its entry
  * points. Transformers and preconditions are unioned over all entries.
  *
  * If a SCC is a cycle, the cycle is broken at the local entry point and
  * a fix-point is computed for each path. All fix-points are unioned.
  *
  * If a SCC is more complex, it is again decomposed but in a very crude
  * way because a recursive call to process sub-SCC is not
  * performed. Instead, regardless of local entry points, transformer
  * fix-points are computed for each sub-SCC. No local preconditions are
  * available to sharpen the transformer analysis. If a sub-SCC is a
  * cycle, its transformer path fix-point is computed. If not, an
  * approximation is made, equivalent to adding control arcs between
  * nodes which guarantee the correctness.
  *
  * Transformers between the entry point and after the current node are
  * very similar to postcondition between the module entry point and the
  * current node. Transformers are obtained like precondition but without
  * a precondition at the CFG entry point.
  *
  * This implementation is not great. It was quickly developped to fix the
  * problems encountered by Nga Nguyen in array bound checking for a PLDI
  * submission. I missed the crucial distinction between a scc, a path and
  * a cycle (or circuit) at first and function names are thus quite
  * misleading. But the results needed were obtain on time.
  *
  * It might be useful to add a pseudo-node as predecessor of the CFG
  * entry node. This pseudo-node would simplify the algorithms and the
  * function profiles. Its post-condition would be the precondition of the
  * CFG or no information. Without it, each node must be checked to see if
  * it is the entry node because, then, it has an extra-predecessor.
  *
  * I should write a report to explain the solver used and the interplay
  * between transformers and preconditions. It would be useful to prove it
  * as well. The use of transformers make it quite fast because it is
  * linear with the nesting depth. It makes it also less accurate. The
  * check over all paths is expensive and restricts its use to small
  * CFG. This is not an algorithmic issue but an implementation
  * issue: Transformers and preconditions associated to partial paths are
  * not stored but recomputed for each possible path.
  *
  * Francois Irigoin, October 2000
  * $Id$
  *
  * $Log: unstructured.c,v $
  * Revision 1.5  2001/02/02 12:17:48  irigoin
  * After bug fixes for tilt.f, before cleaning up and before bug fixes for
  * spice and fppp
  *
  * Revision 1.4  2000/12/04 16:36:13  irigoin
  * Comments added to explain the algorithm used
  *
  * Revision 1.3  2000/11/23 17:16:11  irigoin
  * Too many modifications. Lots of bug fixes for PLDI'2001. New debugging
  * statements, new functions, new consistency checks.
  *
  * Revision 1.2  2000/11/03 17:12:57  irigoin
  * New version on Nov. 3
  *
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
/* static list to_be_processed = list_undefined; */
/*
static bool to_be_processed_p(control c)
{
  return gen_find_eq(c, to_be_processed)!=gen_chunk_undefined;
}
*/
/* These control nodes have a postcondition. They may not be successors of
   the entry node because some predecessors may not be reachable and still
   be processed. */
/* static list already_processed = list_undefined; */
/*
static bool already_processed_p(control c)
{
  return gen_find_eq(c, already_processed)!=gen_chunk_undefined;
}
*/
/* These control nodes are successors of the entry node but they have not
   yet received a postcondition*/
/* static list still_to_be_processed = list_undefined; */
/*
static bool still_to_be_processed_p(control c)
{
  return gen_find_eq(c, still_to_be_processed)!=gen_chunk_undefined;
}
*/
/* heads of sub-cycles */
/* static list secondary_entries = list_undefined; */
/*
static bool secondary_entry_p(control c)
{
  return gen_in_list_p(c, secondary_entries);
}
*/

static void print_control_node(control c)
{
  fprintf(stderr,
	  "ctr %p, %d preds, %d succs: %s", 
          c,
	  gen_length(control_predecessors(c)),
	  gen_length(control_successors(c)),
	  statement_identification(control_statement(c)));
  fprintf(stderr,"\tsuccessors:\n");
  MAP(CONTROL, s, {
    fprintf(stderr, "\t\t%p %s", s,
	    statement_identification(control_statement(s)));
  }, control_successors(c));
  fprintf(stderr,"\tpredecessors:\n");
  MAP(CONTROL, p, {
    fprintf(stderr, "\t\t%p %s", p,
	    statement_identification(control_statement(p)));
  }, control_predecessors(c));
  fprintf(stderr, "\n");
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
/* static statement_mapping statement_to_control_postcondition_map = NULL; */

static transformer load_control_postcondition(statement stat,
					      statement_mapping control_postcondition_map)
{
  transformer post = (transformer)
    hash_get((hash_table) control_postcondition_map,
	     (char *) stat);

  if(post == (transformer) HASH_UNDEFINED_VALUE)
    post = transformer_undefined;

  return post;
}

static void store_control_postcondition(statement stat, transformer post,
					statement_mapping control_postcondition_map)
{
  pips_debug(8, "Store postcondition for statement %s\n",
	     statement_identification(stat));
  ifdebug(8) {
    print_transformer(post);
  }

  pips_assert("The postcondition to insert is consistent", 
	      transformer_consistency_p(post));

  pips_assert("The postcondition is not defined yet", 
	      hash_get((hash_table) control_postcondition_map, (char *) stat)
	      == HASH_UNDEFINED_VALUE);
  hash_put((hash_table) (control_postcondition_map),
	   (char *)stat, (char *) post);
}

static void update_control_postcondition(statement stat, transformer post,
					 statement_mapping control_postcondition_map)
{
  pips_debug(8, "Update postcondition for statement %s\n",
	     statement_identification(stat));
  ifdebug(8) {
    print_transformer(post);
  }

  pips_assert("The postcondition is already defined", 
	      hash_get((hash_table) control_postcondition_map, (char *) stat)
	      != HASH_UNDEFINED_VALUE);
  hash_update((hash_table) (control_postcondition_map),
	      (char *)stat, (char *) post);
}

static statement_mapping make_control_postcondition()
{
  statement_mapping control_postcondition_map = NULL;
  control_postcondition_map = MAKE_STATEMENT_MAPPING();
  return control_postcondition_map;
}

static statement_mapping free_control_postcondition(statement_mapping control_postcondition_map)
{
  /* all postconditions must be freed */
  HASH_MAP(k,v,{
    free_transformer((transformer) v);
  }, control_postcondition_map);

  /* as well as the map */
  /* hash_table_free(statement_to_control_postcondition_map); */
  FREE_STATEMENT_MAPPING(control_postcondition_map);

  return NULL;
}

/* a control can be considered already processed if it has been processed
   already or if it can receive a trivial empty postcondition */
static bool considered_already_processed_p(control c, list to_be_processed,
					   list already_processed,
					   statement_mapping control_postcondition_map)
{
  bool already = gen_in_list_p(c, already_processed);

  if(!already) {
    if(!gen_in_list_p(c, to_be_processed)) {
      /* unreachable node with empty precondition */
      /* transformer pre = transformer_empty(); */
      statement stmt = control_statement(c);
      transformer post = load_control_postcondition(stmt, control_postcondition_map);

      if(transformer_undefined_p(post)) {
	post = transformer_empty();
	/* transformer post = statement_to_postcondition(pre, stmt); */
	store_control_postcondition(stmt, post, control_postcondition_map);
      }
      else {
	pips_assert("Postcondition for unreachable nodes must be empty",
		    transformer_empty_p(post));
      }
      already = TRUE;
    }
  }

  return already;
}

static bool nodes_considered_already_processed_p(list l, list to_be_processed,
						 list already_processed,
						 statement_mapping control_postcondition_map)
{
  bool ready = TRUE;

  MAP(CONTROL, c, {
    if(considered_already_processed_p(c, 
				      to_be_processed,
				      already_processed,
				      control_postcondition_map)) {
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
static bool ready_to_be_processed_p(control c,
				    list to_be_processed,
				    list still_to_be_processed,
				    list already_processed,
				    statement_mapping control_postcondition_map)
{
  bool ready = TRUE;
  MAP(CONTROL, pred, {
    if(gen_in_list_p(pred, already_processed)) {
      /* useless, except for debugging */
      ready &= TRUE;
    }
    else if(!gen_in_list_p(pred, to_be_processed)) {
      /* postcondition must be empty because pred is not reachable */
      /* transformer pre = transformer_empty(); */
      statement stmt = control_statement(pred);
      transformer post = load_control_postcondition(stmt, control_postcondition_map);

      if(transformer_undefined_p(post)) {
	post = transformer_empty();
	/* transformer post = statement_to_postcondition(pre, stmt); */
	store_control_postcondition(stmt, post, control_postcondition_map);
      }
      else {
	pips_assert("Postcondition for unreachable nodes must be empty",
		    transformer_empty_p(post));
      }
    }
    else if(gen_in_list_p(pred, still_to_be_processed)) {
      ready &= FALSE;
    }
    else {
      pips_error("ready_to_be_processed_p",
		 "node pred does not belong to any category: %s",
		 statement_identification(control_statement(pred)));
    }
  }, control_predecessors(c));
  return ready;
}

static bool nodes_ready_to_be_processed_p(list l,
				    list to_be_processed,
				    list still_to_be_processed,
				    list already_processed,
				    statement_mapping control_postcondition_map)
{
  bool ready = TRUE;

  MAP(CONTROL, c, {
    if(ready_to_be_processed_p
       (c, to_be_processed, still_to_be_processed, already_processed, 
	control_postcondition_map)) {
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
static transformer load_predecessor_postcondition_or_test_condition
(control pred,
 control c,
 bool postcondition_p,
 statement_mapping control_postcondition_map)
{
  statement stmt = control_statement(pred);
  transformer post = postcondition_p? 
    transformer_dup(load_control_postcondition(stmt, control_postcondition_map))
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

static transformer load_predecessor_postcondition(control pred, control c,
 statement_mapping control_postcondition_map)
{
  transformer post = load_predecessor_postcondition_or_test_condition
    (pred, c, TRUE, control_postcondition_map);
  return post;
}

static transformer load_predecessor_test_condition(control pred, control c,
 statement_mapping control_postcondition_map)
{
  transformer post = load_predecessor_postcondition_or_test_condition
    (pred, c, FALSE, control_postcondition_map);
  return post;
}

static void process_ready_node(control c, transformer pre_entry, unstructured u,
			       statement_mapping control_postcondition_map,
			       bool postcondition_p)
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
    pre = load_predecessor_postcondition(CONTROL(CAR(preds)), c, control_postcondition_map);
    POP(preds);
  }

  MAP(CONTROL, pred, {
    transformer npre = load_predecessor_postcondition(pred, c, control_postcondition_map);
    transformer lpre = pre;

    pre = transformer_convex_hull(npre, lpre);
    /* memory leak with lpre. pre_entry and postconditions cannot be
       freed: lpre cannot be freed during the first iteration */
    if(pred!=CONTROL(CAR(preds))) free_transformer(lpre);
    lpre = pre;
  }, preds);

  if(postcondition_p) {
    post = statement_to_postcondition(pre, stmt);
  }
  else {
    transformer tf = load_statement_transformer(stmt);
    post = transformer_apply(tf, pre);
  }

  store_control_postcondition(stmt, post, control_postcondition_map);
}

static bool process_unreachable_node(control c,
				     statement_mapping control_postcondition_map,
				     bool postcondition_p)
{
  statement s = control_statement(c);

  pips_debug(9,"Statement %s", statement_identification(s));

  if(statement_ordering(s)==STATEMENT_ORDERING_UNDEFINED) {
    pips_user_warning("Improper control restructuration, statement with no ordering:%s",
		      statement_identification(s));
  }

  if(transformer_undefined_p
     (load_control_postcondition(s, control_postcondition_map))) {
    transformer post = transformer_undefined;

    if(postcondition_p) {
      transformer pre = transformer_empty();
      post = statement_to_postcondition(pre, s);
    }
    else {
      /* Careful, this may have been done earlier by the CONTROL_MAP in
         unstructured_to_transformers() */
      if(transformer_undefined_p(load_statement_transformer(s))) {
	(void) statement_to_transformer(s);
      }
      post = transformer_empty();
    }

    pips_user_warning("After restructuration, unexpected unreachable node:%s",
		      statement_identification(s));

    store_control_postcondition(s, post, control_postcondition_map);
  }
  else if(postcondition_p
	  && transformer_undefined_p(load_statement_precondition(s))) {
    /* Problem with ELSIP in ARC2D after partial redundancy elimination */
    /* pips_error("Statement has a postcondition but no precondition:%s",
       statement_identification(s)); */
    transformer pre = transformer_empty();
    transformer post = statement_to_postcondition(pre, s);

    pips_assert("the new postcondition is empty", transformer_empty_p(post));
    pips_assert("the previous postcondition is empty", 
		transformer_empty_p(load_control_postcondition(s, control_postcondition_map)));

    pips_user_warning("After restructuration (?),"
		      " postcondtion for unexpected unreachable node:%s",
		      statement_identification(s));
  }
  else if(!postcondition_p
	  && transformer_undefined_p(load_statement_transformer(s))) {
    /* Problem with SHALLOW in SWIM */
    (void) statement_to_transformer(s);

    pips_user_warning("After restructuration (?),"
		      " transformer for unexpected unreachable node:%s",
		      statement_identification(s));
  }

  return TRUE;
}

static void check_scc_for_exit_node(list scc, control e)
{
  /* e is assumed to be the exit node of an unstructured. As such it
     cannot have any successor. So it cannot be part of a scc. */
  if(e==gen_find_eq(e, scc)) {
    pips_assert("A node in a scc has at least one successor and no more than two",
		gen_length(control_successors(e))>=1
		&& gen_length(control_successors(e))<=2);
    pips_debug
      (1,"Exit node %s has successor %s",
       statement_identification(control_statement(e)),
       statement_identification
       (control_statement(CONTROL(CAR(control_successors(e))))));
    if(gen_length(control_successors(e))==2) {
      pips_debug
	(1,"And successor %s",
	 statement_identification
	 (control_statement(CONTROL(CAR(CDR(control_successors(e)))))));
    }
    pips_error("check_scc_for_exit_node", "Illegal scc with exit node!");
  }
  /* in a scc, each node should have at least one successor: nodes with no
     sucessors cannot be member of an scc */
  MAP(CONTROL, c, {
    pips_assert("Each scc node has one or two successors", 
		gen_length(control_successors(c))==2
		|| gen_length(control_successors(c))==1)
  }, scc);
}

static list find_ready_scc_in_cfg(unstructured u,
				  list to_be_processed,
				  list still_to_be_processed,
				  list already_processed,
				  statement_mapping control_postcondition_map)
{
  list to_be_searched_for_scc = gen_copy_seq(still_to_be_processed);
  list scc = list_undefined;
  list l = list_undefined;

  pips_debug(5,"Begin\n");

  pips_assert("List still_to_be_processed is not empty",
	      !ENDP(still_to_be_processed));

  for(l=to_be_searched_for_scc; !ENDP(l); ) {
    control head = CONTROL(CAR(l));
    list succs = NIL;
    list preds = NIL;
    list scc_preds = list_undefined;

    pips_debug(5,"Build scc for head %s",
	       statement_identification(control_statement(head)));

    forward_control_map_get_blocs(head, &succs);

    ifdebug(5) {
      pips_debug(5,"All successors:\n");
      print_control_nodes(succs);
    }

    backward_control_map_get_blocs(head, &preds);

    ifdebug(5) {
      pips_debug(5,"All predecessors:\n");
      print_control_nodes(succs);
    }

    gen_list_and(&succs, preds);
    scc = succs;

    if(gen_length(scc)==1) {
      /* because successors and predecessors are a * closure, one node may
         be seen as a scc although it is not its own successor and
         predecessor. This happens with a node without successors. */
      control h = CONTROL(CAR(scc));

      if(ENDP(control_successors(h))) {
	POP(l);
	continue;
      }
    }
    check_scc_for_exit_node(scc, unstructured_exit(u));

    gen_list_and_not(&preds, scc);
    scc_preds = preds;

    if(nodes_considered_already_processed_p(scc_preds, to_be_processed, 
					    already_processed,
					    control_postcondition_map)) {
      /* exit loop */

      ifdebug(5) {
	pips_debug(5,"Ready scc is:\n");
	print_control_nodes(scc);
      }
      l = NIL;
    }
    else {
      /* scc should be removed from to_be_searched to gain time! */
      pips_debug(5,"Look for another scc\n");
      gen_free_list(scc);
      POP(l);
    }
    gen_free_list(scc_preds);
  }

  gen_free_list(to_be_searched_for_scc);

  pips_debug(5,"End\n");

  return scc;
}

/* It is assumed that the scc is ready to be processed: all its
   predecessors must have postconditions. */
static list find_scc_entry_nodes(list scc, unstructured u, list already_processed)
{
  control first = unstructured_control(u);
  list entry_nodes = (first==gen_find_eq(first, scc))?
    CONS(CONTROL, first, NIL):NIL;

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

/* Control information is ignored. All nodes are considered successors of
   all nodes. This is an over-approximation of the real control flow */
static transformer control_node_set_to_fix_point
(list set,
 list secondary_entries,
 statement_mapping statement_to_subcycle_fix_point_map)
{
  transformer settf = transformer_empty();
  transformer fptf = transformer_undefined;
  static transformer subcycle_to_fixpoint(list, control,
					  statement_mapping,
					  list,
					  statement_mapping);

  ifdebug(5) {
    pips_debug(5, "Begin for set:\n");
    print_control_nodes(set);
  }

  MAP(CONTROL, c, {
    statement s = control_statement(c);
    transformer tf = load_statement_transformer(s);
    transformer new_settf = transformer_convex_hull(settf, tf);

    /* do not forget the possible subcycle */
    if(c!=CONTROL(CAR(set)) && gen_in_list_p(c, secondary_entries)) {
      transformer tmp_settf = transformer_undefined;
      transformer fptf = 
	(transformer) hash_get((hash_table) statement_to_subcycle_fix_point_map,
			       (char *) s);
      /* I do not see why it should be ready. We should compute it
	 recursively or thier computations should have been ordered in
	 process_cycle_in_scc() */
      /* pips_assert("The fix point is ready in control_node_set_to_fix_point",
	 fptf!= (transformer) HASH_UNDEFINED_VALUE); */
      if(fptf== (transformer) HASH_UNDEFINED_VALUE) {
	fptf = subcycle_to_fixpoint(set, c,
				    NULL /* control_postcondition_map*/,
				    secondary_entries,
				    statement_to_subcycle_fix_point_map);

	pips_assert("The fix point transformer to insert is consistent", 
		    transformer_consistency_p(fptf));

	hash_put((hash_table) statement_to_subcycle_fix_point_map,
		 (char *) s, (char *) fptf);
      }

      tmp_settf = new_settf;
      new_settf = transformer_convex_hull(tmp_settf, fptf);
      free_transformer(tmp_settf);
    }

    free_transformer(settf);
    settf = new_settf;
  }, set);

  ifdebug(5) {
    pips_debug(5, "Set transformer:");
    print_transformer(settf);
  }

  fptf = (* transformer_fix_point_operator)(settf);

  ifdebug(5) {
    pips_debug(5, "End with set fix point transformer:");
    print_transformer(fptf);
  }

  return fptf;
}

/* Assume the sequence is a simple cycle */
static transformer control_node_sequence_to_fix_point
(list seq, statement_mapping control_postcondition_map, list secondary_entries,
 statement_mapping statement_to_subcycle_fix_point_map)
{
  transformer seqtf = transformer_identity();
  transformer fptf = transformer_undefined;
  control pred = control_undefined;

  ifdebug(5) {
    pips_debug(5, "Begin for sequence:");
    print_control_nodes(seq);
  }

  MAP(CONTROL, c, {
    transformer tf_with_cond = transformer_undefined;
    statement s = control_statement(c);

    /* do not forget the possible subcycles, except the one we might be trying to compute */
    if(c!=CONTROL(CAR(seq)) && gen_in_list_p(c, secondary_entries)) {
      transformer fptf = 
	(transformer) hash_get((hash_table) statement_to_subcycle_fix_point_map,
			       (char *) s);
      /* I do not see why it should be ready. We should compute it recursively */
      pips_assert("The fix point is ready in control_node_sequencet_to_fix_point",
		  fptf!= (transformer) HASH_UNDEFINED_VALUE);
      tf_with_cond = copy_transformer(fptf);
      tf_with_cond = transformer_combine(tf_with_cond,
					 load_statement_transformer(s));
    }
    else {
      tf_with_cond =  copy_transformer(load_statement_transformer(s));
    }

    if(!control_undefined_p(pred)) {
      transformer ctf =
	load_predecessor_test_condition(pred, c,
							control_postcondition_map);
      transformer new_tf_with_cond = transformer_combine(ctf, tf_with_cond);

      free_transformer(tf_with_cond);
      tf_with_cond = new_tf_with_cond;
    }
    seqtf = transformer_combine(seqtf, tf_with_cond);
    free_transformer(tf_with_cond);
    pred = c;
  }, seq);

  ifdebug(5) {
    pips_debug(5, "Sequence transformer:\n");
    print_transformer(seqtf);
  }

  fptf = (* transformer_fix_point_operator)(seqtf);

  ifdebug(5) {
    pips_debug(5, "End with sequence fix point transformer:");
    print_transformer(fptf);
  }

  return fptf;
}

/* Returns a control list starting with the first node from a set of
control nodes. Follow control paths as much as possible to improve
postcondition propagation. The set is assumed to be non-empty. Abort only
if a control has a wrong number of successors. It is assumed that set is a
scc. */
static list control_set_to_control_list(list set)
{
  list path = list_undefined;
  list unassigned = list_undefined;

  ifdebug(5) {
    pips_debug(5, "Find a list covering the set:\n");
    print_control_nodes(set);
    MAP(CONTROL, c, {
      print_control_node(c);
    }, set);
  }

  pips_assert("Set set is defined and not empty",
	      set!=list_undefined && !ENDP(set));

  path = CONS(CONTROL, CONTROL(CAR(set)), NIL);
  unassigned = gen_copy_seq(CDR(set));

  while(!ENDP(unassigned)) {
    control l = CONTROL(CAR(gen_last(path)));
    list succs = gen_copy_seq(control_successors(l));
    int count = 0;
    control next = control_undefined;

    /* Try to find a successor to the last node in path */
    gen_list_and(&succs, unassigned);
    
    count = gen_length(succs);

    if(count==1||count==2) {
      /* choose the first successor in unassigned */
      next = CONTROL(CAR(succs));
    }
    else if(count!=0) {
      pips_error("control_set_to_control_path", "Node has %d successors %s", count,
		 statement_identification(control_statement(l)));
    }

    gen_free_list(succs);

    /* If no successor has been found, start a new control path */
    if(control_undefined_p(next)) {
      pips_assert("There are still nodes to add to path", !ENDP(unassigned));
      /* compute the set of successors of nodes in path which are unassigned */
      MAP(CONTROL, c, {
	list succs2 = gen_copy_seq(control_successors(c));
	gen_list_and(&succs2, unassigned);
	if(!ENDP(succs2)) {
	  pips_assert("There is at most one successor in unassigned",
		      ENDP(CDR(succs2)));
	  next = CONTROL(CAR(succs2));
	  gen_free_list(succs2);
	  break;
	}
	else {
	  gen_free_list(succs2);
	}
      }, path);
    }

    /* add node next to path */
    pips_assert("A node is added at each iteration",!control_undefined_p(next));
    gen_nconc(path, CONS(CONTROL, next, NIL));
    gen_remove(&unassigned, next);
  }

  ifdebug(5) {
    pips_debug(5, "End with list:\n");
    print_control_nodes(path);
  }

  return path;
}
/* Returns a control path starting with the first node from a set of
control nodes. The set is assumed to be non-empty. Returns an empty path
if no solution exists. */
static list control_set_to_control_path(list set)
{
  list path = list_undefined;
  list unassigned = list_undefined;
  /* control f = control_undefined; */
  /* control l = control_undefined; */

  ifdebug(5) {
    pips_debug(5, "Find a path in set:\n");
    print_control_nodes(set);
  }

  pips_assert("Set set is defined and not empty",
	      set!=list_undefined && !ENDP(set));

  path = CONS(CONTROL, CONTROL(CAR(set)), NIL);
  unassigned = gen_copy_seq(CDR(set));

  while(!ENDP(unassigned)) {
    control l = CONTROL(CAR(gen_last(path)));
    list succs = gen_copy_seq(control_successors(l));
    int count = 0;
    control next = control_undefined;

    gen_list_and(&succs, unassigned);
    
    count = gen_length(succs);
    if(count==1) {
      next = CONTROL(CAR(succs));
      gen_nconc(path, CONS(CONTROL, next, NIL));
    }
    else if(count==2) {
      /* The two successors may be a unique node */
      if(CONTROL(CAR(succs))==CONTROL(CAR(CDR(succs)))){
	next = CONTROL(CAR(succs));
	gen_nconc(path, CONS(CONTROL, next, NIL));
      }
      else {
	/* Give up! */
	/* pips_error("control_set_to_control_path", "Subcycle starting at %s",
	   statement_identification(control_statement(l))); */
	gen_free_list(path);
	path = NIL;
	break;
      }
    }
    else if(count==0) {
      pips_error("control_set_to_control_path", "Node has no successor %s",
		   statement_identification(control_statement(l)));
    }
    else {
      pips_error("control_set_to_control_path", "Node has %d successors %s", count,
		 statement_identification(control_statement(l)));
    }
    gen_free_list(succs);
    gen_remove(&unassigned, next);
  }

  ifdebug(5) {
    if(ENDP(path)) {
      pips_debug(5, "End with no covering path\n");
    }
    else {
      pips_debug(5, "End with path:\n");
      print_control_nodes(path);
    }
  }

  return path;
}

/* If there is no cycle, return an empty path */
static list control_set_to_control_cycle(list set)
{
  list path = control_set_to_control_path(set);


  /* If a path has been found, check that it is a cycle */
  if(!ENDP(path)) {
    control f = CONTROL(CAR(path));
    control l = CONTROL(CAR(gen_last(path)));

    /* Check that f is the only successor in path of the last node in the path */
    if(gen_in_list_p(f, control_successors(l))) {
      if(gen_length(control_successors(l))==1) {
	/* OK */
	;
      }
      else if(gen_length(control_successors(l))==2) {
	control succ1 = CONTROL(CAR(control_successors(l)));
	control succ2 = CONTROL(CAR(CDR(control_successors(l))));
	control other = (f==succ1)? succ2 : succ1;

	if(other!=f && gen_in_list_p(other, path)) {
	  pips_error("control_set_to_control_cycle",
		     "Set contains more than one cycles with %s",
		     statement_identification(control_statement(other)));
	}
      }
      else {
	int count = gen_length(control_successors(l));
	pips_error("control_set_to_control_cycle",
		   "Node contains wrong number of successors %d: ",
		   count, statement_identification(control_statement(l)));
      }
    }
    else {
      pips_error("control_set_to_control_cycle", "Set contains no cycles");
    }
  }

  return path;
}

/* This is an auxiliary function for the next one */
static void recursive_subscc_to_cycle_heads(list scc, control h, list path, list * pcycle_heads)
{
  control last = CONTROL(CAR(path));

  MAP(CONTROL, succ, {
    if(gen_in_list_p(succ, scc) && succ != h) {
      if(gen_in_list_p(succ, path)) {
	if(!gen_in_list_p(succ, *pcycle_heads)) {
	  * pcycle_heads = gen_append(* pcycle_heads, CONS(CONTROL, succ, NIL));
	  ifdebug(6) {
	    pips_debug(6, "Current path with %d nodes:\n",
		       gen_length(path));
	    print_control_nodes(path);
	    pips_debug(6, "New cycle head %s", 
		       statement_identification(control_statement(succ)));
	  }
	}
      }
      else {
	list new_path = CONS(CONTROL, succ, path);

	recursive_subscc_to_cycle_heads(scc, h, new_path, pcycle_heads);
	CDR(new_path) = NIL;
	gen_free_list(new_path);
      }
    }
  }, control_successors(last));
}

/* Find heads for cycles in scc, cycles which do not contain h. Path is build backwards. */
static list subscc_to_cycle_heads(list scc, control h)
{
  list cycle_heads = NIL;
  list path = CONS(CONTROL, h, NIL);

  ifdebug(6) {
    pips_debug(6, "Begin for head %s", 
	       statement_identification(control_statement(h)));
    pips_debug(6, "scc contains %d nodes:\n", gen_length(scc));
    print_control_nodes(scc);
  }

  recursive_subscc_to_cycle_heads(scc, h, path, &cycle_heads);

  ifdebug(6) {
    pips_debug(6, "Subcycle contains %d secondary entries:\n",
	       gen_length(cycle_heads));
    print_control_nodes(cycle_heads);
  }

  return cycle_heads;
}

/* find begining of cycles in the scc containing h in cycle. This
   algorithm is wrong is several control paths lead to the same node. A
   cycle will be assumed when it only is a join. */
static list scc_to_secondary_entries(list cycle, control h)
{
  list secondary_entries = NIL;
  int reached = 0;
  list reached_nodes = NIL;
  list newly_reached_nodes = CONS(CONTROL, h, NIL);

  pips_assert("head belongs to scc", gen_in_list_p(h, cycle));

  do {
    list reachable_nodes = NIL;
    reached = 0;
    MAP(CONTROL,c ,{
      MAP(CONTROL, succ, {
	ifdebug(6) {
	  print_control_node(succ);
	}
	if(gen_in_list_p(succ, cycle)) {
	  if(gen_in_list_p(succ, reached_nodes)
	     || gen_in_list_p(succ, newly_reached_nodes)) {
	    if(!gen_in_list_p(succ, secondary_entries) && succ!= h) {
	      secondary_entries = CONS(CONTROL, succ, secondary_entries);
	    }
	  }
	  else {
	    reachable_nodes = CONS(CONTROL, succ, reachable_nodes);
	    reached++;
	  }
	}
      }, control_successors(c));
    }, newly_reached_nodes);

    reached_nodes = gen_nconc(reached_nodes, newly_reached_nodes);
    newly_reached_nodes = reachable_nodes;
  } while(reached!=0);

  return secondary_entries;
}

/* Find the scc's containing h but not the first control node in cycle */
static list head_to_subcycle(list cycle, control h, list * new_secondary_entries)
{
  list subcycle = list_undefined;
  list preds = NIL;
  list succs = NIL;
  control f = CONTROL(CAR(cycle));
  control outernode = control_undefined;
  /* list new_secondary_entries = list_undefined; */

  pips_debug(5, "Begin for head %s", 
	     statement_identification(control_statement(h)));

  pips_assert("The subcycle head is in the cycle", gen_in_list_p(h, cycle));

  /* *new_secondary_entries = scc_to_secondary_entries(cycle, h); */
  *new_secondary_entries = subscc_to_cycle_heads(cycle, h);

  if(h!=f) {
    /* subcycle or more precisely, sub-scc for h is made of h predecessors
       and successors which are not linked thru the cycle entry node f */
    outernode = f;
  }
  else {
    /* one of h's predecessors must not be in the subcycle, or h would not
       be a secondary entry. */
    list outernodes = gen_copy_seq(control_predecessors(h));

    gen_list_and(&outernodes, new_secondary_entries);


    pips_assert("At least one node can be used to break the master scc",
		gen_length(outernodes)>=1);
    if(gen_length(outernodes)>=2) {
      pips_user_warning("Several opportunities to break the scc: %d\n",
			gen_length(outernodes));
    }
    outernode = CONTROL(CAR(outernodes));
    gen_free_list(outernodes);
  }

  forward_control_map_get_blocs_but(h, outernode, &succs);
  backward_control_map_get_blocs_but(h, outernode, &preds);
  gen_list_and(&succs, preds);
  subcycle = succs;

  /* get rid of nodes in the main cycle but add again the head */
  /*
    gen_list_and_not(&subcycle, cycle);
    subcycle = CONS(CONTROL, h, subcycle);
  */

  /* make sure that h comes first */
  if(gen_in_list_p(h, subcycle)) {
    gen_remove(&subcycle, h);
    subcycle = CONS(CONTROL, h, subcycle);
  }
  else {
    /* Here, we are in trouble! */
    ifdebug(5) {
      pips_debug(5, "Head is %s", statement_identification(control_statement(h)));
      pips_debug(5, "Outer node is %s", statement_identification(control_statement(outernode)));
      pips_debug(5, "Initial cycle is:\n");
      print_control_nodes(cycle);
      pips_debug(5, "Predecessors are:\n");
      print_control_nodes(preds);
      pips_debug(5, "Successors are:\n");
      forward_control_map_get_blocs_but(h, outernode, &succs);
      print_control_nodes(succs);
      pips_debug(5, "Subcycle is:\n");
      print_control_nodes(subcycle);
    }
    pips_assert("The head of a scc must be in the scc", FALSE);
  }

  if(gen_length(subcycle)==1) {
    control unique = CONTROL(CAR(subcycle));
    if(gen_in_list_p(unique, control_predecessors(unique))
       && gen_in_list_p(unique, control_successors(unique))) {
      pips_user_warning("One node cycle\n", statement_identification(control_statement(unique)));
    }
    else {
      /* We are in trouble */
      pips_debug(5, "Head is %s", statement_identification(control_statement(h)));
      pips_debug(5, "Outer node is %s", statement_identification(control_statement(outernode)));
      pips_debug(5, "Initial cycle is:\n");
      print_control_nodes(cycle);
      pips_debug(5, "Predecessors are:\n");
      print_control_nodes(preds);
      pips_debug(5, "Successors are:\n");
      forward_control_map_get_blocs_but(h, outernode, &succs);
      print_control_nodes(succs);
      pips_debug(5, "Subcycle is:\n");
      print_control_nodes(subcycle);
      pips_assert("The scc is a scc", FALSE);
    }
  }

  ifdebug(5) {
    pips_debug(5, "Subcycle contains %d nodes:\n", gen_length(subcycle));
    print_control_nodes(subcycle);
    pips_debug(5, "Subcycle contains %d secondary entries:\n",
	       gen_length(*new_secondary_entries));
    print_control_nodes(*new_secondary_entries);
  }
  return subcycle;
}

static transformer subcycle_to_fixpoint
(list cycle, control h, statement_mapping control_postcondition_map, list secondary_entries,
 statement_mapping statement_to_subcycle_fix_point_map)
{
  list new_secondary_entries = NIL;
  list subcycle = list_undefined;
  transformer fptf = transformer_undefined;
  list path = list_undefined;

  pips_debug(5, "Begin for head %s", 
	     statement_identification(control_statement(h)));

  subcycle = head_to_subcycle(cycle, h, &new_secondary_entries);

  if(!ENDP(path = control_set_to_control_cycle(subcycle))) {
    gen_free_list(subcycle);
    subcycle = path;

    ifdebug(5) {
      pips_debug(5, "Subcycle is:\n");
      print_control_nodes(subcycle);
    }

    pips_assert("subcycle is defined and starts with h", subcycle!=list_undefined
		&& CONTROL(CAR(subcycle))==h);

    fptf = control_node_sequence_to_fix_point(subcycle,
					      control_postcondition_map,
					      new_secondary_entries,
					      statement_to_subcycle_fix_point_map);

    ifdebug(5) {
      pips_debug(5, "For path:\n");
      print_control_nodes(subcycle);
      pips_debug(5, "End with fix point transformer:\n");
      print_transformer(fptf);
    }

    gen_free_list(subcycle);
  }
  else {
    /* path = control_set_to_control_cycle(subcycle); */
    /* gen_free_list(subcycle); */
    /* subcycle = path; */

    ifdebug(5) {
      pips_debug(5, "Sublist is:\n");
      print_control_nodes(subcycle);
    }

    pips_assert("sublist is defined and starts with h", subcycle!=list_undefined
		&& CONTROL(CAR(subcycle))==h);

    fptf = control_node_set_to_fix_point(subcycle,
					 new_secondary_entries,
					 statement_to_subcycle_fix_point_map );

    ifdebug(5) {
      pips_debug(5, "For list:\n");
      print_control_nodes(subcycle);
      pips_debug(5, "End with fix point transformer:\n");
      print_transformer(fptf);
    }

    gen_free_list(subcycle);
  }

  return fptf;
}

/* Initially, cycle had to be a cycle but list happens too. You cannot
   assume that the predecessor in cycle is a predecessor in the control
   graph. */
static void cycle_to_postconditions(list cycle,
				    transformer pre,
				    statement_mapping fp_map,
				    list secondary_entries,
				    statement_mapping control_postcondition_map)
{
  transformer post = copy_transformer(pre);
  list cc;

  pips_debug(5, "Begin\n");

  for(cc= cycle;!ENDP(cc); POP(cc)){
    control c = CONTROL(CAR(cc));
    statement s = control_statement(c);
    transformer previous_pre = transformer_undefined;
    transformer composite_pre = transformer_undefined;
    transformer real_pre = transformer_undefined;
    transformer tfs = transformer_undefined;

    /* Apply the subcycle */
    if(gen_in_list_p(c, secondary_entries)) {
      tfs = copy_transformer((transformer) hash_get((hash_table) fp_map, (char *) s));
      pips_assert("The fix point transformer is consistent", 
		  transformer_consistency_p(tfs));
    }
    else {
      tfs = transformer_identity();
    }
    /* Apply the node */
    tfs = transformer_combine(tfs, load_statement_transformer(s));
    real_pre = transformer_apply(tfs, post);
    free_transformer(tfs);

    /* If c has two different successors, add condition information */
    /* NO, this is not part of the node postcondition. The relevant
       information must be added later when the postconditions of
       predecessors are used to propagate preconditions */
    /*
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
    */
    /* convex hull with previous postcondition and storage */
    previous_pre = copy_transformer(load_control_postcondition(s, control_postcondition_map));
    pips_assert("Previous value of postcondition is consistent",
		transformer_consistency_p(previous_pre));
    pips_assert("Value of postcondition for current path is consistent",
		transformer_consistency_p(real_pre));
    composite_pre = transformer_convex_hull(previous_pre, real_pre);
    pips_assert("Previous value of postcondition is consistent",
		transformer_consistency_p(previous_pre));
    update_control_postcondition(s, composite_pre, control_postcondition_map);

    ifdebug(5) {
      pips_debug(5, "Postcondition previous_pre for statement %s",
		 statement_identification(s));
      print_transformer(previous_pre);
      pips_debug(5, "New poscondition composite_pre for statement %s",
		 statement_identification(s));
      print_transformer(composite_pre);
    }

    free_transformer(previous_pre);
    free_transformer(real_pre);
    free_transformer(post);

    /* add arc information */
    if(!ENDP(CDR(cc))) {
      control cnext = CONTROL(CAR(CDR(cc)));
      /* post = real_pre; */ /* composite_pre could be used as well */
      if(gen_in_list_p(cnext, control_successors(c))) {
	post = load_predecessor_postcondition(c, cnext, control_postcondition_map);
      }
      else {
	/* find the predecessor of "cnext" in the list "cycle" */
	control cpred = control_undefined;

	MAP(CONTROL, ctmp, {
	  if(gen_in_list_p(ctmp, cycle)) {
	    cpred = ctmp;
	    break;
	  }
	}, control_predecessors(cnext));
	if(control_undefined_p(cpred)) {
	  list cpreds = gen_copy_seq(control_predecessors(cnext));

	  /* This happens in AXIAL from COCCINELLE because this routine is
             called with the tail of a scc from
             subcycle_to_postcondition(). The dropped head may be the
             unique predecessor of some other nodes in cycle. */
	  ifdebug(5) {
	    pips_debug(5, "No explicit predecessor for node %s",
		       statement_identification(control_statement(cnext)));
	    pips_debug(5, "in set:\n");
	    print_control_nodes(cycle);
	  }
	  /* cpred must be a predecessor of the cycle's head as well */
	  gen_list_and(&cpreds, control_predecessors(CONTROL(CAR(cycle))));
	  if(gen_length(cpreds)==1) {
	    cpred = CONTROL(CAR(cpreds));
	  }
	  pips_assert("A predecessor has been found", !control_undefined_p(cpred));
	  gen_free_list(cpreds);
	}
	post = load_predecessor_postcondition(cpred, cnext, control_postcondition_map);
      }
    }
  }

  pips_debug(5, "End\n");
}

/* Assume that subcycle is a simple cycle with no inner cycles and that h
   does not belong to other subcycles not containing the current entry
   node... (FI: to be improved!) */
static void subcycle_to_postconditions(list cycle,
				       control h,
				       statement_mapping fp_map,
				       list secondary_entries,
				       statement_mapping control_postcondition_map)
{
  list new_secondary_entries = NIL;
  list subcycle = head_to_subcycle(cycle, h, &new_secondary_entries);
  list path = control_set_to_control_cycle(subcycle);
  list tail = list_undefined;
  transformer pre = transformer_undefined;

  pips_debug(5, "Begin for head %s", 
	     statement_identification(control_statement(h)));

  if(!ENDP(path)) {
    /* there is a cyclic path to cover the subcycle */
    gen_free_list(subcycle);
    subcycle = path;

    ifdebug(5) {
      pips_debug(5, "Subcycle is:\n");
      print_control_nodes(subcycle);
    }
  }
  else {
    /* there is no cyclic path covering all nodes in subcycle */
    gen_remove(&subcycle, h);
    subcycle = CONS(CONTROL, h, subcycle);
    path = control_set_to_control_list(subcycle);
    gen_free_list(subcycle);
    subcycle = path;
     ifdebug(5) {
       pips_debug(5, "set is:\n");
       print_control_nodes(subcycle);
     }
  }

  pips_assert("subcycle or subset is defined and starts with h", subcycle!=list_undefined
	      && CONTROL(CAR(subcycle))==h);

  /* There are subcycles with a unique node. See INJALL in wave5... */
  if(!ENDP(tail = CDR(subcycle))) {
    pre = load_predecessor_postcondition(h, CONTROL(CAR(tail)), control_postcondition_map);
    pips_assert("As h is part of the main cycle, its precondition must has been defined",
		!transformer_undefined_p(pre));
    cycle_to_postconditions(tail, pre, fp_map, secondary_entries, control_postcondition_map);
  }

  gen_free_list(subcycle);

  pips_debug(5, "End\n");
}

/* Returns secondary entries in their order of appearance in cycle */
static list sort_secondary_entries(list secondary_entries, list cycle)
{
  list ordered_secondary_entries = NIL;

  MAP(CONTROL, c, {
    if(gen_in_list_p(c, secondary_entries)) {
      ordered_secondary_entries = gen_nconc(ordered_secondary_entries,
					    CONS(CONTROL,c, NIL));
    }
  }, cycle);

  ifdebug(5) {
    pips_debug(5, "Seccondary entries:\n");
    print_control_nodes(secondary_entries);
    pips_debug(5, "Ordered secondary entries:\n");
    print_control_nodes(ordered_secondary_entries);
  }

  return ordered_secondary_entries;
}


/* Accumulate postconditions for one global cycle in the scc. A global
   cycle starts with the first node in scc and ends with it. Preconditions
   are propagated downwards in control statements later, when all scc
   entries and cycles have been processed. */
static void process_cycle_in_scc(list scc, list cycle, transformer pre_entry,
				 unstructured u, list secondary_entries,
				 statement_mapping control_postcondition_map)
{
  transformer tfc = transformer_undefined;
  transformer tfp = transformer_identity();
  /* transformer post = transformer_undefined; */
  /* transformer pre = transformer_undefined; */
  transformer fwd_pre = transformer_undefined;
  transformer back_pre = transformer_undefined;
  transformer generic_pre = transformer_undefined;
  transformer real_pre = transformer_undefined;
  statement_mapping statement_to_subcycle_fix_point_map = MAKE_STATEMENT_MAPPING();
  list ordered_secondary_entries = list_undefined;

  pips_debug(5, "Begin for cycle:\n");
  ifdebug(5) print_control_nodes(cycle);

  /* ordered_secondary_entries = sort_secondary_entries(secondary_entries, cycle); */
  /* ordered_secondary_entries = gen_copy_list(secondary_entries); */
  /* ordered_secondary_entries = gen_copy_seq(secondary_entries); */
  ordered_secondary_entries = secondary_entries;

  /* compute a fix point for each sub-cycle head (i.e. secondary entry) */
  MAPL(hc, {
    control h = CONTROL(CAR(hc));
    statement s = control_statement(h);

    if(gen_in_list_p(h, cycle)) {
      /* Why? It could appear on different paths for one entry or for
         different entries... As long as preconditions are not used to
         refine fix points, it does not matter. */      

      /* pips_assert("The subcycle fix point transformer is not defined yet", 
		  hash_get((hash_table) statement_to_subcycle_fix_point_map, (char *) s)
		  == HASH_UNDEFINED_VALUE); */

      if(hash_get((hash_table) statement_to_subcycle_fix_point_map, (char *) s)
	 == HASH_UNDEFINED_VALUE) {
	transformer fptf = subcycle_to_fixpoint(scc, h,
						control_postcondition_map,
						ordered_secondary_entries,
						statement_to_subcycle_fix_point_map);

	ifdebug(5) {
	  pips_debug(5, "Fix point transformer for cycle starting with %s",
		     statement_identification(control_statement(h)));
	}

	pips_assert("The fix point transformer to insert is consistent", 
		    transformer_consistency_p(fptf));

	hash_put((hash_table) statement_to_subcycle_fix_point_map,
		 (char *) s, (char *) fptf);
      }
      else {
	pips_user_warning("subcycle fix point transformer could be recomputed for node %s",
			  statement_identification(s));
      }
    }
  }, ordered_secondary_entries);

  /* compute transformer along path, as in a sequence but with test
     conditions added, without ignoring subcycles */
  MAPL(cc, {
    control c = CONTROL(CAR(cc));
    statement s = control_statement(c);
    transformer tfs = load_statement_transformer(s);

    /* if the node is the head of a cycle, apply its fix point */
    if(gen_in_list_p(c, ordered_secondary_entries)) {
      transformer sctf = (transformer)
	hash_get((hash_table)statement_to_subcycle_fix_point_map ,
		 (char *) s);

      pips_assert("", sctf != (transformer) HASH_UNDEFINED_VALUE)

      tfp = transformer_combine(tfp, sctf);
    }

    pips_assert("tfp is consistent", transformer_consistency_p(tfp));
    pips_assert("tfs is consistent", transformer_consistency_p(tfs));
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
    pips_assert("Precondition fwd_pre is consistent",
		transformer_consistency_p(fwd_pre));
  }

  /* process forward edges first */
  MAP(CONTROL, p,{
    transformer new_pre = transformer_undefined;
    transformer prep = transformer_undefined;

    if(p!=gen_find_eq(p, scc)) {
      /* entry edge: postcondition of predecessor is available */
      prep = load_predecessor_postcondition(p, CONTROL(CAR(cycle)), 
					    control_postcondition_map);
      new_pre = transformer_convex_hull(fwd_pre, prep);

      free_transformer(fwd_pre);
      free_transformer(prep);
      fwd_pre = new_pre;

      ifdebug(5) {
	pips_debug(5, "Precondition fwd_pre using predecessor %s",
		   statement_identification(control_statement(p)));
	print_transformer(fwd_pre);
	pips_assert("Current precondition fwd_pre is consistent",
		transformer_consistency_p(fwd_pre));
      }

    }
  }, control_predecessors(CONTROL(CAR(cycle))));

  ifdebug(5) {
    pips_debug(5, "Precondition fwd_pre:\n");
    print_transformer(fwd_pre);
    pips_assert("Precondition fwd_pre is consistent",
		transformer_consistency_p(fwd_pre));
  }

  /* Apply simple fix-point to approximate the entry precondition for this
     cycle. Although generic_pre is OK because the fix point operator is f^*,
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

  /* post = real_pre;*/

  /* propagate the real_pre precondition down the cycle */
  cycle_to_postconditions(cycle, real_pre,
			  statement_to_subcycle_fix_point_map,
			  ordered_secondary_entries, control_postcondition_map);

  /* propagate preconditions in sub-cycles */
  MAPL(hc, {
    control h = CONTROL(CAR(hc));

    if(gen_in_list_p(h, cycle)) {
      subcycle_to_postconditions(cycle, h,
				 statement_to_subcycle_fix_point_map,
				 ordered_secondary_entries,
				 control_postcondition_map);
    }
  }, ordered_secondary_entries);

  /* FI: I guess that all entries should be freed too! */
  FREE_STATEMENT_MAPPING(statement_to_subcycle_fix_point_map);
  statement_to_subcycle_fix_point_map = NULL;

  /* gen_free_list(ordered_secondary_entries); */

  pips_debug(5, "End\n");
}

static void add_control_to_cycle(list scc, list cycle, control succ,
				 transformer pre_entry, unstructured u,
				 bool process_it, list * psecondary_entries,
				 statement_mapping control_postcondition_map)
{
  static void build_control_cycles_in_scc(list, list, transformer,
					  unstructured, bool, list *,
					  statement_mapping);
  control e = CONTROL(CAR(cycle));

  ifdebug(6) {
    pips_debug(6, "Begin with cycle:\n");
    print_control_nodes(cycle);
  }

  if(succ==e) {
    if(process_it) process_cycle_in_scc(scc, cycle, pre_entry, u, *psecondary_entries,
					control_postcondition_map);
  }
  else {
    if(succ==gen_find_eq(succ, cycle)) {
      /* we have found a sub-cycle in a cycle :-( */
      ifdebug(6) {
	debug(6,"Sub-cycle starts at: %s",
	      statement_identification(control_statement(succ)));
	print_control_nodes(cycle);
      }
      /* assert that succ has been identified as needing a fix point */
      /* pips_error("add_control_to_cycle", "Inner cycle: not implemented yet!"); */
      if(!process_it) {
	/* add succ to secondary entry list */
	if(succ!=gen_find_eq(succ, *psecondary_entries)) {
	  pips_user_warning("Have you performed a full control restructuration?\n"
			    "New cycle head is %s",
			    statement_identification(control_statement(succ)));
	  /* *psecondary_entries = gen_cons(succ, *psecondary_entries); */
	  *psecondary_entries = CONS(CONTROL, succ, *psecondary_entries);
	}
      }
    }
    else {
      cycle = gen_append(cycle, CONS(CONTROL, succ, NIL));
      build_control_cycles_in_scc(scc, cycle, pre_entry, u,
				  process_it, psecondary_entries,
				  control_postcondition_map);
      /* restore the current cycle value */
      gen_remove(&cycle, succ);
    }
  }

  ifdebug(6) {
    pips_debug(6, "End with same cycle:\n");
    print_control_nodes(cycle);
  }
}

/* try to complete cycle and process it neglecting subcycles when it is
   completed. If no processing is required, collect heads of subcycles in
   secondary_entries. */
static void build_control_cycles_in_scc(list scc, list cycle,
					transformer pre_entry,
					unstructured u,
					bool process_it,
					list* psecondary_entries,
					statement_mapping control_postcondition_map)
{
  control l = CONTROL(CAR(gen_last(cycle)));
  /* control e = CONTROL(CAR(cycle)); */

  ifdebug(6) {
    pips_debug(6, "Begin with cycle:\n");
    print_control_nodes(cycle);
  }

  pips_assert("The cycle is not empty", !ENDP(cycle));

  if(gen_length(control_successors(l))==1) {
    control succ = CONTROL(CAR(control_successors(l)));

    pips_assert("A unique successor must be in scc", succ==gen_find_eq(succ, scc));
    add_control_to_cycle(scc, cycle, succ, pre_entry, u, process_it, psecondary_entries,
			 control_postcondition_map);
  }
  else if(gen_length(control_successors(l))==2) {
    control succ_t = CONTROL(CAR(control_successors(l)));
    control succ_f = CONTROL(CAR(CDR(control_successors(l))));

    if(succ_t==succ_f) {
      /* they must be in the scc */
      pips_assert("A double unique successor must be in scc",
		  succ_t==gen_find_eq(succ_t, scc));
      add_control_to_cycle(scc, cycle, succ_t, pre_entry, u, process_it, psecondary_entries,
			   control_postcondition_map);
    }
    else{
      pips_assert("At least one successor must be in scc",
		  gen_find_eq(succ_t, scc) || gen_find_eq(succ_f, scc));
      if(succ_t==gen_find_eq(succ_t, scc)) {
	add_control_to_cycle(scc, cycle, succ_t, pre_entry, u, process_it, psecondary_entries,
			     control_postcondition_map);
      }
      if(succ_f==gen_find_eq(succ_f, scc)) {
	add_control_to_cycle(scc, cycle, succ_f, pre_entry, u, process_it, psecondary_entries,
			     control_postcondition_map);
      }
    }
  }
  else {
    pips_error("build_control_cycles_in_scc",
	       "A control node in a scc must have 1 or 2 successors: %d successors\n",
	       gen_length(control_successors(l)));
  }

  ifdebug(6) {
    pips_debug(6, "End with same cycle:\n");
    print_control_nodes(cycle);
  }
}

static void generic_compute_and_propagate_precondition
(control c, transformer pre, statement_mapping control_postcondition_map)
{
  transformer pre_c = transformer_dup(pre);
  transformer post = transformer_undefined;
  statement s = control_statement(c);
  transformer pre_s = load_statement_precondition(s);

  if(transformer_undefined_p(pre_s)) {
    transformer post_c = transformer_undefined;
    transformer old_post_c = transformer_undefined;

    pips_debug(5, "Propagate precondition in statement %s", statement_identification(s));

    MAP(CONTROL, p, {
      transformer pre_p = load_predecessor_postcondition(p, c, control_postcondition_map);

      ifdebug(5) {
	pips_debug(5, "Use postcondition of statement %s for statement %s",
		   statement_identification(control_statement(p)),
		   statement_identification(s));
	print_transformer(pre_p);
      }

      post = transformer_convex_hull(pre_p, pre_c);
      free_transformer(pre_c);
      pre_c = post;
    }, control_predecessors(c));

    ifdebug(5) {
      pips_debug(5, "Precondition propagated:\n");
      print_transformer(pre_c);
    }

    /* compute preconditions in c */
    post_c = statement_to_postcondition(pre_c, s);

    /* post_c should be better than load_control_postcondition(c) because of
       convexity but the walk on unstructured is random via newgen which should
       lead to unpredictable results. May be a forward walk would be better. */
    old_post_c = load_control_postcondition(s, control_postcondition_map);
    free_transformer(old_post_c);
    update_control_postcondition(s, post_c, control_postcondition_map);

    pips_debug(5, "Precondition propagated in statement %s", statement_identification(s));
  }
  else {
    pips_debug(5, "Precondition already available for statement %s",
	       statement_identification(s));
  }
}

static void entry_compute_and_propagate_precondition
(control c, transformer pre, statement_mapping control_postcondition_map)
{
  generic_compute_and_propagate_precondition(c, pre, control_postcondition_map);
}

static void internal_compute_and_propagate_precondition
(control c, statement_mapping control_postcondition_map)
{
  transformer pre = transformer_empty();

  generic_compute_and_propagate_precondition(c, pre, control_postcondition_map);

  free_transformer(pre);
}

/* compute all cycles from e to e in scc as well as their associated
   transformers (too bad for the convex approximation). Beware of internal
   cycles. Compute the precondition for the entry point and propagate it
   along each cycle. Union preconditions between cycles. */
static void process_ready_scc_for_one_entry(list scc,
					    transformer pre_entry,
					    unstructured u,
					    control e,
					    statement_mapping control_postcondition_map)
{
  list cycle = CONS(CONTROL, e, NIL);
  list secondary_entries = NIL;
  /* list check_entries = NIL; */

  ifdebug(4) {
    pips_debug(4, "Begin for scc:\n");
    print_control_nodes(scc);
    pips_debug(4, "with entry point: %s\n",
	  statement_identification(control_statement(e)));
  }

  /* pips_assert("Secondary entries must be undefined", secondary_entries == list_undefined); */
  secondary_entries = NIL;

  build_control_cycles_in_scc(scc, cycle, pre_entry, u, FALSE, &secondary_entries,
			      control_postcondition_map);

  ifdebug(4) {
    pips_debug(4,"Secondary entries: %d nodes\n", gen_length(secondary_entries));
    print_control_nodes(secondary_entries);
  }

  build_control_cycles_in_scc(scc, cycle, pre_entry, u, TRUE, &secondary_entries,
			      control_postcondition_map);

  /* pips_assert("check_entries must be empyt", ENDP(check_entries)); */

  gen_free_list(secondary_entries);
  secondary_entries = list_undefined;

  /* most nodes should have executable preconditions after
     restructuration, but there might be a stop somewhere */
  ifdebug(4) {
    int count = 0;
    pips_debug(4, "Make sure that most nodes have non empty postconditions\n");
    MAP(CONTROL, c, {
      transformer post = load_control_postcondition(control_statement(c), control_postcondition_map);

      pips_debug(4, "Postcondition for node %s",
		 statement_identification(control_statement(c)));
      print_transformer(post);
      if(transformer_empty_p(post)) count++;
    }, scc);
    pips_debug(4, "Number of nodes with empty postconditions: %d\n", count);
  }

  ifdebug(4) {
    pips_debug(4, "End for scc:\n");
    print_control_nodes(scc);
    pips_debug(4, "with entry point: %s\n",
	  statement_identification(control_statement(e)));
  }
}

/* Each entry node can be processed independently and the resulting
preconditions be unioned. For a given entry node, a transformer must be
computed for each cycle broken at the chosen entry node. Internal cycles
must be processed differently, without entry precondition. Preconditions
for each cycle must be unioned over the cycles. */

static void process_ready_scc(list scc,
			      transformer pre_entry,
			      unstructured u,
			      list already_processed,
			      statement_mapping control_postcondition_map,
			      bool postcondition_p)
{
  list entry_nodes = find_scc_entry_nodes(scc, u, already_processed);
  list ordered_scc = list_undefined;

  ifdebug(3) {
    pips_debug(3, "Begin for scc:\n");
    print_control_nodes(scc);
    pips_debug(3, "with entry nodes:\n");
    print_control_nodes(entry_nodes);
  }

  /* initialize the postconditions in the scc with a neutral value without
     propagating the conditions obtained downwards in the related
     statement */
  MAP(CONTROL, c, {
      transformer post = transformer_empty();
      statement stmt = control_statement(c);
      store_control_postcondition(stmt, post, control_postcondition_map);
  }, scc);

  /* Process each entry node */
  MAP(CONTROL, e, {
    process_ready_scc_for_one_entry(scc, pre_entry, u, e, control_postcondition_map);
  }, entry_nodes);

  /* Follow control paths as much as possible to improve later
     precondition propagation. This is useful when convexity hurts. */
  if(!gen_in_list_p(CONTROL(CAR(scc)), entry_nodes)) {
    control f = CONTROL(CAR(scc));
    gen_remove(&scc, f);
    scc = CONS(CONTROL, f, scc);
  }
  ordered_scc = control_set_to_control_list(scc);
  /* ordered_scc = gen_copy_seq(scc); */

  /* Display results obtained for all entries and all paths for each entry */
  ifdebug(3) {
    pips_debug(3, "Postconditions obtained for control nodes:\n");
    MAP(CONTROL, c, {
      print_control_node(c);
      print_transformer(load_control_postcondition(control_statement(c),
						   control_postcondition_map));
    }, ordered_scc);
  }

  if(postcondition_p) {
    /* Propagate the preconditions downwards in the underlying statements */
    if(unstructured_control(u)==gen_find_eq(unstructured_control(u), scc)) {
      entry_compute_and_propagate_precondition(unstructured_control(u), pre_entry,
					       control_postcondition_map);
    }
    MAP(CONTROL, c, {
      internal_compute_and_propagate_precondition(c, control_postcondition_map);
    }, ordered_scc);
  }

  gen_free_list(ordered_scc);

  pips_debug(3, "End\n");
}

  
static void local_process_unreachable_node(control c, struct  { 
    bool pcond;
    statement_mapping smap;
} * pcontext)
{
  process_unreachable_node(c, pcontext->smap, pcontext->pcond);
}
  
static void node_to_path_transformer_or_postcondition(control c, struct  { 
    bool pcond;
    statement_mapping smap;
} * pcontext)
{
  bool postcondition_p = pcontext->pcond;
  statement_mapping control_postcondition_map = pcontext->smap;
  statement s = control_statement(c);
  transformer tf = (transformer)
    hash_get((hash_table) control_postcondition_map,
	     (char *) s);

  fprintf(stderr, "Statement %s", statement_identification(s));
  pips_assert("Transformer or postcondition is consistent",
	      transformer_consistency_p(tf));
  print_transformer(tf);
}

/* compute either the postconditions in an unstructured or the transformer
   of this unstructured. In both cases, transformers for all nodes are
   supposed to be available. */
transformer unstructured_to_accurate_postconditions_or_transformer
(transformer pre_u, transformer pre, unstructured u, bool postcondition_p)
{
  transformer post = transformer_undefined;
  list to_be_processed = NIL; /* forward reachable nodes in u */
  list still_to_be_processed = NIL;
  list already_processed = NIL;
  list linked_nodes = NIL; /* all nodes in unstructured u */
  statement_mapping control_postcondition_map = make_control_postcondition();
  
  struct  { 
    bool pcond;
    statement_mapping smap;
  } context = { postcondition_p, control_postcondition_map };

  ifdebug(2) {
    pips_debug(2, "Begin for %s for nodes:\n",
	       postcondition_p? "postconditions" : "transformer");
    /* Do not go down into nested unstructured */
    gen_multi_recurse(u, statement_domain, gen_false, gen_null,
		      control_domain, gen_true, print_control_node, NULL);
    pips_debug(2, "With entry nodes\n");
    print_control_node(unstructured_control(u));
    pips_debug(2, "And exit node\n");
    print_control_node(unstructured_exit(u));
  }

  
  /* wide_forward_control_map_get_blocs(unstructured_control(u), &to_be_processed); */
  forward_control_map_get_blocs(unstructured_control(u), &to_be_processed);
  still_to_be_processed = gen_copy_seq(to_be_processed);

  pips_assert("Node lists are defined", !list_undefined_p(to_be_processed)
	      && !list_undefined_p(still_to_be_processed) && ENDP(already_processed) );

  make_control_postcondition();

  while(!ENDP(still_to_be_processed)) {
    int count = -1;
    do {
      list l = list_undefined;

      /* process forward */
      pips_debug(5, "Try forward processing for\n");
      ifdebug(2) print_control_nodes(still_to_be_processed);

      count = 0;
      for(l=still_to_be_processed; !ENDP(l); ) {
	control c = CONTROL(CAR(l));
	POP(l); /* right away because c's cdr might be modified */
	if(ready_to_be_processed_p(c, to_be_processed,
				   still_to_be_processed,
				   already_processed,
				   control_postcondition_map)) {
	  process_ready_node(c, pre, u, control_postcondition_map, postcondition_p);
	  gen_remove(&still_to_be_processed, c);
	  already_processed = gen_append(already_processed, CONS(CONTROL, c, NIL));
	  count++;
	}
      }
    } while(count!=0);
    if(!ENDP(still_to_be_processed)) {
      list scc = list_undefined;
      /* find a scc and process it */
      pips_debug(2, "Find a scc and process it\n");
      scc = find_ready_scc_in_cfg(u, to_be_processed,
				  still_to_be_processed,
				  already_processed,
				  control_postcondition_map);
      pips_assert("scc is defined\n", scc!=list_undefined);
      pips_assert("scc is not empty\n", !ENDP(scc));
      pips_debug(2, "scc found:\n");
      ifdebug(2) print_control_nodes(scc);
      process_ready_scc(scc, pre, u, already_processed,
			control_postcondition_map, postcondition_p);
      gen_list_and_not(&still_to_be_processed, scc);
      already_processed = gen_append(already_processed, scc);
    }
  }

  /* Make sure that all control nodes have been processed.  gen_recurse()
   cannot be used with hierarchical unstructured graphs.*/
  /*
  CONTROL_MAP(c, {
    process_unreachable_node(c, control_postcondition_map, postcondition_p);
  }, unstructured_control(u), linked_nodes);
  CONTROL_MAP(c, {
    process_unreachable_node(c, control_postcondition_map, postcondition_p);
  }, unstructured_exit(u), linked_nodes);
  */

  
  gen_context_multi_recurse(
			    u, (void *) & context,
			    statement_domain, gen_false, gen_null,
			    control_domain, gen_true, local_process_unreachable_node,
			    NULL);
  
  ifdebug(2) {
    pips_debug(2, "%s for unstructured\n",
	       postcondition_p? "Postconditions": "Path transformer");
    gen_context_multi_recurse(
			      u, (void *) & context,
			      statement_domain, gen_false, gen_null,
			      control_domain, gen_true,
			      node_to_path_transformer_or_postcondition,
			      NULL);
    pips_debug(2, "End of map\n");
  }

  post = copy_transformer
    (load_control_postcondition(control_statement(unstructured_exit(u)),
				control_postcondition_map));
  control_postcondition_map = free_control_postcondition(control_postcondition_map);

  gen_free_list(to_be_processed);
  to_be_processed = list_undefined;
  gen_free_list(still_to_be_processed);
  still_to_be_processed = list_undefined;
  gen_free_list(already_processed);
  already_processed = list_undefined;

  ifdebug(2) {
    pips_debug(2, "End with unstructured postcondition:\n");
    print_transformer(post);
  }

  return post;
}

static transformer
unstructured_to_postconditions(
    transformer pre,
    transformer pre_first,
    unstructured u)
{
  list nodes = NIL ;
  control entry_node = unstructured_control(u) ;
  control exit_node = unstructured_exit(u) ;
  transformer c_pre = transformer_undefined;
  transformer post = transformer_undefined;
  transformer exit_post = transformer_undefined;

  debug(8,"unstructured_to_postconditions","begin\n");

  /* SHARING! Every statement gets a pointer to the same precondition!
   * I do not know if it's good or not but beware the bugs!!!
   */
  /* FI: changed to make free_transformer_mapping possible without 
   * testing sharing.
   *
   * pre and pre_first can or not be used depending on the
   * unstructured structure. They are always duplicated and
   * the caller has to take care of their de-allocation.
   */
  CONTROL_MAP(c, {
    statement st = control_statement(c) ;
    if(c==entry_node && ENDP(control_predecessors(c)) && statement_test_p(st)) {
      /* special case for the first node if it has no predecessor */
      /* and if it is a test, as it always should, at least if */
      /* unspaghettify has been applied... */
      /* this is pretty useless and should be generalized to the
	 DAG part of the CFG */
      c_pre = transformer_dup(pre_first);
      post = statement_to_postcondition(c_pre, st);
      transformer_free(post);
    }
    else {
      transformer c_pre_m = transformer_undefined;

      c_pre = transformer_dup(pre);
      c_pre_m = c_pre;

      /* refine the precondition if the node has only one
	 predecessor and if this predecessor is a test and if the
	 test can be exploited */
      if(gen_length(control_predecessors(c))==1 && c!=entry_node) {
	control prev_c = CONTROL(CAR(control_predecessors(c)));
	statement prev_st = control_statement(prev_c);

	if(statement_test_p(prev_st)) {
	  /* the condition is TRUE if c is the first successor of prev_c */
	  bool true_false = (c==(CONTROL(CAR(control_successors(prev_c)))));
	  expression e = test_condition(statement_test(prev_st));

	  c_pre_m = precondition_add_condition_information(c_pre, e, true_false);
	  /* If the free is performed, core dump guaranteed on some
             examples: see unclear comments about the previously called
             function:-( */
	  /* free_transformer(c_pre); */
	}
      }

      post = statement_to_postcondition(c_pre_m, st);
      if(c==exit_node) {
	exit_post = post;
      }
      else {
	transformer_free(post);
      }
    }
  }, entry_node, nodes);

  gen_free_list(nodes) ;

  if(transformer_undefined_p(exit_post)) {
    exit_post = transformer_empty();
  }

  ifdebug(8) {
    debug(8,"unstructured_to_postconditions","exit postcondition:\n");
    (void) print_transformer(exit_post) ;
  }
  debug(8,"unstructured_to_postconditions","end\n");

  return exit_post;
}

/* compute pre- and post-conditions in an unstructured from the entry
   precondition pre and return the exit postcondition. pre_u is pre
   filtered by the u's transformer and can be used for any node.  */
transformer unstructured_to_accurate_postconditions
(transformer pre_u, transformer pre, unstructured u)
{
  transformer post = transformer_undefined;
  list succs = NIL;
  control head = unstructured_control(u);
  /* control tail = unstructured_exit(u); */

  forward_control_map_get_blocs(head, &succs);

  if(gen_length(succs)>SEMANTICS_MAX_CFG_SIZE1) {
      pips_user_warning("\nControl flow graph too large for an accurate analysis (%d nodes)\n"
			"Have you fully restructured your code?\n", gen_length(succs));
    post = unstructured_to_postconditions(pre_u, pre, u);
  }
  else {
    post = unstructured_to_accurate_postconditions_or_transformer
      (pre_u, pre, u, TRUE);
  }
  gen_free_list(succs);

  pips_assert("Postcondition for unstructured is consistent",
	      transformer_consistency_p(post));

  return post;
}

static void unreachable_node_to_transformer(control c)
{
  statement s = control_statement(c);

  pips_debug(9,"Statement %s", statement_identification(s));

  if(statement_ordering(s)==STATEMENT_ORDERING_UNDEFINED) {
    pips_user_warning("Improper control restructuration, statement with no ordering:%s",
		      statement_identification(s));
  }

  if(transformer_undefined_p(load_statement_transformer(s))) {
    pips_user_warning("After restructuration, unexpected unreachable node:%s",
		      statement_identification(s));
    (void) statement_to_transformer(s);
  }
}

/* This function is also used when computing preconditions if the exit
   node is not reached. It assumes that transformers for all statements in
   the unstructured have already been computed. */
transformer 
unstructured_to_global_transformer(
    unstructured u)
{
  /* Assume any reachable node is executed at each iteration. A fix-point
     of the result can be used to approximate the node preconditions. Some
     nodes can be discarded because they do not modify the store such as
     IF statements (always) and CONTINUE statements (if they do not link
     the entry and the exit nodes). */
  
  list nodes = NIL;
  /* Entry node */
  control entry_node = unstructured_control(u);
  control exit_node = unstructured_exit(u);
  transformer tf_u = transformer_empty();
  transformer fp_tf_u = transformer_undefined;
  
  pips_debug(8,"begin\n");

  FORWARD_CONTROL_MAP(c, {
    statement st = control_statement(c);
    /* transformer_convex_hull has side effects on its arguments:-( */
    /* Should be fixed now, 29 June 2000 */
    /* transformer tf_st = copy_transformer(load_statement_transformer(st)); */
    transformer tf_st = load_statement_transformer(st);
    transformer tf_old = tf_u;
    
    if(statement_test_p(st)) {
      /* Any side effect? */
      if(!ENDP(transformer_arguments(tf_st))) {
	tf_u = transformer_convex_hull(tf_old, tf_st); /* test */
	free_transformer(tf_old);
      }
    }
    else {
      if(continue_statement_p(st)) {
	if(gen_find_eq(entry_node, control_predecessors(c))!=chunk_undefined
	   && gen_find_eq(exit_node, control_successors(c))!=chunk_undefined) {
	  tf_u = transformer_convex_hull(tf_old, tf_st); /* continue */
	  free_transformer(tf_old);
	}
      }
      else {
	tf_u = transformer_convex_hull(tf_old, tf_st); /* other */
	free_transformer(tf_old);
      }
    }

    ifdebug(1) {
      pips_assert("tf_st is internally consistent",
		  transformer_internal_consistency_p(tf_st));
      pips_assert("tf_u is internally consistent",
		  transformer_internal_consistency_p(tf_u));
    }
    
  }, entry_node, nodes) ;
  
  gen_free_list(nodes) ;
  
  fp_tf_u = (*transformer_fix_point_operator)(tf_u);
  /* fp_tf_u = transformer_derivative_fix_point(tf_u); */
  
  ifdebug(8) {
    pips_debug(8,"Result for one step tf_u:\n");
    print_transformer(tf_u);
    pips_assert("tf_u is internally consistent",
		transformer_internal_consistency_p(tf_u));
    pips_debug(8,"Result for fix-point fp_tf_u:\n");
    print_transformer(fp_tf_u);
    pips_assert("fp_tf_u is internally consistent",
		transformer_internal_consistency_p(fp_tf_u));
  }
  
  pips_debug(8,"end\n");
  
  return fp_tf_u;
}

/* It is assumed that all transformers for nodes in u have already been computed */
transformer unstructured_to_accurate_transformer(unstructured u, list e)
{
  transformer tf = transformer_undefined;
  list succs = NIL;
  control head = unstructured_control(u);
  control tail = unstructured_exit(u);

  forward_control_map_get_blocs(head, &succs);

  if(gen_in_list_p(tail, succs)) {
    /* computing the transformer for u is like computing the postcondition
       with no information on entry */
    transformer pre_u = transformer_identity();
    transformer pre = transformer_identity();

    /* These tests should be performed at the scc level */
    if(gen_length(succs)>SEMANTICS_MAX_CFG_SIZE2) {
      pips_user_warning("\nControl flow graph too large for any analysis (%d nodes)\n"
			"Have you fully restructured your code?\n", gen_length(succs));
      tf = effects_to_transformer(e);
    }
    else if(gen_length(succs)>SEMANTICS_MAX_CFG_SIZE1) {
      pips_user_warning("\nControl flow graph too large for an accurate analysis (%d nodes)\n"
			"Have you fully restructured your code?\n", gen_length(succs));
      tf = unstructured_to_global_transformer(u);
    }
    else {
      tf = unstructured_to_accurate_postconditions_or_transformer
	(pre_u, pre, u, FALSE);
    }
  }
  else {
    /* The unstructured is never exited, but all nodes are supposed to
       have transformers. This should never occur if the control
       restructuration were clean. */
    gen_multi_recurse(
		      u,
		      statement_domain, gen_false, gen_null,
		      control_domain, gen_true, unreachable_node_to_transformer,
		      NULL);
    tf = transformer_empty();
  }

  gen_free_list(succs);

  return tf;
}
