/* semantical analysis
  *
  * Computation of transformers and postconditions and total preconditions
  * in unstructured (i.e. CFG).
  *
  * Accurate results are obtained by restructuring CFG into a set of
  * non-deterministic CFG's using Bourdoncle's partitioning as implemented
  * in control/bourdoncle.c. The cycle heads are chosen using Bourdoncle's
  * heuristics. Then the graph is augmented with new nodes in order that
  * cycle heads are the only entry and exit node of the cycles in each
  * scc. So the input CFG is transformed into a non-deterministic
  * DAG. Some nodes of the DAG point to SCC's. Each SCC is represented by
  * a DAG whose leaves point back to the cycle head. The SCC DAG may
  * contain nodes pointing to sub-SCC's.
  *
  * It is not clear if you want to store the node pre or
  * postcondition. The postcondition may depend on the outgoing arc. But
  * the precondition would have to be processed by the node transformer
  * once for each arc. To avoid both problems the postconditions are
  * stored without the control information. It is added later according to
  * the outgoing arc considered. But it is not satisfactory to apply a
  * reverse transformer to retrieve the precondition or to re-apply a
  * convex hull on the input after having taken care of the potential
  * cycle. All in all, it would be easier to store preconditions...
  *
  * Transformers between the entry point and after the current node are
  * very similar to postcondition between the module entry point and the
  * current node. Transformers are obtained like precondition but without
  * a precondition at the CFG entry point.
  *
  * It is possible to store the star or the plus fix points. I first
  * stored the star fix points. It is often needed. It is safe because it
  * can be applied several times withtout impacting the result. However,
  * it does not retain information about the cycle body
  * transformer. Because the convex hull operator looses information, it
  * is more accurate to use the plus fix point and to apply a convex hull
  * to the direct precondition and to the cycle output. Thus, if the entry
  * node is a test, its conditions can be added before the convex hull is
  * performed.
  *
  * It might be useful to add a pseudo-node as predecessor of the CFG
  * entry node. This pseudo-node would simplify the algorithms and the
  * function profiles. Its post-condition would be the precondition of the
  * CFG or no information. Without it, each node must be checked to see if
  * it is the entry node because, then, it has an extra-predecessor.
  *
  * Francois Irigoin, July 2002 (First version: October 2000)
  *
  * $Id$
  *
  * $Log: unstructured.c,v $
  * Revision 1.13  2002/07/19 12:09:58  irigoin
  * First draft of the new version of unstructured.c working for all
  * Validation/unstrucnn.f with nn in[01-13]. Lots of memory leaks, code
  * replications, not understood behavior, and redundant computations.
  *
  * Ideally, node preconditions and edge preconditions should be memoized
  * (i.e. stored in hash tables) instead of postconditions. But edges do not
  * really exist: I should use their corresponding CONS cells.
  *
  * Revision 1.12  2002/07/15 16:46:25  irigoin
  * Intermediate state. Validation OK for unstruc01.f to unstruc06.f
  *
  * Revision 1.11  2002/07/09 06:52:31  irigoin
  * Comments added for the last version of unstructured.c not using
  * Bourdoncle's partitioning heuristics
  *
  * Revision 1.10  2002/07/03 11:07:19  irigoin
  * check "!get_bool_property("SEMANTICS_ANALYZE_UNSTRUCTURED")" added to
  * circumvent an EDF bug in emergency for Corinne.
  *
  * Revision 1.9  2001/12/05 17:19:38  irigoin
  * First plug to handle total preconditions
  *
  * Revision 1.8  2001/10/22 15:47:27  irigoin
  * New interface to cope with transformers computed using preconditions
  *
  * Revision 1.7  2001/07/19 17:59:56  irigoin
  * Improved debugging messages. Bug fix for Validation/tilt.f and tilt3.f:
  * secondary_entries were not passed down or computed correcty and internal
  * cycles were missed.
  *
  * Revision 1.6  2001/02/07 18:16:17  irigoin
  * Lots of improvement to handle recursion in fixpoint computation. Should
  * still be improved. Subscc should be handled like scc by enumerating paths
  * instead of looking for a unique covering path, regardless of subsubcyles.
  *
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
  * */

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

/*
 * Prettyprinting of control nodes for debugging purposes
 */

static void print_control_node(control c)
{
  fprintf(stderr,
	  "ctr %p, %d preds, %d succs: %s", 
          c,
	  gen_length(control_predecessors(c)),
	  gen_length(control_successors(c)),
	  safe_statement_identification(control_statement(c)));
  fprintf(stderr,"\tsuccessors:\n");
  MAP(CONTROL, s, {
    fprintf(stderr, "\t\t%p %s", s,
	    safe_statement_identification(control_statement(s)));
  }, control_successors(c));
  fprintf(stderr,"\tpredecessors:\n");
  MAP(CONTROL, p, {
    fprintf(stderr, "\t\t%p %s", p,
	    safe_statement_identification(control_statement(p)));
  }, control_predecessors(c));
  fprintf(stderr, "\n");
}

static void print_control_nodes(list l)
{
  if(ENDP(l)) {
    fprintf(stderr, "empty control list");
  }
  else {
    MAP(CONTROL, c, {
      fprintf(stderr, "%x, %s", (unsigned int) c,
	      safe_statement_identification(control_statement(c)));
    }, l);
  }
  fprintf(stderr, "\n");
}

/* 
 * STORAGE AND RETRIEVAL OF INTERMEDIATE POSTCONDITIONS
 *
 * Node postconditions cannot be recomputed several times because
 * preconditions are stored. Since they have to be used several times in
 * an unknown order, it is necessary to store them.
 * 
 * A static variable was initially used which reduced the number of
 * parameters passed but this was not compatible with recursive calls, the
 * embedding of unstructured within unstructured, and so on.
 *  */

static transformer load_control_postcondition(control c,
					      control_mapping control_postcondition_map)
{
  transformer post = (transformer)
    hash_get((hash_table) control_postcondition_map,
	     (void *) c);

  if(post == (transformer) HASH_UNDEFINED_VALUE)
    post = transformer_undefined;

  return post;
}

static transformer get_control_precondition(control c,
					    control_mapping control_postcondition_map,
					    unstructured u,
					    transformer pre_entry)
{
  transformer pre = transformer_undefined;
  list preds = control_predecessors(c);
  static transformer load_arc_precondition(control, control, control_mapping);

  pips_assert("c is meaningfull", !meaningless_control_p(c));

  pips_debug(2, "Begin for node %p and statement %s\n", c,
	     statement_identification(control_statement(c)));

  /* Compute its precondition from the postconditions of its predecessor
     and from the entry precondition. Use arc pred->c information to deal
     with tests. */

  if(c==unstructured_control(u)) {
    /* Do not forget the unstructured precondition for the entry node */
    /* FI: I do not know why it has to be replicated. Probably because the
       statement containing the unstructured and the statement of the
       entry node may share the same precondition. */
    pre = copy_transformer(pre_entry);
  }
  else {
    pre = load_arc_precondition(CONTROL(CAR(preds)), c, control_postcondition_map);
    POP(preds);
  }

  MAP(CONTROL, pred, {
    transformer npre = load_arc_precondition(pred, c, control_postcondition_map);
    transformer lpre = pre;

    pips_assert("The predecessor's postcondition npre is defined",
		!transformer_undefined_p(npre));

    pre = transformer_convex_hull(npre, lpre);
    /* memory leak with lpre. pre_entry and postconditions cannot be
       freed: lpre cannot be freed during the first iteration */
    if(pred!=CONTROL(CAR(preds))) free_transformer(lpre);
    lpre = pre;
  }, preds);

  ifdebug(2) {
    pips_debug(2, "End for node %p and statement %s with precondition %p:\n", c,
	       statement_identification(control_statement(c)), pre);
    print_transformer(pre);
  }

  return pre;
}

static void store_control_postcondition(control c, transformer post,
					control_mapping control_postcondition_map)
{
  statement stat = control_statement(c);

  pips_assert("The statement is defined",
	      !meaningless_control_p(c) && !statement_undefined_p(stat));
  pips_debug(8, "Store postcondition for control %p: %s\n",
	     c, statement_identification(stat));
  ifdebug(8) {
    print_transformer(post);
  }

  pips_assert("The postcondition to insert is consistent", 
	      transformer_consistency_p(post));

  pips_assert("The postcondition is not defined yet", 
	      hash_get((hash_table) control_postcondition_map, (void *) c)
	      == HASH_UNDEFINED_VALUE);
  hash_put((hash_table) (control_postcondition_map),
	   (void *) c, (void *) post);
}

static void update_control_postcondition(control c, transformer post,
					 control_mapping control_postcondition_map)
{
  statement stat = control_statement(c);

  pips_assert("The statement is defined",
	      !meaningless_control_p(c) && !statement_undefined_p(stat));
  pips_debug(8, "Update postcondition for control %s\n",
	     statement_identification(stat));
  ifdebug(8) {
    print_transformer(post);
  }

  pips_assert("The postcondition is already defined", 
	      hash_get((hash_table) control_postcondition_map, (void *) c)
	      != HASH_UNDEFINED_VALUE);
  hash_update((hash_table) (control_postcondition_map),
	      (void *)c, (void *) post);
}

void print_control_postcondition_map
(control_mapping control_postcondition_map)
{
  if(hash_table_entry_count(control_postcondition_map)>0) {
    HASH_MAP(c, p, {
      statement s = control_statement((control) c);
      fprintf(stderr, "Statement %s, Temporary postcondition:\n",
	      statement_identification(s));
      print_transformer((transformer) p);
    }, control_postcondition_map);
  }
  else {
    pips_assert("The CFG contains at least one node with one statement",
		FALSE);
  }
}

static control_mapping make_control_postcondition_map()
{
  control_mapping control_postcondition_map = NULL;
  control_postcondition_map = MAKE_CONTROL_MAPPING();
  return control_postcondition_map;
}

static control_mapping free_control_postcondition_map(control_mapping control_postcondition_map)
{
  /* all postconditions must be freed */
  HASH_MAP(k,v,{
    free_transformer((transformer) v);
  }, control_postcondition_map);

  FREE_CONTROL_MAPPING(control_postcondition_map);

  /* return control_mapping_undefined; */
  return NULL;
}

/* 
 * STORAGE AND RETRIEVAL OF FIX POINT TRANSFORMERS
 *
 *  */

static transformer load_control_fix_point(control c,
					  control_mapping control_fix_point_map,
					  hash_table ancestor_map)
{
  control c_a = control_to_ancestor(c, ancestor_map);
  transformer fptf = (transformer)
    hash_get((hash_table) control_fix_point_map,
	     (char *) c_a);

  if(fptf == (transformer) HASH_UNDEFINED_VALUE)
    fptf = transformer_undefined;

  return fptf;
}

static void store_control_fix_point(control c, transformer fptf,
					control_mapping control_fix_point_map)
{
  statement stat = control_statement(c);

  pips_assert("The statement is defined",
	      !meaningless_control_p(c) && !statement_undefined_p(stat));
  pips_debug(8, "Store fix_point for control %p: %s\n",
	     c, statement_identification(stat));
  ifdebug(8) {
    print_transformer(fptf);
  }

  pips_assert("The fix_point to insert is consistent", 
	      transformer_consistency_p(fptf));

  pips_assert("The fix_point is not defined yet", 
	      hash_get((hash_table) control_fix_point_map, (void *) c)
	      == HASH_UNDEFINED_VALUE);
  hash_put((hash_table) (control_fix_point_map),
	   (void *)c, (void *) fptf);
}

static void update_control_fix_point(control c, transformer fptf,
					 control_mapping control_fix_point_map)
{
  statement stat = control_statement(c);

  pips_assert("The statement is defined",
	      !meaningless_control_p(c) && !statement_undefined_p(stat));
  pips_debug(8, "Update fix point for control %s\n",
	     statement_identification(stat));
  ifdebug(8) {
    print_transformer(fptf);
  }

  pips_assert("The fix point is already defined", 
	      hash_get((hash_table) control_fix_point_map, (void *) c)
	      != HASH_UNDEFINED_VALUE);
  hash_update((hash_table) (control_fix_point_map),
	      (void *)c, (void *) fptf);
}

static control_mapping make_control_fix_point_map()
{
  control_mapping control_fix_point_map = NULL;
  control_fix_point_map = MAKE_CONTROL_MAPPING();
  return control_fix_point_map;
}

static control_mapping free_control_fix_point_map(control_mapping control_fix_point_map)
{
  /* all fix_points must be freed */
  HASH_MAP(k,v,{
    free_transformer((transformer) v);
  }, control_fix_point_map);

  FREE_CONTROL_MAPPING(control_fix_point_map);

  /* return control_mapping_undefined; */
  return NULL;
}

/*
 * SCHEDULING OF THE TRANSFORMER OR POSTCONDITION COMPUTATIONS ON A DAG
 *
 */

/* a control can be considered already processed if it has been processed
   already or if it can receive a trivial empty postcondition */
static bool considered_already_processed_p(control c, list to_be_processed,
					   list already_processed,
					   control_mapping control_postcondition_map)
{
  bool already = gen_in_list_p(c, already_processed);

  if(!already) {
    if(!gen_in_list_p(c, to_be_processed)) {
      /* unreachable node with empty precondition */
      /* transformer pre = transformer_empty(); */
      transformer post = load_control_postcondition(c, control_postcondition_map);

      if(transformer_undefined_p(post)) {
	post = transformer_empty();
	/* transformer post = statement_to_postcondition(pre, stmt); */
	store_control_postcondition(c, post, control_postcondition_map);
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
						 control_mapping control_postcondition_map)
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
				    control_mapping control_postcondition_map)
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
      transformer post = load_control_postcondition(c, control_postcondition_map);

      if(transformer_undefined_p(post)) {
	post = transformer_empty();
	/* transformer post = statement_to_postcondition(pre, stmt); */
	store_control_postcondition(c, post, control_postcondition_map);
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
				    control_mapping control_postcondition_map)
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

/*
 * MANAGEMENT OF ARC POSTCONDITIONS: TEST CONDITIONS MUST BE TAKEN INTO ACCOUNT
 *
 *  - load_predecessor_postcondition_or_test_condition()
 *
 *  - load_predecessor_postcondition()
 *
 *  - load_predecessor_test_condition()
 *
 * */

/* A new postcondition may have to be allocated because a unique control
   may have two successors and hence two preconditions. To keep memory
   management tractable, a new postcondition always is allocated.

   This assumes a deterministic CFG (unstructured). See
   load_arc_precondition() to deal with non-deterministic CFG's.  */
static transformer load_predecessor_postcondition_or_test_condition
(control pred,
 control c,
 bool postcondition_p,
 control_mapping control_postcondition_map)
{
  statement stmt = control_statement(pred);
  transformer post = postcondition_p? 
    transformer_dup(load_control_postcondition(pred, control_postcondition_map))
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
      post = precondition_add_condition_information(post, e,
						    transformer_undefined, TRUE);
    }
  }
  else {
    /* Must be the false branch of a test */
    expression e = test_condition(statement_test(stmt));
    post = precondition_add_condition_information(post, e,
						  transformer_undefined, FALSE);
  }

  return post;
}

static transformer load_predecessor_postcondition(control pred, control c,
 control_mapping control_postcondition_map)
{
  transformer post = load_predecessor_postcondition_or_test_condition
    (pred, c, TRUE, control_postcondition_map);
  return post;
}

static transformer load_predecessor_test_condition(control pred, control c,
 control_mapping control_postcondition_map)
{
  transformer post = load_predecessor_postcondition_or_test_condition
    (pred, c, FALSE, control_postcondition_map);
  return post;
}

/* Returns the precondition of c associated to arc pred->c in a newly
   allocated transformer. */
static transformer load_arc_precondition(control pred, control c,
 control_mapping control_postcondition_map)
{
  transformer pre =
    transformer_dup(load_control_postcondition(pred, control_postcondition_map));
  int i = gen_position(c, control_successors(pred));
  int noo = gen_occurences(c, control_successors(pred));

  pips_assert("c is a successor of pred", i!=0);
  /* let's assume that Bourdoncle's restructuring does not clutter the
     successor list too much. */
  pips_assert("c does not appear more than twice in the successor list",
	      noo<=2);

  if(control_test_p(pred)) {
    statement stmt = control_statement(pred);

    pips_assert("stmt is a test", statement_test_p(stmt));

    if(noo==2) {
      /* Assume that the same node is in the true and false successor lists. */
      /* Do not bother with the test condition... and FI loose the side effects. */
      /* If side effects can be detected, perform a convex hull of the
	 true and false branches as nothing else is available. */
      ;
    }
    else {
      /* add the test condition */
      expression e = test_condition(statement_test(stmt));

      if(i%2==1) { /* One of the true successors */
	pre = precondition_add_condition_information(pre, e,
						     transformer_undefined, TRUE);
      
      }
      else{ /* One of the false successors */
	pre = precondition_add_condition_information(pre, e,
						     transformer_undefined, FALSE);
      }
    }
  }

  ifdebug(2) {
    string msg = control_test_p(pred)? (i%2==1? "true" : "false"): "standard";
    pips_debug(2, "End for %s arc of position %d between predecessor node pred=%p"
	       " and node c=%p with precondition %p:\n",
	       msg, i, pred, c, pre);
    print_transformer(pre);
  }

  return pre;
}

/*
 * Handle the fix_point_map
 *
 *  - load_cycle_fix_point(c, fix_point_map)
 *
 *  - 
 *
 * */

transformer load_cycle_fix_point(control c, hash_table fix_point_map)
{
  transformer fptf = transformer_undefined;

  if((fptf = (transformer) hash_get(fix_point_map, (void *) c))
     == (transformer) (HASH_UNDEFINED_VALUE)) 
    pips_assert("c is not a cycle head", FALSE);

  return fptf;
}

/*
 * Perform a convex hull of the postconditions of the predecessors and
 * compute the node transformer even if no predecessors exist and store
 * the postcondition if required:
 *
 *  - process_ready_node()
 *
 *  - process_unreachable_node()
 *
 * */

static void process_ready_node(control c,
			       transformer pre_entry,
			       transformer n_pre,
			       unstructured u,
			       control_mapping control_postcondition_map,
			       hash_table ancestor_map,
			       hash_table scc_map,
			       list partition,
			       control_mapping fix_point_map,
			       bool postcondition_p)
{
  transformer post = transformer_undefined;
  statement stmt = control_statement(c);
  list preds = control_predecessors(c);
  transformer pre = transformer_undefined;
  
  ifdebug(2) {
    pips_debug(2, "Begin with pre_entry=%p:\n", pre_entry);
    print_transformer(pre_entry);
    pips_debug(5, "to process node %s\n", statement_identification(control_statement(c))); 
  }

  /* Compute its precondition pre from the postconditions of its predecessor
     and from the entry precondition. Use arc pred->c information to deal
     with tests. */

  if(c==unstructured_control(u)) {
    /* Do not forget the unstructured precondition for the entry node */
    /* FI: I do not know why it has to be replicated. Probably because the
       statement containing the unstructured and the statement of the
       entry node may share the same precondition. */
    pre = copy_transformer(pre_entry);
  }
  else {
    pre = load_arc_precondition(CONTROL(CAR(preds)), c, control_postcondition_map);
    POP(preds);
  }

  MAP(CONTROL, pred, {
    transformer npre = load_arc_precondition(pred, c, control_postcondition_map);
    transformer lpre = pre;

    pre = transformer_convex_hull(npre, lpre);
    /* memory leak with lpre. pre_entry and postconditions cannot be
       freed: lpre cannot be freed during the first iteration */
    /* if(pred!=CONTROL(CAR(preds))) free_transformer(lpre); */
    free_transformer(lpre);
    free_transformer(npre);
    lpre = pre;
  }, preds);

  /* If the control is a cycle head, find and apply its fix point
     tranformer to the precondition before proceeding into the node itself */

  if(cycle_head_p(c, ancestor_map, scc_map)) {
    transformer fptf = load_control_fix_point(c, fix_point_map, ancestor_map);
    transformer pre_cycle = transformer_undefined;
    transformer pre_no_cycle = transformer_undefined;

    if(transformer_undefined_p(fptf)) {
      unstructured scc_u = cycle_head_to_scc(c, ancestor_map, scc_map);
      fptf = cycle_to_flow_sensitive_postconditions_or_transformers
	(partition, /* Bourdoncle's processing ordering */
	 scc_u, /* A non-deterministic acyclic control flow graph */
	 ancestor_map, /* Mapping from non-deterministic nodes to deterministic nodes */
	 scc_map, /* mapping from deterministic nodes to non-deterministic cycles */
	 fix_point_map,
	 pre, /* precondition on entry */
	 n_pre, /* precondition true for any node. */
	 control_postcondition_map, /* */
	 FALSE); /* Compute transformers */
    }
    /* The cycle is entered */
    pre_cycle = transformer_apply(fptf, pre);
    /* Or the cycle is not entered and the store is not in fptf's domain. */
      pre_no_cycle = transformer_dup(pre);
    if(control_test_p(c)) {
      /* Have we followed a true or a false branch? We'll know later but
         it is better to add this information before the convex hull is
         performed. */
      statement stmt = control_statement(c);
      expression e = test_condition(statement_test(stmt));

      /* FI: I do not know why an empty context is passed down. */
      if(true_successors_only_p(c)) {
	pre_no_cycle = precondition_add_condition_information(pre_no_cycle, e,
							      transformer_undefined, TRUE);
      }
      else if(false_successors_only_p(c)) {
	pre_no_cycle = precondition_add_condition_information(pre_no_cycle, e,
							      transformer_undefined, FALSE);
      }
    }
    free_transformer(pre);
    pre = transformer_convex_hull(pre_cycle, pre_no_cycle);
    free_transformer(pre_cycle);
    free_transformer(pre_no_cycle);
  }

  /* Propagate the precondition thru the node to obtain a postcondition
     which does not include arc information (FI: hard if side-effects?) */

  if(postcondition_p) {
    /* FI: The transformer might have to be recomputed before the call if
       the option is selected? It does not seem to be located in
       statement_to_postcondition() */
    /* FI: is this correct when the statement is a test since the node
       only exploits the condition? It might be because of the convex hull
       nullifying the condition and hence the arc information. */
    /* FI: statement_to_postcondition() cannot always be used because the
       statement may appear in different non-deterministic nodes; pre is
       only of of the many preconditions that can hold before stmt is
       executed; statement_to_postcondition() should be called later when
       the whole unstructured has been analyzed. */
    /* post = statement_to_postcondition(pre, stmt); */
    transformer tf = load_statement_transformer(stmt);

    pips_assert("The statement transformer is defined",
		!transformer_undefined_p(tf));
    post = transformer_apply(tf, pre);
  }
  else {
    /* The statement transformer may have been computed earlier thru a fix
       point calculation. */
    transformer tf = load_statement_transformer(stmt);

    if(transformer_undefined_p(tf)) {
      tf = statement_to_transformer(stmt, pre);
    }
    post = transformer_apply(tf, pre);
  }

  store_control_postcondition(c, post, control_postcondition_map);
}

static bool process_unreachable_node(control c,
				     control_mapping control_postcondition_map,
				     bool postcondition_p)
{
  statement s = control_statement(c);

  pips_debug(9,"Statement %s", safe_statement_identification(s));

  if(!meaningless_control_p(c)) {
    if(statement_ordering(s)==STATEMENT_ORDERING_UNDEFINED) {
      pips_user_warning("Improper control restructuration, statement with no ordering:%s",
			statement_identification(s));
    }

    if(transformer_undefined_p
       (load_control_postcondition(c, control_postcondition_map))) {
      transformer post = transformer_undefined;

      if(postcondition_p) {
	transformer pre = transformer_empty();
	post = statement_to_postcondition(pre, s);
      }
      else {
	/* Careful, this may have been done earlier by the CONTROL_MAP in
	   unstructured_to_transformers() */
	if(transformer_undefined_p(load_statement_transformer(s))) {
	  /* FI: We should use the generic node precondition n_pre; it
	     should be an additional component in the context. */
	  (void) statement_to_transformer(s, transformer_undefined);
	}
	post = transformer_empty();
      }

      pips_user_warning("After restructuration, unexpected unreachable node:%s",
			statement_identification(s));

      store_control_postcondition(c, post, control_postcondition_map);
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
		  transformer_empty_p(load_control_postcondition(c, control_postcondition_map)));

      pips_user_warning("After restructuration (?),"
			" postcondtion for unexpected unreachable node:%s",
			statement_identification(s));
    }
    else if(!postcondition_p
	    && transformer_undefined_p(load_statement_transformer(s))) {
      /* Problem with SHALLOW in SWIM */
      (void) statement_to_transformer(s, transformer_undefined);

      pips_user_warning("After restructuration (?),"
			" transformer for unexpected unreachable node:%s",
			statement_identification(s));
    }
  }
  return TRUE;
}

/*
 * DERIVE STATEMENT PRECONDITIONS
 *
 */

/* State that d depends on c */
static void add_cycle_dependency(control d, control c, control_mapping cycle_dependencies_map)
{
  list dependencies = (list) hash_get((hash_table) cycle_dependencies_map,
				      (void *) d);

  if(dependencies == (list) HASH_UNDEFINED_VALUE) {
    dependencies = CONS(CONTROL, c, NIL);
    hash_put((hash_table) (cycle_dependencies_map),
	   (void *) d, (void *) dependencies);
  }
  else {
    /* Update the hash_table by side effect */
    dependencies = gen_nconc(dependencies, CONS(CONTROL, c, NIL));
  }
}

/* c has been processed and no other cycle can depend on it */
static void remove_cycle_dependencies(control c, control_mapping cycle_dependencies_map)
{
  HASH_MAP(d, dependencies, {
    /* Oops... gen_remove() might change the pointer */
    gen_remove((list*) (&dependencies), c);
    hash_update((hash_table) (cycle_dependencies_map),
	   (void *) d, (void *) dependencies);
  }, cycle_dependencies_map);
}

/* Some cycle may appear nowhere but in the DAG and have no entries in the
   table. In that case, they have an empty dependency list. */
static list get_cycle_dependencies(control d, control_mapping cycle_dependencies_map)
{
  list dependencies = (list) hash_get((hash_table) cycle_dependencies_map,
				      (void *) d);

  if(dependencies == (list) HASH_UNDEFINED_VALUE) {
    dependencies = NIL;
  }

  return dependencies;
}

static void free_cycle_dependencies(control_mapping cycle_dependencies_map) {
  HASH_MAP(d, dependencies, {
    /* All dependency lists should be empty after full processing */
    pips_assert("No dependencies are left", ENDP(dependencies));
  }, cycle_dependencies_map);

  FREE_CONTROL_MAPPING(cycle_dependencies_map);
}

void update_temporary_precondition(void * k,
				   transformer pre,
				   hash_table precondition_map)
{
  transformer t_pre = (transformer) hash_get(precondition_map, k);

  if(t_pre == (transformer) HASH_UNDEFINED_VALUE) {
    pips_debug(2, "No previous precondition. Current one %p:\n", pre);
    ifdebug(2) print_transformer(pre);
    hash_put(precondition_map, k, (void *) transformer_dup(pre));
  }
  else {
    transformer n_t_pre = transformer_convex_hull(pre, t_pre);
    hash_update(precondition_map, k, (void *) n_t_pre);
    pips_debug(2, "Previous precondition %p:\n", t_pre);
    ifdebug(2) print_transformer(t_pre);
    pips_debug(2, "New precondition %p:\n", n_t_pre);
    ifdebug(2) print_transformer(n_t_pre);
    free_transformer(t_pre);
  }
}

void update_statement_temporary_precondition(statement s,
					 transformer pre,
					 statement_mapping statement_temporary_precondition_map)
{
  pips_debug(2, "For statement %s:\n", statement_identification(s));
  update_temporary_precondition
    ((void *) s, pre, (hash_table) statement_temporary_precondition_map);
}

void print_statement_temporary_precondition
(statement_mapping statement_temporary_precondition_map)
{
  if(hash_table_entry_count(statement_temporary_precondition_map)>0) {
    HASH_MAP(s, p, {
      fprintf(stderr, "Statement %s, Temporary precondition:\n",
	      statement_identification((statement) s));
      print_transformer((transformer) p);
    }, statement_temporary_precondition_map);
  }
  else {
    pips_assert("The DAG contains at least one control node with one statement", FALSE);
  }
}

void update_cycle_temporary_precondition(control a,
					 transformer pre,
					 control_mapping cycle_temporary_precondition_map)
{
  statement s = control_statement(a);

  pips_debug(2, "For control %p with statement %s:\n", a, statement_identification(s));
  update_temporary_precondition
    ((void *) a, pre, (hash_table) cycle_temporary_precondition_map);
}

transformer load_cycle_temporary_precondition(control a,
					      control_mapping cycle_temporary_precondition_map)
{
  transformer t_pre = (transformer) hash_get(cycle_temporary_precondition_map, (void *) a);

  pips_assert("The cycle precondition is available",
	      t_pre != (transformer) HASH_UNDEFINED_VALUE);

    return t_pre;
}

transformer load_statement_temporary_precondition(statement s,
						  statement_mapping statement_temporary_precondition_map)
{
  transformer t_pre = (transformer) hash_get(statement_temporary_precondition_map, (void *) s);

  pips_assert("The cycle precondition is available",
	      t_pre != (transformer) HASH_UNDEFINED_VALUE);

    return t_pre;
}

static void cycle_to_flow_sensitive_preconditions
(list partition,           /* Is it relevant? */
 unstructured c_u,         /* The cycle unstructured */
 hash_table ancestor_map, 
 hash_table scc_map, 
 control_mapping fix_point_map, 
 control_mapping cycle_temporary_precondition_map, /* To be updated when a cycle head is encountered */
 statement_mapping statement_temporary_precondition_map, /* To be updated each time a statement is encountered */
 transformer c_pre,        /* precondition of cycle c_u, already processed with the fix point */
 transformer pre, /* aproximate precondition holding for all nodes */
 control_mapping control_postcondition_map) /* Very likely the transformer paths wrt the entry point and not really a postcondition. Let's update it as a real postcondition. */
{
  /* Yet another forward DAG propagation with an exception for the cycle
     head: cut-and-paste and adapt */
  list to_be_processed = NIL; /* forward reachable nodes in u */
  list still_to_be_processed = NIL;
  list already_processed = NIL;
  control e = unstructured_control(c_u);
  statement es = control_statement(e);
  transformer post = transformer_undefined;

  /* FI: Please the compiler complaining about useless parameters (and
     remove them later!) */
  pips_assert("Please the compiler", partition==partition);
  pips_assert("Please the compiler", ancestor_map==ancestor_map);
  pips_assert("Please the compiler", scc_map==scc_map);
  pips_assert("Please the compiler", fix_point_map==fix_point_map);
  pips_assert("Please the compiler",
	      cycle_temporary_precondition_map==cycle_temporary_precondition_map);
  pips_assert("Please the compiler", pre==pre);
  
  /* wide_forward_control_map_get_blocs(unstructured_control(u), &to_be_processed); */
  forward_control_map_get_blocs(unstructured_control(c_u), &to_be_processed);
  still_to_be_processed = gen_copy_seq(to_be_processed);

  /* Take care of the entry node whose precondition does not rely on its
     predecessors */
  /* post = statement_to_postcondition(c_pre, es); */ /* c_pre is USED */ 
  post = transformer_apply(load_statement_transformer(es), c_pre);
  update_control_postcondition(e, post, control_postcondition_map);
  gen_remove(&still_to_be_processed, e);
  already_processed = CONS(CONTROL, e, NIL);

  /* Propagate the postcondition down */

  while(!ENDP(still_to_be_processed)) {
    int count = -1;
    do {
      list l = list_undefined;

      /* process forward */
      pips_debug(5, "Try forward processing for\n");
      ifdebug(5) print_control_nodes(still_to_be_processed);

      count = 0;
      for(l=still_to_be_processed; !ENDP(l); ) {
	control c = CONTROL(CAR(l));
	POP(l); /* right away because c's cdr might be modified */
	if(meaningless_control_p(c) 
	   || ready_to_be_processed_p(c, to_be_processed,
				      still_to_be_processed,
				      already_processed,
				      control_postcondition_map)) {
	  if(!meaningless_control_p(c)) {
	    control a = control_to_ancestor(c, ancestor_map);
	    /*
	    process_ready_node(c, n_pre, n_pre, ndu, control_postcondition_map,
			       ancestor_map, scc_map, partition, fix_point_map, postcondition_p);
	    */
	    statement c_s = control_statement(c);
	    transformer pre = 
	      get_control_precondition(c, control_postcondition_map, c_u, c_pre);
	    transformer e_pre = transformer_undefined;
	    transformer f_e_pre = transformer_undefined;

	    /* Already done earlier
	    if(cycle_head_p(c, ancestor_map, scc_map)) {
	      transformer fp_tf =  load_control_fix_point(a, fix_point_map, ancestor_map);
	      e_pre = transformer_apply(fp_tf, pre);
	      update_cycle_temporary_precondition
		(a, e_pre, cycle_temporary_precondition_map);
	    }
	    else {
	      e_pre = transformer_dup(pre);
	    }
	    */
	    e_pre = transformer_dup(pre);
	    if(cycle_head_p(c, ancestor_map, scc_map)) {
	      update_cycle_temporary_precondition
		(a, e_pre, cycle_temporary_precondition_map);
	    }
	    update_statement_temporary_precondition
	      (c_s, e_pre, statement_temporary_precondition_map);
	    /* post = statement_to_postcondition(transformer_dup(e_pre), c_s); */
	    /* FI: you should use the really current precondition using
               load_statement_temporary_precondition() */
	    /* post = transformer_apply(load_statement_transformer(c_s), e_pre); */
	    /* Hard to track memory leaks... */
	    /* It is much too late to update the control postcondition! */
	    /* update_control_postcondition(c, post, control_postcondition_map); */
	    /* FI: just to see what happens, without any understanding of the process */
	    f_e_pre = load_statement_temporary_precondition(c_s,
							    statement_temporary_precondition_map);
	    post = transformer_apply(load_statement_transformer(c_s), f_e_pre);
	    update_control_postcondition(c, post, control_postcondition_map);
	  }
	  gen_remove(&still_to_be_processed, c);
	  already_processed = gen_append(already_processed, CONS(CONTROL, c, NIL));
	  count++;
	}
      }
    } while(count!=0);
    if(!ENDP(still_to_be_processed)) {
      pips_assert("still_to_be_processed is empty because of the Bourdoncle's restructuring",
		  FALSE);
    }
  }
  /* The precondition for the entry node may be improved by now using the
     precondition of its predecessors and the cycle entry. OK, but to what end? */
}

static void dag_to_flow_sensitive_preconditions
(list partition, 
 unstructured ndu, 
 hash_table ancestor_map, 
 hash_table scc_map, 
 control_mapping fix_point_map, 
 transformer pre_u, 
 transformer pre,
 control_mapping control_postcondition_map)
{
  /* A statement may appear in several nodes, and the same is true for cycles. */
  statement_mapping statement_temporary_precondition_map = MAKE_STATEMENT_MAPPING();
  control_mapping cycle_temporary_precondition_map = MAKE_CONTROL_MAPPING();
  control_mapping cycle_dependencies_map = MAKE_CONTROL_MAPPING();
  list still_to_be_processed = NIL;
  list already_processed = NIL;
  control e = unstructured_control(ndu);
  list nl = NIL;
  list c_c = list_undefined;

  ifdebug(2) {
    pips_debug(2, "Begin with control_postcondition_map=%p:\n",
	       control_postcondition_map);
    print_control_postcondition_map(control_postcondition_map);
  }

  /* 
   * Process first the nodes in the embedding graph, in any order since
   * node postconditions are assumed available.
   *
   */
  control_map_get_blocs(e, &nl);

  for(c_c = nl; !ENDP(c_c); POP(c_c)) {
    control c = CONTROL(CAR(c_c));

    if(!meaningless_control_p(c)) {
      control a = control_to_ancestor(c, ancestor_map);
      transformer pre = get_control_precondition(c, control_postcondition_map, ndu, pre_u);
      transformer e_pre = transformer_undefined;
      statement s = control_statement(c);

      if(cycle_head_p(c, ancestor_map, scc_map)) {
	transformer fp_tf = load_control_fix_point(a, fix_point_map, ancestor_map);

	/* DUPLICATED CODE */

	/* The cycle is entered */
	transformer pre_cycle = transformer_apply(fp_tf, pre);
	/* Or the cycle is not entered and the store is not in fptf's domain. */
	transformer pre_no_cycle = transformer_dup(pre);
	if(control_test_p(c)) {
	  /* Have we followed a true or a false branch? We'll know later but
	     it is better to add this information before the convex hull is
	     performed. */
	  statement stmt = control_statement(c);
	  expression e = test_condition(statement_test(stmt));

	  /* FI: I do not know why an empty context is passed down. */
	  if(true_successors_only_p(c)) {
	    pre_no_cycle = precondition_add_condition_information(pre_no_cycle, e,
								  transformer_undefined, TRUE);
	  }
	  else if(false_successors_only_p(c)) {
	    pre_no_cycle = precondition_add_condition_information(pre_no_cycle, e,
								  transformer_undefined, FALSE);
	  }
	}
	/* free_transformer(pre); */
	e_pre = transformer_convex_hull(pre_cycle, pre_no_cycle);
	free_transformer(pre_cycle);
	free_transformer(pre_no_cycle);
 
	/* END OF DUPLICATED CODE */

	/* e_pre = transformer_apply(fp_tf, pre); */
	update_cycle_temporary_precondition(a, e_pre, cycle_temporary_precondition_map);
      }
      else {
	e_pre = transformer_dup(pre);
      }
      /* FI: sharing between statement_temporary_precondition and
         statement_temporary_precondition; freeing the second one frees
         part of the first one */
      update_statement_temporary_precondition(s, e_pre, statement_temporary_precondition_map);
    }
  }

  gen_free_list(nl);

  ifdebug(2) {
    pips_debug(2, "\nPreconditions for statements after ND DAG processing:\n");
    print_statement_temporary_precondition(statement_temporary_precondition_map);
    pips_debug(2, "End of preconditions for statements after ND DAG processing:\n\n");
  }

  /*
   * Process the cycles from top to bottom in width first?
   * Let's use the partition generated by Bourdoncle?!?
   *
   * Since cycles are not duplicated, the preconditions of all their
   * instances must be unioned before preconditions can be propagated down
   * in the cycles.
   *
   * It is not clear than a topological order exists. Assume that C2
   * occurs in an entry path to C1 and within C1. Somehow the precise
   * precondition of C1 depends on C2's postcondition which depends on
   * C2's precondition. C2's precondition depends on C1's postcondition
   * which depends on C1's precondition and here is the loop.
   *
   * However, in the previous case, the first instance of C2 can be dealt
   * with its transformer and the precondition of its first instance.
   *
   * */

  /* Build the cycle dependencies */
  pips_debug(2, "Compute dependencies for %d cycle%s\n",
	     hash_table_entry_count(scc_map),
	     hash_table_entry_count(scc_map)>1? "s":"");
  HASH_MAP(c, cndu, {
    /* control a = control_to_ancestor(c, ancestor_map); */
    /* unstructured scc_ u = cycle_head_to_scc(c, ancestor_map, scc_map); */
    list nl = NIL;
    list c_cc = list_undefined;
    control head = unstructured_control((unstructured) cndu);
    statement hs = control_statement(head);

    pips_debug(8, "Compute dependencies for cycle %p with statement %s\n",
	       head, statement_identification(hs));

    control_map_get_blocs(head, &nl);

    for(c_cc=nl; !ENDP(c_cc); POP(c_cc)) {
      control cc = CONTROL(CAR(c_cc));

      if(!meaningless_control_p(cc)
	 && cycle_head_p(cc, ancestor_map, scc_map)) {
	control aa = control_to_ancestor(cc, ancestor_map);
	if(aa!=c) {
	  /* then cc depends on c */
	  add_cycle_dependency(aa, (control) c, cycle_dependencies_map);
	}
      }
    }

    still_to_be_processed = CONS(CONTROL, (control) c, still_to_be_processed);
    gen_free_list(nl);
  }, scc_map);

  ifdebug(2) {
    if(hash_table_entry_count(cycle_dependencies_map)>0) {
      pips_debug(2, "cycle dependencies:\n");
      HASH_MAP(c, l, {
	statement s = control_statement((control) c);
	pips_debug(2, "Cycle %p with statement %s depends on:\n",
		   c, statement_identification(s));
	print_control_nodes((list) l);
      }, cycle_dependencies_map);
    }
    else {
      /* No cycles */
      pips_assert("No cycles left to be processed", ENDP(still_to_be_processed));
    }
  }

  /* While some cycles still have to be processed. */
  while(!ENDP(still_to_be_processed)) {
    int count = -1;
    do {
      list l = list_undefined;
      count = 0;
      for(l=still_to_be_processed; !ENDP(l); ) {
	control c = CONTROL(CAR(l));

	list dep = get_cycle_dependencies(c, cycle_dependencies_map);
	POP(l); /* right away because c's cdr might be modified */

	ifdebug(8) {
	  if(ENDP(dep)) {
	    pips_debug(8, "Process cycle %p with statement %s\n",
		       c, statement_identification(control_statement(c)));
	  }
	}

	if(ENDP(dep)) {
	  /* The precondition of cycle c is complete */
	  control a = control_to_ancestor(c, ancestor_map);
	  transformer c_pre = load_cycle_temporary_precondition(a, cycle_temporary_precondition_map);
	  transformer fp_tf = load_control_fix_point(c, fix_point_map, ancestor_map);
	  unstructured c_u = cycle_head_to_scc(c, ancestor_map, scc_map);
	  transformer f_c_pre = transformer_apply(fp_tf, c_pre);
	  transformer e_c_pre = transformer_convex_hull(f_c_pre, c_pre);

	  cycle_to_flow_sensitive_preconditions
	    (partition, c_u, ancestor_map, scc_map, fix_point_map,
	     cycle_temporary_precondition_map, statement_temporary_precondition_map,
	     e_c_pre, pre, control_postcondition_map);
	  remove_cycle_dependencies(c, cycle_dependencies_map);
	  gen_remove(&still_to_be_processed, c);
	  already_processed = gen_append(already_processed, CONS(CONTROL, c, NIL));
	  count++;
	}
      }
    } while(count!=0);

    pips_assert("All cycles have been processed",
		ENDP(still_to_be_processed));
  }

  /*
   * Exploit the temporary preconditions
   *
   */
  HASH_MAP(s, s_pre, {
    transformer post =
      statement_to_postcondition((transformer) s_pre, (statement) s);
    free_transformer(post);
  }, statement_temporary_precondition_map);

}


/*
 * PROCESSING OF CYCLIC OR ACYCLIC CFG FOR PATH TRANSFORMERS, PATH
 * PRECONDITIONS, INNER TRANSFORMERS AND INNER PRECONDITIONS.
 *
 *  - dag_or_cycle_to_flow_sensitive_postconditions_or_transformers()
 *
 *  - dag_to_flow_sensitive_postconditions_or_transformers()
 *
 *  - cycle_to_flow_sensitive_postconditions_or_transformers()
 * */

transformer dag_or_cycle_to_flow_sensitive_postconditions_or_transformers
(list partition, /* Bourdoncle's processing ordering */
 unstructured ndu, /* A non-deterministic acyclic control flow graph */
 hash_table ancestor_map, /* Mapping from non-deterministic nodes to deterministic nodes */
 hash_table scc_map, /* mapping from deterministic nodes to non-deterministic cycles */
 control_mapping fix_point_map, /* mapping from cycle heads to fix point transformers*/
 transformer e_pre, /* precondition on entry */
 transformer n_pre, /* precondition true for any node. */
 hash_table control_postcondition_map, /* */
 bool postcondition_p, /* Compute transformers or preconditions */
 bool is_dag) /* Should ndu be a dag or a cycle? */
{
  transformer tf = transformer_undefined;
  list to_be_processed = NIL; /* forward reachable nodes in u */
  list still_to_be_processed = NIL;
  list already_processed = NIL;
  
  ifdebug(2) {
    pips_debug(2, "Begin with e_pre=%p:\n", e_pre);
    print_transformer(e_pre);
  }

  /* wide_forward_control_map_get_blocs(unstructured_control(u), &to_be_processed); */
  forward_control_map_get_blocs(unstructured_control(ndu), &to_be_processed);
  still_to_be_processed = gen_copy_seq(to_be_processed);

  pips_assert("Node lists are defined", !list_undefined_p(to_be_processed)
	      && !list_undefined_p(still_to_be_processed) && ENDP(already_processed) );

  if(!is_dag && !postcondition_p) {
    /* The entry node must be handled in a specific way... but not in the
       specific way implemented in process_ready_node() which only deals
       with DAG's. The entry node cannot be a meaningless control node. */
    control e = unstructured_control(ndu);
    statement es = control_statement(e);
    transformer etf = load_statement_transformer(es);
    transformer init = transformer_identity();
    transformer post = transformer_undefined;

    /* Since control nodes have been replicated, it is difficult to
       predict if es has already been processed or not. */
    if(transformer_undefined_p(etf)) {
      etf = statement_to_transformer(es, n_pre);
    }

    post = transformer_apply(etf, init);

    store_control_postcondition(e, post, control_postcondition_map);

    gen_remove(&still_to_be_processed, e);
    already_processed = CONS(CONTROL, e, NIL);
  }

  /* make_control_postcondition_map(); */

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
	if(meaningless_control_p(c) 
	   || ready_to_be_processed_p(c, to_be_processed,
				      still_to_be_processed,
				      already_processed,
				      control_postcondition_map)) {
	  if(!meaningless_control_p(c)) {
	    process_ready_node(c, e_pre, n_pre, ndu, control_postcondition_map,
			       ancestor_map, scc_map, partition, fix_point_map, postcondition_p);
	  }
	  gen_remove(&still_to_be_processed, c);
	  already_processed = gen_append(already_processed, CONS(CONTROL, c, NIL));
	  count++;
	}
      }
    } while(count!=0);
    if(!ENDP(still_to_be_processed)) {
      pips_assert("still_to_be_processed is empty because of the Bourdoncle's restructuring",
		  FALSE);
    }
  }

  /* Compute fix point transformer for cycle */
  if(!is_dag && !postcondition_p) {
    /* Compute the convex hull of the paths associated to each predecessor
       of the entry. Since this is a cycle, preds cannot be empty. */
    control e = unstructured_control(ndu);
    list preds = control_predecessors(e);
    control pred = CONTROL(CAR(preds));
    transformer path_tf =
      load_arc_precondition(pred, e, control_postcondition_map);
    transformer fp_tf = transformer_undefined;
    /* transformer fp_tf_plus = transformer_undefined; */
    control e_a = control_to_ancestor(e, ancestor_map);

    POP(preds);
    MAP(CONTROL, pred, {
      transformer pred_tf =
	load_arc_precondition(pred, e, control_postcondition_map);
      transformer old_path_tf = path_tf;

      pips_assert("Partial path transformer pred_tf is defined",
		  !transformer_undefined_p(pred_tf));
      path_tf = transformer_convex_hull(old_path_tf, pred_tf);
      free_transformer(old_path_tf);
    }, preds);

    fp_tf =  (*transformer_fix_point_operator)(path_tf);
    /* If an entry range is known, do not use it as there may be more than
       one occurence of the cycle and more than one entry range. Keep this
       refinement to the precondition computation phase? It might be too
       late. The the data structure should be change to store more than
       one fix point per cycle.  */
    /* Not convincing anyway: you should apply fp_tf_plus to e_pre for the
       next two lines to make sense. */
    /* Use the + fix point to preserve more information about the output
       and do not forget to perform a convex hull when you need a * fix
       point in process_ready_node() */
    fp_tf = transformer_combine(fp_tf, path_tf);
    /*
    fp_tf = transformer_convex_hull(fp_tf_plus, e_pre);
    */

    free_transformer(path_tf);
    /* e may have many synonyms */
    store_control_fix_point(e_a, fp_tf, fix_point_map);

    /* We might want to return the corresponding fix point? */
    tf = fp_tf;
  }

  gen_free_list(to_be_processed);
  to_be_processed = list_undefined;
  gen_free_list(still_to_be_processed);
  still_to_be_processed = list_undefined;
  gen_free_list(already_processed);
  already_processed = list_undefined;

  ifdebug(2) {
    pips_debug(2, "End for %s of %s %p with entry node %s and with transformer %p\n", 
	       postcondition_p? "postcondition" : "transformer",
	       is_dag? "dag" : "cycle", ndu,
	       statement_identification(control_statement(unstructured_control(ndu))), tf);
    print_transformer(tf);
  }

  return tf;
}

transformer dag_to_flow_sensitive_postconditions_or_transformers
(list partition,          /* Bourdoncle's processing ordering */
 unstructured ndu,        /* A non-deterministic acyclic control flow graph */
 hash_table ancestor_map, /* Mapping from non-deterministic nodes to deterministic nodes */
 hash_table scc_map,      /* mapping from deterministic nodes to non-deterministic cycles */
 control_mapping fix_point_map, /* mapping from cycle heads to fix point transformers*/
 transformer e_pre,       /* precondition on entry */
 transformer n_pre,       /* precondition true for any node. */
 hash_table control_postcondition_map, /* */
 bool postcondition_p)    /* Compute transformers or preconditions */
{
  transformer tf = transformer_undefined;

  tf = dag_or_cycle_to_flow_sensitive_postconditions_or_transformers
    (partition, ndu, ancestor_map, scc_map, fix_point_map, e_pre, n_pre,
     control_postcondition_map, postcondition_p, TRUE);

  return tf;
}

transformer cycle_to_flow_sensitive_postconditions_or_transformers
(list partition, /* Bourdoncle's processing ordering */
 unstructured ndu, /* A non-deterministic acyclic control flow graph */
 hash_table ancestor_map, /* Mapping from non-deterministic nodes to deterministic nodes */
  hash_table scc_map, /* mapping from deterministic nodes to non-deterministic cycles */
 control_mapping fix_point_map, /* mapping from cycle heads to fix point transformers*/
 transformer e_pre, /* precondition on entry */
 transformer n_pre, /* precondition true for any node. */
 hash_table control_postcondition_map, /* */
 bool postcondition_p) /* Compute transformers or preconditions */
{
  transformer tf = transformer_undefined;

  tf = dag_or_cycle_to_flow_sensitive_postconditions_or_transformers
    (partition, ndu, ancestor_map, scc_map, fix_point_map, e_pre, n_pre,
     control_postcondition_map, postcondition_p, FALSE);

  return tf;
}


/*
 * PROCESSING OF AN UNSTRUCTURED
 *
 *  - local_process_unreachable_node()
 *
 *  - node_to_path_transformer_or_postcondition()
 *
 *  - unstructured_to_accurate_postconditions_or_transformer()
 *
 */
  
static void local_process_unreachable_node(control c, struct  { 
    bool pcond;
    control_mapping smap;
} * pcontext)
{
  process_unreachable_node(c, pcontext->smap, pcontext->pcond);
}
  
/* A debug function to prettyprint the result */
static void node_to_path_transformer_or_postcondition(control c, struct  { 
    bool pcond;
    control_mapping smap;
} * pcontext)
{
  if(!meaningless_control_p(c)) {
    control_mapping control_postcondition_map = pcontext->smap;
    statement s = control_statement(c);
    transformer tf = load_control_postcondition(c, control_postcondition_map);

    pips_assert("Postcondition tf is defined", !transformer_undefined_p(tf));
    fprintf(stderr, "Statement %s", statement_identification(s));
    pips_assert("Transformer or postcondition is consistent",
		transformer_consistency_p(tf));
    print_transformer(tf);
  }
}
  
/* A debug function to prettyprint the fix points */
static void print_cycle_head_to_fixpoint(control c, struct  { 
    bool pcond;
    control_mapping smap;
} * pcontext)
{
  control_mapping control_fix_point_map = pcontext->smap;
  statement s = control_statement(c);
  transformer tf = load_control_postcondition(c, control_fix_point_map);

  /* Most control nodes are not associated to a fix point */
  if(!transformer_undefined_p(tf)) {
    fprintf(stderr, "Statement %s", statement_identification(s));
    pips_assert("Transformer or postcondition is consistent",
		transformer_consistency_p(tf));
    print_transformer(tf);
  }
}

/* compute either the postconditions in an unstructured or the transformer
   of this unstructured. In both cases, transformers for all nodes used to be
   supposed to be available. */
transformer unstructured_to_flow_sensitive_postconditions_or_transformers
(transformer pre_u, /* precondition at entry: e_pre */
 transformer pre,   /* precondition true for every node: n_pre */
 unstructured u, 
 bool postcondition_p)
{
  transformer post = transformer_undefined;
  hash_table ancestor_map = hash_table_undefined;
  hash_table scc_map = hash_table_undefined;
  control_mapping fix_point_map  = make_control_fix_point_map();
  unstructured ndu = unstructured_undefined;
  list partition = bourdoncle_partition(u, &ndu, &ancestor_map, &scc_map);
  control_mapping control_postcondition_map = make_control_postcondition_map();
  transformer pre_u_r = transformer_range(pre_u);
  transformer pre_r = transformer_range(pre);

  struct  { 
    bool pcond;
    control_mapping smap;
  } context = { postcondition_p, control_postcondition_map };
  
  struct  { 
    bool pcond;
    control_mapping smap;
  } fcontext = { postcondition_p, fix_point_map };

  ifdebug(2) {
    pips_debug(2, "Begin for %s with nodes:\n",
	       postcondition_p? "postconditions" : "transformer");
    /* Do not go down into nested unstructured */
    gen_multi_recurse(u, statement_domain, gen_false, gen_null,
		      control_domain, gen_true, print_control_node, NULL);
    pips_debug(2, "With entry nodes\n");
    print_control_node(unstructured_control(u));
    pips_debug(2, "And exit node\n");
    print_control_node(unstructured_exit(u));
    pips_debug(2, "And embedding graph:\n");
    gen_multi_recurse(ndu, statement_domain, gen_false, gen_null,
		      control_domain, gen_true, print_control_node, NULL);
  }

  /* Propagate the precondition in the DAG and recompute the cycle
     transformers or compute the path transformers and the cycle fix point
     transformers. */
  (void) dag_to_flow_sensitive_postconditions_or_transformers
    (partition, ndu, ancestor_map, scc_map, fix_point_map,
     postcondition_p? pre_u : pre_u_r, postcondition_p? pre : pre_r,
     control_postcondition_map, postcondition_p);
  
  if(postcondition_p) {
    /* Compute the real precondition for each statement and propagate
       postconditions in it. */

    dag_to_flow_sensitive_preconditions
    (partition, ndu, ancestor_map, scc_map, fix_point_map, pre_u, pre,
     control_postcondition_map);
  }

  /* Take care of unreachable nodes */
  gen_context_multi_recurse(
			    ndu, (void *) & context,
			    statement_domain, gen_false, gen_null,
			    control_domain, gen_true, local_process_unreachable_node,
			    NULL);

  ifdebug(2) {
    pips_debug(2, "%s for unstructured\n",
	       postcondition_p? "Postconditions": "Path transformer");
    gen_context_multi_recurse(
			      ndu, (void *) & context,
			      statement_domain, gen_false, gen_null,
			      control_domain, gen_true,
			      node_to_path_transformer_or_postcondition,
			      NULL);
    pips_debug(2, "End of map\n");
    if(hash_table_entry_count(fix_point_map)>0) {
      pips_debug(2, "Fix point map:\n");
      gen_context_multi_recurse(
				u, (void *) & fcontext,
				statement_domain, gen_false, gen_null,
				control_domain, gen_true,
				print_cycle_head_to_fixpoint,
				NULL);
      pips_debug(2, "End of fix point map\n");
      pips_debug(2, "Dump fix point map %p:\n", fix_point_map);
      HASH_MAP(k, v, {
	control c = (control) k;
	statement s = control_statement(c);
	transformer fp_tf = (transformer) v;

	print_control_node(c);
	fprintf(stderr, "Statement %s", statement_identification(s));
	print_transformer(fp_tf);
      }, fix_point_map);
      pips_debug(2, "End of fix point map dump\n");
    }
    else {
      fprintf(stderr, "No fix point. Empty fix point map\n");
    }
  }

  post = copy_transformer
    (load_control_postcondition(unstructured_exit(ndu),
				control_postcondition_map));
  control_postcondition_map = free_control_postcondition_map(control_postcondition_map);
  fix_point_map = free_control_fix_point_map(fix_point_map);

  ifdebug(2) {
    pips_assert("The postcondition post is defined", !transformer_undefined_p(post));
    pips_debug(2, "End with unstructured postcondition:\n");
    print_transformer(post);
  }

  return post;
}

/*
 * COMPUTATION OF POSTCONDITIONS OF UNSTRUCTURED AND IN UNSTRUCTURED
 *
 *
 *  - unstructured_to_postconditions()
 *
 *  - unstructured_to_accurate_postconditions()
 *
 *  - unreachable_node_to_transformer()
 *
 *  - unstructured_to_global_transformer()
 *
 *  - unstructured_to_accurate_transformer()
 *
 *
 */

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

  pips_debug(8, "begin\n");

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
	  transformer context = transformer_range(pre);

	  c_pre_m = precondition_add_condition_information(c_pre, e, context, true_false);
	  free_transformer(context);
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
    pips_debug(8, "exit postcondition:\n");
    (void) print_transformer(exit_post) ;
  }
  pips_debug(8, "end\n");

  return exit_post;
}

/* compute pre- and post-conditions in an unstructured from the entry
   precondition pre and return the exit postcondition. pre_u is pre
   filtered by the u's transformer and can be used for any node.  */
transformer unstructured_to_flow_sensitive_postconditions
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
  else if(!get_bool_property("SEMANTICS_ANALYZE_UNSTRUCTURED")) {
    pips_user_warning("\nControl flow graph not analyzed accurately"
		      " because property SEMANTICS_ANALYZE_UNSTRUCTURED is not set\n");
    post = unstructured_to_postconditions(pre_u, pre, u);
  }
  else {
    post = unstructured_to_flow_sensitive_postconditions_or_transformers
      (pre_u, pre, u, TRUE);
  }
  gen_free_list(succs);

  pips_assert("Postcondition for unstructured is consistent",
	      transformer_consistency_p(post));

  return post;
}

transformer unstructured_to_postcondition(
    transformer pre,
    unstructured u,
    transformer tf)
{
    transformer post;
    control c;

    pips_debug(8, "begin\n");

    pips_assert("unstructured u is defined", u!=unstructured_undefined);

    c = unstructured_control(u);
    if(control_predecessors(c) == NIL && control_successors(c) == NIL) {
	/* there is only one statement and no arcs in u; no need for a
           fix-point */
	pips_debug(8, "unique node\n");
	/* FI: pre should not be duplicated because
	 * statement_to_postcondition() means that pre is not
	 * going to be changed, just post produced.
	 */
	post = statement_to_postcondition(transformer_dup(pre),
					  control_statement(c));
    }
    else {
	/* Do not try anything clever! God knows what may happen in
	   unstructured code. Postcondition post is not computed recursively
	   from its components but directly derived from u's transformer.
	   Preconditions associated to its components are then computed
	   independently, hence the name unstructured_to_postconditionS
	   instead of unstructured_to_postcondition */
	/* propagate as precondition an invariant for the whole
	   unstructured u assuming that all nodes in the CFG are fully
	   connected, unless tf is not feasible because the unstructured
	   is never exited or exited thru a call to STOP which invalidates
	   the previous assumption. */
      transformer tf_u = transformer_undefined;
      transformer pre_n = transformer_undefined;

	pips_debug(8, "complex: based on transformer\n");
	if(transformer_empty_p(tf)) {
	  tf_u = unstructured_to_flow_insensitive_transformer(u);
	}
	else {
	  tf_u = tf;
	}
	pre_n = invariant_wrt_transformer(pre, tf_u);
	ifdebug(8) {
	  pips_debug(8, "filtered over approximated precondition holding for any node pre_n:\n");
	  (void) print_transformer(pre_n) ;
	}
	/* FI: I do not know if I should duplicate pre or not. */
	/* FI: well, dumdum, you should have duplicated tf! */
	/* FI: euh... why? According to comments about transformer_apply()
	 * neither arguments are modified...
	 */
	/* post = unstructured_to_postconditions(pre_n, pre, u); */
	post = unstructured_to_flow_sensitive_postconditions(pre, pre_n, u);
	pips_assert("A valid postcondition is returned",
		    !transformer_undefined_p(post));
	if(transformer_undefined_p(post)) {
	  post = transformer_apply(transformer_dup(tf), pre);
	}
	transformer_free(pre_n);
    }

    pips_debug(8, "end\n");

    return post;
}

/*
 * COMPUTATION OF TRANSFORMERS FOR UNSTRUCTUREDS AND THEIR COMPONENTS
 *
 *  - unreachable_node_to_transformer()
 *
 *  - unstructured_to_flow_insensitive_transformer()
 *
 *  - unstructured_to_accurate_transformer()
 *
 *
 */

static void unreachable_node_to_transformer(control c)
{
  statement s = control_statement(c);

  pips_debug(9,"Statement %s", statement_identification(s));

  if(statement_ordering(s)==STATEMENT_ORDERING_UNDEFINED) {
    pips_user_warning("Improper control restructuration, statement with no ordering:%s"
		      "May result from an unreachable exit node\n",
		      statement_identification(s));
  }

  if(transformer_undefined_p(load_statement_transformer(s))) {
    pips_user_warning("After restructuration, unexpected unreachable node:%s",
		      statement_identification(s));
    (void) statement_to_transformer(s, transformer_undefined);
  }
}

/* This simple fix-point over-approximates the CFG by a fully connected
 * graph. Each vertex can be executed at any time any number of times.
 *
 * The fix point is easy to compute because it is the fix point of the
 * convex hull all each node transformers.
 *
 * The result is sometimes surprising as users see the real paths and
 * cannot understand why a variable is declared modified when it obviously
 * is not. This mostly choses in node preconditions since the overall
 * unstructured transformer is not displayed usually and since it is
 * globally correct.
 *
 * But it is a perfectly valid over-approximation usable by automatic
 * analyses.
 *
 * Using the effects of the unstructured to derive a fix point leads to
 * the same surprise although it is as correct, using the same over
 * approximation of the control flow graph but a cruder version of the
 * transformers.
 *
 * This function is also used when computing preconditions if the exit
 * node is not reached (?). It assumes that transformers for all
 * statements in the unstructured have already been computed.
 *
 * */

transformer unstructured_to_flow_insensitive_transformer(unstructured u)
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

/* Computation of transformers associated to each node of u and to each of
 * its sub-statements.
 * */

static void unstructured_to_transformers(
					 unstructured u,
					 transformer pre) /* pre is valid for any node of u */
{
  list blocs = NIL ;
  control ct = unstructured_control(u) ;

  pips_debug(5,"begin\n");
  
  /* There is no need to compute transformers for unreachable code,
   * using CONTROL_MAP, but this may create storage and prettyprinter
   * problems because of the data structure inconsistency.
   */
  CONTROL_MAP(c, {
    statement st = control_statement(c) ;
    (void) statement_to_transformer(st, pre) ;
  }, ct, blocs) ;
  
  gen_free_list(blocs) ;
  
  pips_debug(5,"end\n");
}

transformer unstructured_to_transformer(unstructured u,
					transformer e_pre, /* precondition on entrance */
					list e) /* effects of u */
{
  transformer tf = transformer_undefined;
  list succs = NIL;
  control head = unstructured_control(u);
  control tail = unstructured_exit(u);
  /* approximate store condition for all control nodes: simple context for
     improved transformer derivation; it is not a precondition and should have no
     arguments. */
  transformer pre = transformer_undefined;
  /* Same as previous one for the store on entry in the unstructured. This
     the entry node usually has predecessors in the CFG, this is not the
     context for the entry node. It is not a precondition. */
  transformer pre_u = transformer_undefined;

  if(transformer_undefined_p(e_pre)) {
    /* No information available on entrance */
    pre = transformer_identity();
  }
  else {
    /* Cheapest fix point transformer. The flow insensitive fix point
       could be used instead. */
    transformer fpf = effects_to_transformer(e);

    /* pre is replaced by its range condition later when needed*/
    pre = transformer_safe_apply(fpf, e_pre);
    free_transformer(fpf);
  }

  pips_debug(8,"begin\n");

  forward_control_map_get_blocs(head, &succs);

  /* Transformers should be computed precisely whether the unstructured is
     left by the exit node or by an explicit or implicit call to STOP. */
  if(TRUE || gen_in_list_p(tail, succs)) {
    /* computing the transformer for u is like computing the postcondition
       with no information on entry: no, the input context may be used to
       refine the transformer. */

    if(transformer_undefined_p(e_pre)) {
      pre_u = transformer_identity();
    }
    else {
      /* pre_u is restricted to its range later when needed. */
      pre_u = transformer_dup(e_pre);
    }

    /* These tests should be performed at the scc level */
    if(gen_length(succs)>SEMANTICS_MAX_CFG_SIZE2) {
      pips_user_warning("\nControl flow graph too large for any analysis (%d nodes)\n"
			"Have you fully restructured your code?\n", gen_length(succs));
      unstructured_to_transformers(u, pre);
      tf = effects_to_transformer(e);
    }
    else if(gen_length(succs)>SEMANTICS_MAX_CFG_SIZE1) {
      pips_user_warning("\nControl flow graph too large for an accurate analysis (%d nodes)\n"
			"Have you fully restructured your code?\n", gen_length(succs));
      unstructured_to_transformers(u, pre);
      tf = unstructured_to_flow_insensitive_transformer(u);
    }
    else if(!get_bool_property("SEMANTICS_ANALYZE_UNSTRUCTURED")) {
      pips_user_warning("\nControl flow graph not analyzed accurately"
			" because property SEMANTICS_ANALYZE_UNSTRUCTURED is not set\n");
      unstructured_to_transformers(u, pre);
      tf = unstructured_to_flow_insensitive_transformer(u);
    }
    else {
      tf = unstructured_to_flow_sensitive_postconditions_or_transformers
	(pre_u, pre, u, FALSE);
    }
    free_transformer(pre_u);
  }

  if(!gen_in_list_p(tail, succs)) {
  /* Do something for nodes unreachable from the entry but linked to the exit */
    /* The control flow graph is never exited... by the exit node */
    /* The unstructured is never exited, but all nodes are supposed to
       have transformers. This would never occur if the control
       restructuration were clean unless an infinite loop is stopped
       within a called procedure. Control effects are not reported. */
    /* FI: pre should be used! */
    gen_multi_recurse(
		      u,
		      statement_domain, gen_false, gen_null,
		      control_domain, gen_true, unreachable_node_to_transformer,
		      NULL);
  }

  /* Might be useless because it's now performed just above and more
     generally by a gen_multi_recurse() */
  if(!gen_in_list_p(tail, succs)) {
    pips_assert("if exit is not reached, tf is empty", transformer_empty_p(tf));
    tf = transformer_empty();
  }

  gen_free_list(succs);
  free_transformer(pre);

  pips_debug(8,"end\n");

  return tf;
}

/*
 * TOTAL PRECONDITIONS
 *
 */

transformer unstructured_to_flow_sensitive_total_preconditions
(transformer t_post_u, transformer pre, unstructured u)
{
  transformer t_pre = transformer_undefined;
  transformer post = transformer_undefined;
  /* control tail = unstructured_exit(u); */

  pips_assert("Not implemented yet", FALSE);
  pips_assert("Shut up the compiler", t_post_u==t_post_u && pre==pre && u==u);

  pips_assert("Total precondition for unstructured is consistent",
	      transformer_consistency_p(t_pre));

  return post;
}
