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
/* Some utilities to deal with the control graph.
   It is mainly used by my unspaghettify and the controlizer.

   Ronan Keryell.
*/

#ifndef lint
char vcid_ri_util_control[] = "$Id$";
#endif /* lint */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "linear.h"

#include "genC.h"
#include "ri.h"
#include "ri-util.h"
#include "text-util.h"
#include "misc.h"

/* \defgroup control Control and unstructured methods

   Here are most of the functions that deals with control graph in PIPS
   that are encapsulated in what is called an "unstructured" in the RI.

   @{
 */


/* \defgroup control_visitors Control node visitors */

/* @{ */

/* Build recursively the list of all controls reachable from a control of
   an unstructured.

   It is usually called from the CONTROL_MAP macro, with the entry node of
   an unstructured as initial argument. It uses both successors and
   predecessors to define reachability, i.e. the graph arcs are
   considered edges.

   @param c is a control node to start with

   @param l is a list used to stored the visited nodes. It must be
   initialized to the list of nodes to skip. To visit all the nodes from
   c, just give a list variable initialized to NIL
*/
void control_map_get_blocs(control c, list *l)
{
    MAPL( cs,
    {if( CONTROL( CAR( cs )) == c ) return ;},
    *l ) ;
    *l = CONS( CONTROL, c, *l ) ;
    MAPL( cs,
    {control_map_get_blocs( CONTROL( CAR( cs )), l );},
    control_successors( c )) ;
    MAPL( ps, {control_map_get_blocs( CONTROL( CAR( ps )), l );},
    control_predecessors( c )) ;
}

/* Build recursively a control path from b to e

   It is used for debugging purposes

   @param b is a control node to begin with

   @param e is a control node to end with

   @param pp is a pointer to the list used to stored the path from b
   to e. It includes both b and e.

   @param vp is a pointer to the list used to stored the visited nodes. It must be
   initialized to the list of nodes to skip if any. To visit all the nodes from
   c, just give a list variable initialized to NIL. It usually must be
   by the caller.

   @param dir request a forward path is strictly positive, a backward
   path is stricly negative and an undirected path if zero.
*/
void find_a_control_path(control b, control e, list * pp, list * vp, int dir)
{
  if(b==e) {
    *pp = CONS(CONTROL, b, NIL);
    return;
  }

  if(ENDP(*pp) && !gen_in_list_p(b, *vp)) {
    *vp = CONS(CONTROL, b, *vp);
    if(dir>=0) {
      FOREACH(CONTROL, s, control_successors(b)) {
	find_a_control_path(s, e, pp, vp, dir);
	if(!ENDP(*pp)) {
	  pips_assert("e is the last element of *pp",
		      e==CONTROL(CAR(gen_last(*pp))));
	  *pp = CONS(CONTROL, s, *pp);
	  return;
	}
      }
    }
    if(dir<=0) {
      FOREACH(CONTROL, p, control_predecessors(b)) {
	find_a_control_path(p, e, pp, vp, dir);
	if(!ENDP(*pp)) {
	  pips_assert("e is the last element of *pp",
		      e==CONTROL(CAR(gen_last(*pp))));
	  *pp = CONS(CONTROL, p, *pp);
	  return;
	}
      }
    }
  }
  pips_assert("*pp is empty", ENDP(*pp));
  return;
}

/* Build recursively the list of all controls backward-reachable from a
   control of an unstructured.

   It is usually called from the BACKWARD_CONTROL_MAP macro, with the
   entry node of an unstructured as initial argument. It uses predecessors
   to define reachability.

   @param c is a control node to start with

   @param l is a list used to stored the visited nodes. It must be
   initialized to the list of nodes to skip. To visit all the nodes from
   c, just give a list variable initialized to NIL
*/
void
backward_control_map_get_blocs( c, l )
control c ;
cons **l ;
{
    MAPL( cs, {if( CONTROL( CAR( cs )) == c ) return ;}, *l ) ;
    *l = CONS( CONTROL, c, *l ) ;
    MAPL( cs,
	  {backward_control_map_get_blocs( CONTROL( CAR( cs )), l );},
	  control_predecessors( c )) ;
}


/* Transitive closure of c's predecessors, but for control f.

   It is like backward_control_map_get_blocs() but avoid visiting a
   control node. It is used to visit subgraphs begining at c and ending at
   f (excluded).

   @param c is a control node to start with

   @param f is a control node not to visit

   @param l is a list used to stored the visited nodes. It must be
   initialized to the list of nodes to skip. To visit all the nodes from
   c, just give a list variable initialized to NIL
*/
void
backward_control_map_get_blocs_but(control c, control f, list * l )
{
  if(gen_in_list_p(c, *l) || c == f) return;
  *l = CONS( CONTROL, c, *l ) ;
  MAPL( cs, {
    backward_control_map_get_blocs_but( CONTROL( CAR( cs )), f, l );
  }, control_predecessors( c )) ;
}


/* Build recursively the list of all controls forward-reachable from a
   control of an unstructured.

   It is usually called from the FORWARD_CONTROL_MAP macro, with the entry
   node of an unstructured as initial argument. It uses successors to
   define reachability.

   @param c is a control node to start with

   @param l is a list used to stored the visited nodes. It must be
   initialized to the list of nodes to skip. To visit all the nodes from
   c, just give a list variable initialized to NIL
*/
void
forward_control_map_get_blocs( c, l )
control c ;
cons **l ;
{
    MAPL( cs, {if( CONTROL( CAR( cs )) == c ) return ;}, *l ) ;
    *l = CONS( CONTROL, c, *l ) ;
    MAPL( cs,
	  {forward_control_map_get_blocs( CONTROL( CAR( cs )), l );},
	  control_successors( c )) ;
}

/* Transitive closure of c's successors, but for control f.

   It is like forward_control_map_get_blocs() but avoid visiting a
   control node. It is used to visit subgraphs begining at c and ending at
   f (excluded).

   @param c is a control node to start with

   @param f is a control node not to visit

   @param l is a list used to stored the visited nodes. It must be
   initialized to the list of nodes to skip. To visit all the nodes from
   c, just give a list variable initialized to NIL
*/
void
forward_control_map_get_blocs_but(control c, control f, list * l )
{
  if(gen_in_list_p(c, *l) || c == f) return;
  *l = CONS( CONTROL, c, *l ) ;
  MAPL( cs, {
    forward_control_map_get_blocs_but( CONTROL( CAR( cs )), f, l );
  }, control_successors( c )) ;
}


/* Same as above, but follows successors by minimal path lengths. It is OK
   if there is only one path length when computing transformers and/or
   preconditions. However, if a node is reached by several paths, the node
   with the minimal maximal path length should come first.

   This last condition assumes infinite path length for nodes in cycles
   (?). It is not implemented. */
void
wide_forward_control_map_get_blocs( c, l )
control c ;
cons **l ;
{
  list nodes = NIL;
  list frontier = CONS( CONTROL, c, NIL );
  list new_frontier = NIL;

  while(!ENDP(frontier)) {
    MAP(CONTROL,c ,{
      MAP(CONTROL, cp, {
	if(!is_control_in_list_p(cp, nodes)
	   && !is_control_in_list_p(cp, frontier)
	   && !is_control_in_list_p(cp, new_frontier))
	  new_frontier = CONS(CONTROL, cp, new_frontier);
      }, control_successors(c));
    }, frontier);
    nodes = gen_append(nodes, frontier);
    frontier = new_frontier;
    new_frontier = NIL;
  }

  *l = nodes;
}
/* @} */


/* \defgroup control_methods Functions to manage control the graph and
   unstructured

   @{
*/



/* Test if a control node is in a list of control nodes

   FI: is this different from gen_in_list_p(), but for the c type and
   the qualifiers?

   @param c control node to look for

   @param cs is the list of control to search through

   @return true if the control node is in the list
*/
bool
is_control_in_list_p(control c,
		     list cs)
{
    MAP(CONTROL, cc, {
	if (cc == c)
	    return true;
    }, cs);
    return false;
}

/* Count the number of occurences of a control node in a list of control
   nodes

   @param c control node to look for

   @param cs is the list of control to count through

   @return the number of occurences
 */
int
occurences_in_control_list(control c,
			   list cs)
{
    return gen_occurences((gen_chunk *) c, cs);
}


/* Replace in a list of control nodes all the instance of a control node
   by another one

   @param l is the list of control nodes

   @param c_old is the control node to remove

   @param c_new is the control node to put instead
 */
void
control_list_patch(list l,
		   control c_old,
		   control c_new)
{
    gen_list_patch(l, (gen_chunk *) c_old, (gen_chunk *) c_new);
}


/* Transfer a control node as a predecessor from one node to another one

   It disconnected a node @p c that was pointing to (was a predecessor of)
   @p old_node and reconnect it to @p new_node that becomes its new
   successor instead.

   @param old_node the control node that was a successor of @p c

   @param new_node the control node that will be a successor of @p c

   @param c the control node that is disconnected of @p old_node and
   connected to @p new_node
 */
void
transfer_control_predecessor(control old_node,
			     control new_node,
			     control c)
{
    MAP(CONTROL, predecessor, {
	if (predecessor == c)
	    /* Add c as a predecessor of new_node: */
	    control_predecessors(new_node) =
		gen_nconc(control_predecessors(new_node),
			  CONS(CONTROL, c, NIL));
    }, control_predecessors(old_node));
    /* Remove c as a predecessor of old_node: */
    gen_remove(&control_predecessors(old_node), c);
    /* Correct the reverse link c->old_node to c->new_node: */
    control_list_patch(control_successors(c), old_node, new_node);
}


/* Transfer a control node as a successor of one node to another one

   It disconnected a node @p c that was coming from (was a successor of)
   @p old_node and reconnect it to @p new_node that becomes its new
   predecessor instead.

   @param old_node the control node that was a predecessor of @p c

   @param new_node the control node that will be a predecessor of @p c

   @param c the control node that is disconnected of @p old_node and
   connected to @p new_node
*/
void
transfer_control_successor(control old_node,
			   control new_node,
			   control c)
{
    MAP(CONTROL, successor, {
	if (successor == c)
	    /* Add c as a predecessor of new_node: */
	    control_successors(new_node) =
		gen_nconc(control_successors(new_node),
			  CONS(CONTROL, c, NIL));
    }, control_successors(old_node));
    /* Remove c as a successor of old_node: */
    gen_remove(&control_successors(old_node), c);
    /* Correct the reverse link c->old_node to c->new_node: */
    control_list_patch(control_predecessors(c), old_node, new_node);
}


/* Replace all the references to a control node by a new one in the
   successors & predecessors of a list of controls

   @param old_node is the control node to replace

   @param new_node is the control node to replace

   @param controls is a list of controls we want to disconnect from @p
   old_node and reconnect to @p new_node instead
 */
void
replace_control_related_to_a_list(control old_node,
				  control new_node,
				  list controls)
{
    /* Need a intermediate list since we cannot iterate on
       control_successors(old_node) for example and modifying it...*/
    list controls_to_change = NIL;
    /* Since we need to keep successors order (to avoid for example
       IF/THEN/ELSE transformed in IF/ELSE/THEN), iterate directly on
       the links of old_node instead of on controls: */
    /* First transfer the successors in controls from old_node to
       new_node: */
    MAP(CONTROL, c, {
	if (gen_in_list_p(c, controls))
	    /* Use gen_nconc to keep test order: */
	    controls_to_change = gen_nconc(controls_to_change,
					   CONS(CONTROL, c, NIL));
    }, control_successors(old_node));
    /* And then do the modification: */
    MAP(CONTROL, c, {
	pips_debug(8, "Transfer old node %p to new node %p in successor %p\n", old_node, new_node, c);
	if (c != old_node)
	    transfer_control_successor(old_node, new_node, c);
	else {
	    /* Hmmm... We need to transfer a loop around old_node to a
               loop around new_node: */
	    /* Create the new loop around new_node: */
	    control_successors(new_node) =
		gen_nconc(control_successors(new_node),
			  CONS(CONTROL, new_node, NIL));
	    control_predecessors(new_node) =
		gen_nconc(control_predecessors(new_node),
			  CONS(CONTROL, new_node, NIL));
	    /* Delete the old one. Use gen_remove_once() instead of
               gen_remove() to deal with double loops around a node
               (See hierarchy02.f in validation): */
	    gen_remove_once(&control_successors(old_node), (gen_chunk *) old_node);
	    gen_remove_once(&control_predecessors(old_node), (gen_chunk *) old_node);
	}
    }, controls_to_change);
    gen_free_list(controls_to_change);

    /* And then transfer the predecessors in controls from old_node to
       new_node (the previous double loops have disappeared here): */
    controls_to_change = NIL;
    MAP(CONTROL, c, {
	if (gen_in_list_p(c, controls))
	    /* Use gen_nconc to keep test order: */
	    controls_to_change = gen_nconc(controls_to_change,
					   CONS(CONTROL, c, NIL));
    }, control_predecessors(old_node));
    /* And then do the modification: */
    MAP(CONTROL, c, {
	pips_debug(8, "Transfer old node %p to new node %p in predecessor %p\n", old_node, new_node, c);
	transfer_control_predecessor(old_node, new_node, c);
    }, controls_to_change);
    gen_free_list(controls_to_change);
}


/* Test the coherency of a control node network from a control node.

   Do not verify the fact that nodes could appear twice in the case of
   unstructured tests.

   @param c is the control node we want to start the verification from
 */
void
check_control_coherency(control c)
{
    list blocs = NIL;
    int i1, i2;
    set stmts = set_make(set_pointer);

    control_consistent_p(c);

    CONTROL_MAP(ctl, {

	/* Test the coherency of the successors */
	MAP(CONTROL, cc, {
	    /* if (!is_control_in_list_p(ctl, control_predecessors(cc))) { */
	    if ((i1=occurences_in_control_list(ctl, control_predecessors(cc)))
		!= (i2=occurences_in_control_list(cc, control_successors(ctl)))) {
		if(i1==0) {
		    pips_debug(0, "Control node %p not in the predecessor list of %p\n", ctl, cc);
		}
		else {
		    pips_debug(0, "Control %p occurs %d times in the predecessor list of %p"
			       " while control %p occurs %d times in the successor list of %p\n",
			       ctl, i1, cc, cc, i2, ctl);
		}
		ifdebug(8)
		    pips_assert("Control is correct", false);
	    }
	}, control_successors(ctl));

	/* Test the coherency of the predecessors */
	MAP(CONTROL, cc, {
	    /* if (!is_control_in_list_p(ctl, control_successors(cc))) { */
	    if ((i1=occurences_in_control_list(ctl, control_successors(cc)))
		!= (i2=occurences_in_control_list(cc, control_predecessors(ctl)))) {
	      bool consistent_p = false;
		if(i1==0) {
		    pips_debug(0, "Control node %p not in the successor list of %p\n", ctl, cc);
		}
		else {
		  if(statement_test_p(control_statement(cc)) && i1>=2 && i2==1) {
		    consistent_p = true;
		  }
		  else {
		    pips_debug(0, "Control %p occurs %d times in the successor list of %p"
			       " while control %p occurs %d times in the predecessor list of %p\n",
			       ctl, i1, cc, cc, i2, ctl);
		  }
		}
		ifdebug(8) {
		  if(!consistent_p) {
		    pips_debug(8, "control %p is not correct\n", cc);
		    pips_assert("Control is correct", consistent_p);
		  }
		}
	    }
	}, control_predecessors(ctl));

	/* Check that the statement are consistent */
	statement_consistent_p(control_statement(ctl));

	/* Check that two nodes do not point towards the same
	   statement as this makes label resolution ambiguous */
	if(set_belong_p(stmts, control_statement(ctl))) {
	  fprintf(stderr, "Statement %p is pointed by two different control nodes\n", control_statement(ctl));
	  pips_assert("each statement appears in at most one control node", false);
	}
	else
	  stmts = set_add_element(stmts, stmts, (void *) control_statement(ctl));
    }, c, blocs);

    set_free(stmts);
    gen_free_list(blocs);
}


// FI: as commented, needed for debugging purposes
// #if 0
/*
  Prettyprinting of control nodes for debugging purposes
*/
void print_control_node(control c)
{
  fprintf(stderr,
	  "ctr %p, %zd preds, %zd succs: %s",
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


/* Display identification of a list of control nodes */
void print_control_nodes(list l)
{
  if(ENDP(l)) {
    fprintf(stderr, "empty control list");
  }
  else {
    MAP(CONTROL, c, {
      fprintf(stderr, "%p, %s", c,
	      safe_statement_identification(control_statement(c)));
      // The version used in control/bourdoncle.c also had this check
      // (void) check_control_statement(c);
    }, l);
  }
  fprintf(stderr, "\n");
}
//#endif


/* Display the adresses a list of control nodes

   @param cs is the control node list
*/
void
display_address_of_control_nodes(list cs)
{
	MAP(CONTROL, cc,
	    {
		fprintf(stderr, "%p,", cc);
	    }, cs);
}


/* Display all the control nodes reached or reachable from c for debugging
   purpose

   Display also the statement of each control node if the debug level is
   high enough

   @param c is the control node we start the visit from
*/
void display_linked_control_nodes(control c)
{
  list blocs = NIL;
  set stmts = set_make(set_pointer);

  CONTROL_MAP(ctl, {
    fprintf(stderr, "%p (pred (#%zd)=", ctl,
	    gen_length(control_predecessors(ctl)));
    display_address_of_control_nodes(control_predecessors(ctl));
    fprintf(stderr, " succ (#%zd)=", gen_length(control_successors(ctl)));
    display_address_of_control_nodes(control_successors(ctl));
    fprintf(stderr, "), ");
    ifdebug(8) {
      fprintf(stderr, "\n");
      pips_debug(8, "Statement %p of control %p:\n", control_statement(ctl), ctl);
      safe_print_statement(control_statement(ctl));
    }
    if(set_belong_p(stmts, control_statement(ctl))) {
      fprintf(stderr, "Statement %p is pointed by two different control nodes\n",
	      control_statement(ctl));
    }
    else
      stmts = set_add_element(stmts, stmts, (void *) control_statement(ctl));
  }, c, blocs);
  gen_free_list(blocs);
  set_free(stmts);

  fprintf(stderr, "---\n");
}


/* Remove all the control nodes (with their statements) from @p c in the
   successor tree of @p c up to the nodes with more than 1 predecessor,
   that is when it reach another flow.

   The entry node of the unstructured is given to avoid removing it
   when there is an unreachable sequence pointing on it.

   If a control node contains a FORMAT, assume that it is useful and
   stop removing.

   The @param do_not_delete_node is expected to be the entry or the exit node
   for example in order not to delete them.

   @param c is the control we start the deletion from.

   @param do_not_delete_node is a control node we stop at when encountered

   @param do_not_delete_node_either is another control node we stop at
   when encountered
 */
void
remove_unreachable_following_control(control c,
				     control do_not_delete_node,
                                     control do_not_delete_node_either)
{
    list the_successors;

    /* If this is the do_not_delete_node nodes: stop deleting: */
    if (c == do_not_delete_node || c == do_not_delete_node_either)
	return;
    /* If this is not or no more the begin of a sequence, stop deleting: */
    if (control_predecessors(c) != NIL)
	return;
    /* If there is a FORMAT inside a control node, just stop deleting
       the control nodes since we cannot decide locally if the FORMAT
       is useful or not: */
    if (format_inside_statement_p(control_statement(c)))
	return;

    /* Save the successor list since we iterate o it and discard it at
       the same time: */
    the_successors = gen_copy_seq(control_successors(c));
    /* Ok, we can delete. For each successor of c: */
    MAP(CONTROL, a_successor, {
       /* Remove any predecessor reference of itself in this
          successor: */
       unlink_2_control_nodes(c, a_successor);
       remove_unreachable_following_control(a_successor,
                                            do_not_delete_node,
                                            do_not_delete_node_either);
    }, the_successors);
    gen_free_list(the_successors);

    /* Discard the control node itself: */
    pips_debug(7, "Discarding control node %p.\n", c);
    ifdebug(7) {
	display_linked_control_nodes(c);
    }
    free_control(c);
}


/* Remove all the control sequences that are unreachable and that
   begin with a node without any predecessor. It is an old version and
   a normal user should use
   remove_all_unreachable_controls_of_an_unstructured() instead.

   It is still buggy on Validation/Syntax/asgoto.f... */
void
remove_some_unreachable_controls_of_an_unstructured(unstructured u)
{
    list blocs = NIL;
    list control_remove_list = NIL;

    /* The entry point of the unstructured: */
    control entry_node = unstructured_control(u);
    control exit_node = unstructured_exit(u);
    bool exit_node_has_been_seen = false;
    pips_debug(7, "From control %p, exit %p.\n", entry_node, exit_node);
    ifdebug(7) {
	display_linked_control_nodes(entry_node);
    }

    CONTROL_MAP(c,
		{
		    if (c != entry_node)
			/* Well, the entry node is guessed as
			   reachable... :-) */
			if (control_predecessors(c) == NIL) {
			    /* A control without predecessor is
			       unreachable, so it is dead code: */
			    pips_debug(7, "Want to discard control %p.\n", c);
			    control_remove_list = CONS(CONTROL,
						       c,
						       control_remove_list);
			}
		    if (c == exit_node)
			/* Note that we could have entry_node ==
			   exit_node... */
			exit_node_has_been_seen = true;
		},
		entry_node,
		blocs);
    gen_free_list(blocs);

    /* Now remove all the marqued sequences from the entry_node: */
    MAP(CONTROL, c,
	{
	    remove_unreachable_following_control(c, entry_node, exit_node);
	},
	control_remove_list);
    gen_free_list(control_remove_list);

    if (!exit_node_has_been_seen) {
	/* Do not forget the unreachable exit part if it is not connex
           to the entry_node: */
	blocs = NIL;
	control_remove_list = NIL;
	CONTROL_MAP(c,
		    {
			if (c != exit_node)
			    /* Do not remove the exit_node... */
			    if (control_predecessors(c) == NIL) {
				/* A control without predecessor is
				   unreachable, so it is dead code: */
				control_remove_list = CONS(CONTROL,
							   c,
							   control_remove_list);
			    }
		    },
		    exit_node,
		    blocs);
	gen_free_list(blocs);
	/* Now remove all the marqued sequences from the entry_node: */
	MAP(CONTROL, c,
	    {
		remove_unreachable_following_control(c, entry_node, exit_node);
	    },
	    control_remove_list);
	gen_free_list(control_remove_list);
    }
}


/* Remove all control nodes that are not forward reachable from the
   entry node. Warning: useful FORMAT that are unreachable are also
   discarded, so...

   @param u is the unstructured to clean
*/
void
remove_all_unreachable_controls_of_an_unstructured(unstructured u)
{
    list blocs = NIL;

    set useful_controls = set_make(set_pointer);
    set unreachable_controls = set_make(set_pointer);

    /* The entry point of the unstructured: */
    control entry_node = unstructured_control(u);
    control exit_node = unstructured_exit(u);
    pips_debug(7, "From control %p, exit %p.\n", entry_node, exit_node);
    ifdebug(7) {
	display_linked_control_nodes(entry_node);
    }

    /* Mark all the forward-reachable nodes: */
    FORWARD_CONTROL_MAP(c, {
       pips_debug(5, "Forward visiting control node %p.\n", c);
       set_add_element(useful_controls,
                       useful_controls,
                       (char *) c);
    },
       entry_node,
       blocs);
    gen_free_list(blocs);
    blocs = NIL;

    /* Now build the remove list from all the non-marked nodes: */
    CONTROL_MAP(c, {
       pips_debug(5, "Testing control node %p.\n", c);
       if (! set_belong_p(useful_controls, (char *) c)) {
          pips_debug(5, "Adding to the removing list control node %p.\n", c);
          set_add_element(unreachable_controls,
                          unreachable_controls,
                          (char *) c);
       }
    },
       entry_node,
       blocs);
    gen_free_list(blocs);
    blocs = NIL;

    /* The same thing from the exit node that may be not reachable
       from the entry node: */
    CONTROL_MAP(c, {
       pips_debug(5, "Testing from exit control node %p.\n", c);
       if (! set_belong_p(useful_controls, (char *) c)) {
          pips_debug(5, "From exit node: Adding to the removing list control node %p.\n", c);
          set_add_element(unreachable_controls,
                          unreachable_controls,
                          (char *) c);
       }
    },
       exit_node,
       blocs);
    gen_free_list(blocs);

    /* And delete them: */
    SET_MAP(cc, {
       control c = (control) cc;
       if (c == exit_node) {
          pips_debug(5, "Skipping discarding exit control node %p.\n", c);
       }
       else {
          pips_debug(5, "Discarding control node %p.\n", c);
          if (format_inside_statement_p(control_statement(c)))
             pips_user_warning("Discarding an unreachable FORMAT that may be "
                               "usefull. "
                               "Try to use the GATHER_FORMATS_AT_BEGINNING "
                               "property.\n");
          remove_a_control_from_an_unstructured_without_relinking(c);
       }
    },
       unreachable_controls);
    set_free(useful_controls);
    set_free(unreachable_controls);
}


/* Replace each occurence of c in a_source_control_list_of_c with a
   a_dest_control_list_of_c:

   @param c is the control node to unlink. It is not freed

   @param a_source_control_list_of_c is the list of control nodes to be
   linked to the @p a_dest_control_list_of_c list of control nodes

   @param a_dest_control_list_of_c is the list of control nodes to be
   linked from the @p a_source_control_list_of_c list of control nodes

   @param which_way precise if we deal with successors or predecessors:

   - if it is source_is_predecessor_and_dest_is_successor: @p
   a_source_control_list_of_c is considered as predecessors of @p c and @p
   a_dest_control_list_of_c is considered as successors of @p c

   -if it is source_is_successor_and_dest_is_predecessor: @p
   a_source_control_list_of_c is considered as successors of @p c and @p
   a_dest_control_list_of_c is considered as predecessors of @p c
 */
void
remove_a_control_from_a_list_and_relink(control c,
                                        list a_source_control_list_of_c,
                                        list a_dest_control_list_of_c,
                                        remove_a_control_from_a_list_and_relink_direction which_way)
{
   MAPL(a_control_list,
        {
           list *the_dest_of_a_source_list = NULL;
           list the_position_of_c = NIL;
           list the_position_before_c = NIL;
           list l = NIL;

           control a_dest_of_a_source = CONTROL(CAR(a_control_list));
           /* Now, find the corresponding dest in the source list
              with the same value as c: */
           switch(which_way) {
             case source_is_predecessor_and_dest_is_successor:
               the_dest_of_a_source_list = &control_successors(a_dest_of_a_source);
               break;
             case source_is_successor_and_dest_is_predecessor:
               the_dest_of_a_source_list = &control_predecessors(a_dest_of_a_source);
               break;
             default:
               pips_assert("remove_a_control_from_a_list_and_relink with not a good \"which_way\".\n", false);
           }

           /* Find the reference to c in the the_dest_of_a_source_list: */
           the_position_before_c = NIL;
           MAPL(a_list,
                {
                   /* l is local to the MAPL... */
                   if (CONTROL(CAR(the_position_of_c = a_list)) == c)
                      break;
                   the_position_before_c = the_position_of_c;
                },
                   *the_dest_of_a_source_list);

           /* And add the a_dest_control_list_of_c instead of c: */
           /* First concatenate (with copy) a_dest_control_list_of_c
              before what follow c in the list: */
           l = gen_append(a_dest_control_list_of_c, CDR(the_position_of_c));

           /* Then place this list instead of c in
              *the_dest_of_a_source_list: */
           if (the_position_before_c == NIL)
              /* Modify the begin of the list: */
              *the_dest_of_a_source_list = l;
           else
              CDR(the_position_before_c) = l;

           /* Deallocate the cons that point to c: */
           CDR(the_position_of_c) = NIL;
           gen_free_list(the_position_of_c);
        },
           a_source_control_list_of_c);

   /* c is now in the memory nowhere: */
   control_predecessors(c) = NIL;
   control_successors(c) = NIL;
}


/* Remove a control node from a control graph

   The control node is freed and its predecessors are relinked to its
   successor and relink the successor and the predecessor.

   If you want to preserve the control statement, do a
   control_statement(c) = statement_undefined
   before calling this function.

   @param[in,out] c is the control node to unlink and to free

   Assume that it cannot have more than 1 successor (so no test node)

   If the graph is in an unstructured and @p c is either the entry or
   exit node, do not forget to update the entry or exit node.
*/
void
remove_a_control_from_an_unstructured(control c)
{
   list the_predecessors = control_predecessors(c);
   list the_successors = control_successors(c);

   int number_of_successors = gen_length(the_successors);

   /* Unlink from the predecessor. Note that a node may have more than
      one predecessor. Since we cannot discard an IF this way, we have
      at most 1 successor: */
   pips_assert("remove_a_control_from_an_unstructured:"
	       " no more than one successor",
	       number_of_successors <= 1);
   remove_a_control_from_a_list_and_relink(c,
                                           the_predecessors,
                                           the_successors,
                                           source_is_predecessor_and_dest_is_successor);

   /* Unlink from the successor: */
   remove_a_control_from_a_list_and_relink(c,
                                           the_successors,
                                           the_predecessors,
                                           source_is_successor_and_dest_is_predecessor);

   /* Remove the control node: */
   free_control(c);
}


/* It removes a control node from its successor and predecessor.

   It can be applied to an unstructured "IF".

   @param c is the control node to unlink and to free

   If the graph is in an unstructured and @param c is either the entry or
   exit node, do not forget to update the entry or exit node.
*/
void
remove_a_control_from_an_unstructured_without_relinking(control c)
{
   /* Use a copy since we iterate and discard at the same time: */
   list the_predecessors = gen_copy_seq(control_predecessors(c));
   list the_successors = gen_copy_seq(control_successors(c));

   MAP(CONTROL, a_predecessor, {
      unlink_2_control_nodes(a_predecessor, c);
   }, the_predecessors);
   gen_free_list(the_predecessors);

   MAP(CONTROL, a_successor, {
      unlink_2_control_nodes(c, a_successor);
   }, the_successors);
   gen_free_list(the_successors);

   pips_assert("The control node should not have any connection here,",
	       control_predecessors(c) == NIL && control_successors(c) == NIL);
   /* Remove the control node itself: */
   free_control(c);
}


/* Used to discard an unstructured without touching its
   statements.

   The statements are assumed to be referenced in another
   way.

   @param he unstructured to free
*/
void
discard_an_unstructured_without_its_statements(unstructured u)
{
   list blocs = NIL;

   /* Protect the statements by unlinking them: */
   CONTROL_MAP(c,
               {
                  control_statement(c) = statement_undefined;
               },
               unstructured_control(u),
               blocs);
   gen_free_list(blocs);

   /* And then free the discard the unstructured: */
   free_unstructured(u);
}


/* Remove a control node without touching its statement, its predecessors
   and successors, if any.

   @param c is the control node to free
 */
void
free_a_control_without_its_statement(control c) {
  /* Protect the statement: */
  control_statement(c) = statement_undefined;
  gen_free_list(control_successors(c));
  control_successors(c) = NIL;
  gen_free_list(control_predecessors(c));
  control_predecessors(c) = NIL;

  free_control(c);
}


/* Remove a control sequence without touching its statements.

   It also removes the reference to the sequence from the predecessors or
   the successors.

   @param begin is the control node we start from

   @param end is the control node to stop at

   Of course, there should be a unique path from @p begin to @p end.
*/
void
discard_a_control_sequence_without_its_statements(control begin,
                                                  control end)
{
   control c;
   list successor_list;

   /* Unlink any extern reference to the control sequence: */
   MAP(CONTROL, a_predecessor,
       {
          gen_remove(&control_successors(a_predecessor),
                     begin);
       },
       control_predecessors(begin));

   MAP(CONTROL, a_successor,
       {
          gen_remove(&control_predecessors(a_successor) ,
                     end);
       },
       control_successors(end));

   for(c = begin; ; c = CONTROL(CAR(successor_list))) {
      /* To pass through the free: */
      successor_list = control_successors(c);

      pips_assert("discard_a_control_sequence_without_its_statements: not a sequence.", gen_length(successor_list) <= 1);

      free_a_control_without_its_statement(c);

      if (c == end)
         break;
   }
}


/* Take a control sequence and return a list of all the statements in
   the sequence (in the same order... :-) ).

   @param begin is the control node we start from

   @param end is the control node to stop at

   Of course, there should be a unique path from @p begin to @p end.
*/
list
generate_a_statement_list_from_a_control_sequence(control begin,
                                                  control end)
{
   control c;
   list the_statements_of_the_sequence = NIL;

   /* Because of the way CONS is working, reversed the walk through
      the sequence. */

   for(c = end; ; c = CONTROL(CAR(control_predecessors(c)))) {
      int number_of_predecessor = gen_length(control_predecessors(c));
      pips_assert("discard_a_control_sequence_without_its_statements: not a sequence.", number_of_predecessor <= 1);

      /* Add the statement to the list: */
      the_statements_of_the_sequence = CONS(STATEMENT,
                                            control_statement(c),
                                            the_statements_of_the_sequence);
      if (c == begin)
         break;
   }

   return the_statements_of_the_sequence;
}


/* Add an edge between 2 control nodes.

   Assume that this edge does not already exist or the source should be an
   unstructured IF. FI: I am still puzzled how this can work when
   tests are involved since the semantics of the first and second
   successor is not paid attention at at all.

   @param source is the control node the edge starts from

   @param target is the control node the edge ends to
 */
void
link_2_control_nodes(control source,
		     control target)
{
  // FI: should we check here that the statement of "source" is
  // defined and if it is defined and that it already has a successor
  // then it is a test? Might be better done here than later when
  // trying to print out the statements...
  // FI: I think this explains why for loops are improperly desugared...

  // FI: this assert is too strong because if statements are
  // considered potentially structured and are linked like any other
  // statement when processing a statement
  //
  // pips_assert("source is not a test\n",
  //      statement_undefined_p(control_statement(source))
  //      || !statement_test_p(control_statement(source)));

  // FI: to avoid memory leaks and/or inconsistency
  //pips_assert("source has no successor\n", ENDP(control_successors(source)));

#if 0
  if(!ENDP(control_successors(source)))
    // FI: this should never happen if the graph is properly
    // handled...
    // I could re-instate the assert and/or just set a breakpoint on
    // the gen_free_list()
    gen_free_list(control_successors(source));
  control_successors(source) = CONS(CONTROL, target, NIL);
#endif

  // FI: assume the callers knows what it is doing when dealing with tests...
  control_successors(source) = CONS(CONTROL,
				    target,
				    control_successors(source));

  // FI: guess to get for loop properly desugared... but it breaks
  // something else...
  //control_successors(source) =
  //    gen_nconc(control_successors(source), CONS(CONTROL, target, NIL));
  control_predecessors(target) = CONS(CONTROL,
				      source,
				      control_predecessors(target));
}

/* Add an edge between 2 control nodes.

   Assume that this edge does not already exist or the source should be an
   unstructured IF. FI: I am still puzzled how this can work when
   tests are involved since the semantics of the first and second
   successor is not paid attention at at all.

   @param source is the control node the edge starts from

   @param target is the control node the edge ends to
 */
void
link_3_control_nodes(control c_test,
		     control c_then, control c_else)
{
  if(!ENDP(control_successors(c_test)))
    gen_free_list(control_successors(c_test));
  control_successors(c_test) =
    CONS(CONTROL, c_then, CONS(CONTROL, c_else, NIL));

  control_predecessors(c_then) = CONS(CONTROL,
				      c_test,
				      control_predecessors(c_then));
  control_predecessors(c_else) = CONS(CONTROL,
				      c_test,
				      control_predecessors(c_else));
}


/* Remove all edged between 2 control nodes.

   Note: if the source is a test, the false branch may be come the
   true branch if the true branch is unlinked

   @param source is the control node the edges start from

   @param target is the control node the edges end to
*/
void
unlink_2_control_nodes(control source,
		       control target)
{
  // FI: no check that the nodes are properly linked before unlinking
    gen_remove(&control_successors(source), target);
    gen_remove(&control_predecessors(target), source);
}


/* Insert a control node between 2 connected control nodes

   @param c is the control node to insert

   @param before is the control node before where @p c is to be inserted

   @param after is the control node after where @p c is to be inserted

   Assume that @p c is not already connected and that @p after is the
   successor of @p before.

   Note: statements associated to nodes are not tested in case they
   are undefined.
*/
void insert_control_in_arc(control c, control before, control after) {
  // These first two assertions are wrong when "before" is a test whose two
  // branches reach "after":
  //
  // if(c) goto l1; else goto l1; l1: ;
  //
  // This is unlikely but does happen when macros are used or when
  // some code is generated automatically. FI assumes that the two
  // substitutions will happen simultaneously thanks to the
  // substitutions in the list
  pips_assert("c is not already a successor of before",
	      !is_control_in_list_p(c, control_successors(before)));

  // This may be wrong if c has already been inserted in another
  //arc. Nevertheless it may be useful to insert it in the "before->after" arc.
  //
  //pips_assert("c is not a predecessor of after",
  //	      !is_control_in_list_p(c, control_predecessors(after)));

  // Make sure that before and after are properly linked
  pips_assert("after is a successor of before",
	      is_control_in_list_p(after, control_successors(before)));
  pips_assert("before is a predecessor of after",
	      is_control_in_list_p(before, control_predecessors(after)));

  // FI: we might also assert that the statement associated to c is
  // not a test... but it is not possible when sequences are
  // transformed into a chain of nodes. So, temporarily, you can have
  // test control nodes with only one successor. May be definitely if
  // they are structured?
  //pips_assert("c is not a test: it has only one successor at most",
  //	      !statement_test_p(control_statement(c)));

  // FI: when before is a test, how do you know if c must be in the
  // true or in the false branch?
  /* If there is no ambiguity about the linking, use Ronan's technique */
  if(gen_length(control_successors(before))==1
     && gen_length(control_successors(c))==0) {
    unlink_2_control_nodes(before, after);
    link_2_control_nodes(before, c);
    link_2_control_nodes(c, after);
  }
  else if(gen_length(control_successors(c))==0) {
    /* Let's try to preserve the true and false branch */
    bool success1 = gen_replace_in_list(control_successors(before), after, c);
    bool success2 = gen_replace_in_list(control_predecessors(after), before, c);
    // The order is meaning less, but it is easier to debug if the
    // order is preserved
    control_predecessors(c)
      = gen_nconc(control_predecessors(c), CONS(CONTROL, before, NIL));
    //control_predecessors(c) = CONS(CONTROL, before, NIL);
    control_successors(c) = CONS(CONTROL, after, NIL);

    pips_assert("after and before were linked", success1 && success2);
    pips_debug(8, "control %p inserted between before=%p and after=%p\n",
	       c, before, after);
  }
  else if(gen_length(control_successors(c))==1
       && CONTROL(CAR(control_successors(c)))==after) {
    // We may handle the case outlined above?
    // The two substitutions should have occured simultaneously?
    // No, simply the more general case:
    // precondition: x -> c && c -> after && before -> after
    // postcondition x -> c && before -> c && c -> after
    // pips_internal_error("Should not happen\n");
    // Preserve the branch positions for tests
    bool success = gen_replace_in_list(control_successors(before), after, c);
    gen_remove(&control_predecessors(after), before);
    control_predecessors(c)
      = gen_nconc(control_predecessors(c), CONS(CONTROL, before, NIL));
    pips_assert("after and before were linked", success);
  }
  else
    pips_internal_error("No semantics, no implementation...\n");
}

/* Fuse a 2 control nodes

   It adds the statement of the second one to the statement of the first
   one. Assumes that the second node is the only successor of the first
   one.

   The second control node is freed.

   It does not update the entry or exit field of the unstructured.

   @param first is the first control node

   @param second is the control node to fuse in the first one. It is
   precised because it is possible that there are not connected.
 */
void
fuse_2_control_nodes(control first,
		     control second)
{
    if (gen_length(control_successors(second)) == 2) {
	/* If the second node has 2 successors, it is a test node. The
	   fused node has 2 successors and must be a test too. So, the
	   only thing I can do is to remove the first statement, just
	   keeping its comments. And how about the label? It might be
	   useful for syntactic reasons and only reachable via END=nnn
	   after prettyprinting (see Validation/redlec2.f) */
	string first_comment =
	    gather_all_comments_of_a_statement(control_statement(first));
	entity first_label = statement_to_label(control_statement(first));

	if(!entity_empty_label_p(first_label)) {
	  entity second_label = statement_to_label(control_statement(second));
	    pips_user_warning("Useless label %s\n",
			      entity_name(first_label));
	  if(!entity_empty_label_p(second_label)) {
	    /* A return might be OK?!? What does the caller expect? The
               first label must be useless and is dropped. */
	    /* pips_user_warning("Useless label %s\n",
	       entity_name(first_label)); */
	    /* pips_internal_error("Two labels for one control node"); */
	    ;
	  }
	  else {
	    statement_label(control_statement(second)) = first_label;
	    ;
	  }
	}
	insert_comments_to_statement(control_statement(second),
				     first_comment);
	control_statement(first) = control_statement(second);
    }
    else {
	/* If not, build a block with the two statements: */
	statement st = make_empty_statement();
	statement_instruction(st) =
	    make_instruction_block(CONS(STATEMENT,
					control_statement(first),
					CONS(STATEMENT,
					     control_statement(second), NIL)));
	/* Reconnect the new statement to the node to fuse: */
	control_statement(first) = st;
    }
    control_statement(second) = statement_undefined;

    /* Unlink the second node from the first one: */
    gen_free_list(control_successors(first));
    gen_remove(&control_predecessors(second), first);

    /* Link the first node with the successors of the second one in
       the forward direction: */
    control_successors(first) =
	control_successors(second);
    /* Update all the predecessors of the successors: */
    MAP(CONTROL, c,
	{
	    MAPL(cp,
		 {
		     if (CONTROL(CAR(cp)) == second)
			 CONTROL_(CAR(cp)) = first;
		 }, control_predecessors(c));
	}, control_successors(first));

    /* Transfer the predecessors of the second node to the first one.
       Note that the original second -> first predecessor link has
       already been removed. But a loop from second to second appear
       at this point as a link from first to second. Nasty bug... */
    MAP(CONTROL, c,
	{
	    control_predecessors(first) = CONS(CONTROL,
					       c,
					       control_predecessors(first));
	    MAPL(cp,
		 {
		     if (CONTROL(CAR(cp)) == second) {
			 CONTROL_(CAR(cp)) = first;
		     }
		 }, control_successors(c));
	}, control_predecessors(second));

    /* Now we remove the useless intermediate node "second": */
    /* Do not gen_free_list(control_successors(second)) since it is
       used as control_successors(first) now: */
    control_successors(second) = NIL;
    gen_free_list(control_predecessors(second));
    control_predecessors(second) = NIL;
    free_control(second);
}

/*
  @}
*/
