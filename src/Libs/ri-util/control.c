/* Some utilities to deal with the control graph.
   It is mainly used by my unspaghettify and the controlizer.

   Ronan Keryell.
   */

/* 
 * $Id$
 */

#ifndef lint
char vcid_ri_util_control[] = "%A% ($Date: 2000/11/03 15:45:26 $, ) version $Revision$, got on %D%, %T% [%P%].\n Copyright (c) École des Mines de Paris Proprietary.";
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

/* Build recursively the list of all controls reachable from any control of
 * an unstructured. It is usually called from the CONTROL_MAP macro,
 * with the entry node of an unstructured as initial argument. It uses
 * both successors and predecessors to define reachability.
 *
 * l must be initialized, if only to NIL, but no list_undefined;
 */

void
control_map_get_blocs( c, l )
control c ;
cons **l ;
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

/* Same as above, but follows predecessors only */

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

/* transitive closure of c's predecessors, but for control f. This is used
   to process subgraphs. */
void
backward_control_map_get_blocs_but(control c, control f, list * l )
{
  if(gen_in_list_p(c, *l) || c == f) return;
  *l = CONS( CONTROL, c, *l ) ;
  MAPL( cs, {
    backward_control_map_get_blocs_but( CONTROL( CAR( cs )), f, l );
  }, control_predecessors( c )) ;
}

/* Same as above, but follows successors only */

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

/* transitive closure of c's successors, but for control f. This is used
   to process subgraphs. */
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

/* Test if a control node is in a list of control nodes: */
bool
is_control_in_list_p(control c,
		     list cs)
{
    MAP(CONTROL, cc, {
	if (cc == c)
	    return TRUE;
    }, cs);
    return FALSE;
}

/* Count the number of occurences of a control node in a list of control nodes */
int
occurences_in_control_list(control c,
		     list cs)
{
    return gen_occurences((gen_chunk *) c, cs);
}


/* Test the coherency of a control network.

   Do not verify the fact that nodes could appear twice in the case of
   unstructured tests. */
void
check_control_coherency(control c)
{
    list blocs = NIL;
    int i1, i2;

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
		    pips_assert("Control is correct", FALSE);
	    }
	}, control_successors(ctl));

	/* Test the coherency of the predecessors */
	MAP(CONTROL, cc, {
	    /* if (!is_control_in_list_p(ctl, control_successors(cc))) { */
	    if ((i1=occurences_in_control_list(ctl, control_successors(cc)))
		!= (i2=occurences_in_control_list(cc, control_predecessors(ctl)))) {
		if(i1==0) {
		    pips_debug(0, "Control node %p not in the successor list of %p\n", ctl, cc);
		}
		else {
		    pips_debug(0, "Control %p occurs %d times in the successor list of %p"
			       " while control %p occurs %d times in the predecessor list of %p\n",
			       ctl, i1, cc, cc, i2, ctl);
		}
		ifdebug(8)
		    pips_assert("Control is correct", FALSE);
	    }
	}, control_predecessors(ctl));
    }, c, blocs);
    gen_free_list(blocs);  
}


/* Display a list of control: */
void
display_address_of_control_nodes(list cs)
{
	MAP(CONTROL, cc,
	    {
		fprintf(stderr, "%p,", cc);
	    }, cs);
}


/* Display all the control nodes from c for debugging purpose: */
void
display_linked_control_nodes(control c) {
    list blocs = NIL;
    CONTROL_MAP(ctl, {
	fprintf(stderr, "%p (pred (#%d)=", ctl,
		gen_length(control_predecessors(ctl)));
	display_address_of_control_nodes(control_predecessors(ctl));
	fprintf(stderr, " succ (#%d)=", gen_length(control_successors(ctl)));
	display_address_of_control_nodes(control_successors(ctl));
	fprintf(stderr, "), ");
	ifdebug(8) {
	    pips_debug(0, "Statement of control %p:\n", ctl);
	    print_statement(control_statement(ctl));
	}
    }, c, blocs);
    gen_free_list(blocs);
    fprintf(stderr, "---\n");
}


/* Remove all the control nodes (with its statement) from c in the
   successor tree of c up to the nodes with more than 1 predecessor.
   The entry node of the unstructured is given to avoid removing it
   when there is an unreachable sequence pointing on it.

   If a control node contains a FORMAT, assume that it is useful and
   stop removing.

   The do_not_delete_node is expected to be the entry or the exit node
   for example in order not to delete them. */
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
    bool exit_node_has_been_seen = FALSE;
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
			exit_node_has_been_seen = TRUE;
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
   discarded, so... */
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
                               "property.");
          remove_a_control_from_an_unstructured_without_relinking(c);
       }      
    },
       unreachable_controls);
    set_free(useful_controls);
    set_free(unreachable_controls);
}


/* Replace each occurence of c in a_source_control_list_of_c with a
   a_dest_control_list_of_c: */
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
               pips_assert("remove_a_control_from_a_list_and_relink with not a good \"which_way\".\n", FALSE);
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


/* It removes a control node from its successor and predecessor list
   and relink the successor and the predecessor.   
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
   pips_assert("remove_a_control_from_an_unstructured: more than one successor", number_of_successors <= 1);
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


/* It removes a control node from its successor and predecessor. Can
   apply to unstructured "IF". */
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
   statements. The statements are assumed to be referenced in another
   way: */
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


/* Used to discard a control sequence without touching its statements.
 It also removes the reference to the sequence from the predecessors
 or the successors. */
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
      
      /* Protect the statement: */
      control_statement(c) = statement_undefined;
      control_successors(c) = NIL;
      control_predecessors(c) = NIL;

      free_control(c);

      if (c == end)
         break;
   }
}


/* Take a control sequence and return a list of all the statements in
   the sequence (in the same order... :-) ). */
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
   Assume that this edge does not already exist. */
void
link_2_control_nodes(control source,
		     control target)
{
    control_successors(source) = CONS(CONTROL,
				      target,
				      control_successors(source));
    control_predecessors(target) = CONS(CONTROL,
					source,
					control_predecessors(target));
}


/* Remove an edge between 2 control nodes.
   Assume that this edge does already exist. */
void
unlink_2_control_nodes(control source,
		       control target)
{
    gen_remove(&control_successors(source), target);
    gen_remove(&control_predecessors(target), source);
}


/* Fuse a 2 control node and add the statement of the second one to
   the statement of the first one. Assumes that the second node is the
   only successor of the first one:\. Do not update the entry or exit
   field of the unstructured. */
void
fuse_2_control_nodes(control first,
		     control second)
{
    if (gen_length(control_successors(second)) == 2) {
	/* If the second node has 2 successors, it is a test node. The
	   fused node has 2 successors and must be a test too. So, the
	   only thing I can do is to remove the first statement, just
	   keeping its comments: */
	string first_comment =
	    gather_all_comments_of_a_statement(control_statement(first));
	insert_comments_to_statement(control_statement(second),
				     first_comment);
	control_statement(first) = control_statement(second);	
    }
    else {
	/* If not, build a block with the 2 statements: */
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
			 CONTROL(CAR(cp)) = first;
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
			 CONTROL(CAR(cp)) = first;
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
