/*
   Hierarchize the control graph of a module.

   Replace unstructured by unstructured of unstructured and so on if
   possible by using interval graphs.

   Ronan.Keryell@cri.ensmp.fr
   */

#include <stdlib.h> 
#include <stdio.h> 
#include <string.h> 

#include "linear.h"

#include "genC.h"
#include "ri.h"

#include "interval_graph.h"

/* Instantiation of the interval graph: */
typedef interval_vertex_label vertex_label;
/* Since the arc are not used, just put a dummy type for arc_label. It
   will be initialized to interval_vertex_label_undefined anyway: */
typedef interval_vertex_label arc_label;

#include "graph.h"
#include "ri-util.h"
#include "misc.h"
#include "control.h"


/* *** Warning ***

   We use the NewGen graph structure in reverse order since we need
   predecessor graph and not successor graph. Furthermore, we do not
   use the arc_label.

   Let us lay down this cpp and type lies for programmer's sake...
   */
#define vertex_predecessors vertex_successors
#define predecessor_vertex successor_vertex
#define make_predecessor make_successor
#define free_predecessor free_successor
#define PREDECESSOR SUCCESSOR
#define PREDECESSOR_TYPE SUCCESSOR_TYPE
typedef successor predecessor;


/* Remove all the predecessors of an interval: */
static void
remove_interval_predecessors(vertex interval)
{
    /* Detach the predecessor vertex: */
    MAP(PREDECESSOR, p, {
	predecessor_vertex(p) = vertex_undefined;
    }, vertex_predecessors(interval));
    /* And remove all the predecessors: */
    gen_full_free_list(vertex_predecessors(interval));
    vertex_predecessors(interval) = NIL;
}


/* Remove all the instances of a predecessor pred of an interval.

   Return TRUE if pred was really in the predecessor list: */
static bool
remove_interval_predecessor(vertex interval,
			    vertex pred)
{
    bool pred_has_been_found_p = FALSE;
    list predecessors_to_remove = NIL;
    /* Detach the predecessor vertex: */
    MAP(PREDECESSOR, p, {
	if (predecessor_vertex(p) == pred) {
	    predecessor_vertex(p) = vertex_undefined;
	    predecessors_to_remove = CONS(PREDECESSOR, p, predecessors_to_remove);
	    pred_has_been_found_p = TRUE;
	}
    }, vertex_predecessors(interval));
    /* And now remove the predecessors than own pred: */
    MAP(PREDECESSOR, p, {
	gen_remove(&vertex_predecessors(interval), p);
	free_predecessor(p);
    }, predecessors_to_remove);

    return pred_has_been_found_p;
}


/* Add an interval node to an interval and remove the node. */
static void
add_node_to_interval(graph intervals,
		     vertex interval,
		     vertex node)
{
    /* Replace every allusion to node by interval in the interval
       graph: */
    MAP(VERTEX, v, {
	MAPL(ip, {
	    if (predecessor_vertex(PREDECESSOR(CAR(ip))) == node)
		predecessor_vertex(PREDECESSOR(CAR(ip))) = interval;
	}, vertex_predecessors(v));
    }, graph_vertices(intervals));

    /* Concatenate the control nodes to the interval and preserve the
       order to keep the entry node first: */
    interval_vertex_label_controls(vertex_vertex_label(interval)) =
	gen_nconc(interval_vertex_label_controls(vertex_vertex_label(interval)),
		  interval_vertex_label_controls(vertex_vertex_label(node)));
    /* Protect the control nodes from later deletion (useless to free
       the list since it is still lived via gen_nconc. Thanks to
       purify... :-) */
    interval_vertex_label_controls(vertex_vertex_label(node)) = NIL;
    /* Detach the node: */
    remove_interval_predecessors(node);
    /* Clean up the useless old node: */
    gen_remove(&graph_vertices(intervals), node);
    free_vertex(node);
}


/* Add the interval from (node, intervals) to interval graph
   intervals and update selected_nodes accordingly if : */
static void
add_to_interval_or_create_new_interval(vertex node,
				       graph intervals,
				       set selected_nodes)
{
    bool a_node_has_been_added;
    /* Since we modify in place the current interval graph, do not
       perturbate the do/MAP loop on intervals below. Just keep the
       list of interval to be removed later. */
    list intervals_to_be_removed = NIL;
    /* The new interval will be the node itself, begin of the new
       interval. Just select it and keep it: */
    set_add_element(selected_nodes, selected_nodes, (char *) node);
    
    do {
	a_node_has_been_added = FALSE;
	/* Find a candidate through all the intervals: */
	MAP(VERTEX, candidate, {
	    if (!set_belong_p(selected_nodes, (char *) candidate)) {
		bool all_predecessors_are_in_current_interval = TRUE;
		/* Test that the candidate has all its predecessors in
		   the interval we are building: */
		MAP(PREDECESSOR, predecessor, {
		    if (predecessor_vertex(predecessor) != node) {
			all_predecessors_are_in_current_interval = FALSE;
			break;
		    }
		}, vertex_predecessors(candidate));
		if (all_predecessors_are_in_current_interval) {
		    /* Ok, this node belong to the new interval: */
		  /*  add_node_to_interval(candidate,
					 node,
					 intervals,
					 selected_nodes,
					 &intervals_to_be_removed);*/
		    /* Look for the next appliant: */
		    a_node_has_been_added = TRUE;
		    break;
		}
	    }
	}, graph_vertices(intervals));
    } while(a_node_has_been_added);

    gen_full_free_list(intervals_to_be_removed);
}


static void	
display_interval_graph(graph intervals)
{
    MAP(VERTEX, node, {
	pips_debug(0, "Interval %p, control nodes:\n", node);
	display_address_of_control_nodes(interval_vertex_label_controls(vertex_vertex_label(node)));
	pips_debug(0, "Interval predecessors:\n");
	MAP(SUCCESSOR, p, {
	    pips_debug(0, "\t%p\n", predecessor_vertex(p));
	}, vertex_predecessors(node));    
    }, graph_vertices(intervals));
}


/* Build an interval graph from an older interval graph and put it in the
   older one. An interval is a subgraph whose header dominate all its
   nodes. It can be seen as a natural loop plus an acyclic structure that
   dangles from the nodes of that loop.

   Algorithm use the T1/T2 analysis that can be found from pages 665
   to 670 of:

@book{dragon-1986,
    author = {Alfred V. Aho and Ravi Sethi and Jeffrey D. Ullman},
    title = {Compilers Principles, Techniques and Tools},
    publisher = {Addison-Wesley Publishing Company},
    year = 1986
    }
*/
static bool
interval_graph(graph intervals)
{
    bool a_node_has_been_fused;
    bool the_interval_graph_has_been_modified = FALSE;
    vertex entry_interval = VERTEX(CAR(graph_vertices(intervals)));

    /* Apply the T2 transformation, that is equivalent to my
       fuse_sequences_in_unstructured elsewhere. Just use a less
       optimized algorithm here: */
    do {
	a_node_has_been_fused = FALSE;
	MAP(VERTEX, node, {
	    pips_debug(8, "vertex %p.\n", node);
	    if (node != entry_interval
		/* The entry interval is kept untouched */
		&& gen_length(vertex_predecessors(node)) == 1) {
		/* Fuse a node with its only predecessor: */
		vertex p_v = predecessor_vertex(PREDECESSOR(CAR(vertex_predecessors(node))));
		pips_debug(8, "\tonly one vertex predecessor %p.\n", p_v);
		add_node_to_interval(intervals, p_v, node);
		/* Let's go to find a new interval to fuse: */
		a_node_has_been_fused = TRUE;
		break;
	    }
	}, graph_vertices(intervals));
	the_interval_graph_has_been_modified |= a_node_has_been_fused;	
    } while (a_node_has_been_fused);

    /* T1 transformation on page 668: Remove the eventual arcs to
       itself: */
    MAP(VERTEX, node, {
	/* If a loop around a interval node is removed, it considered
	   as a graph modification: */
	the_interval_graph_has_been_modified
	    |= remove_interval_predecessor(node, node);
    }, graph_vertices(intervals));
    
    return the_interval_graph_has_been_modified;
}


/* Get the interval node of a control or create it if it does not
   exist and add it to the interval graph. */
static vertex
create_or_get_an_interval_node(control c,
			       graph intervals,
			       hash_table control_to_interval_node)
{
    vertex interval;
    if (!hash_defined_p(control_to_interval_node, (char *) c)) {
	/* This control has never been seen before: allocate the
	   peering interval node with no successor: */
	interval =
	    make_vertex(make_interval_vertex_label(CONS(CONTROL, c, NIL)),
			NIL);
	/* Use gen_nconc since the entry interval must be the first
	   one: */
	graph_vertices(intervals) = gen_nconc(graph_vertices(intervals),
					      CONS(VERTEX, interval, NIL));
	hash_put(control_to_interval_node, (char *) c, (char *) interval);
    }
    else
	interval = (vertex) hash_get(control_to_interval_node, (char *) c);

    return interval;
}


/* Duplicate the control graph in a format suitable to deal with
   intervals later.

   The interval graph format is the predecessor control graph in
   fact. */
static graph
control_graph_to_interval_graph_format(control entry_node)
{
    list blocs = NIL;
    graph intervals = make_graph(NIL);

    hash_table control_to_interval_node = hash_table_make(hash_pointer, 0);
    pips_debug(5, "Control entry node %p:\n", entry_node);   
    CONTROL_MAP(c, {
	vertex interval =
	    create_or_get_an_interval_node(c,
					   intervals,
					   control_to_interval_node);
	pips_debug(6, "\tControl %p -> interval %p\n", c,  interval);   
	MAP(CONTROL, p, {
	    vertex vertex_predecessor =
		create_or_get_an_interval_node(p,
					       intervals,
					       control_to_interval_node);
	    predecessor v_p = make_predecessor(interval_vertex_label_undefined,
					       vertex_predecessor);
	    vertex_predecessors(interval) =
		CONS(PREDECESSOR, v_p, vertex_predecessors(interval));
	    pips_debug(7, "\t\tControl predecessor %p -> interval %p\n",
		   p,  vertex_predecessor);   
	}, control_predecessors(c));
    }, entry_node, blocs);
    gen_free_list(blocs);
    
    hash_table_free(control_to_interval_node);

    return intervals;
}


/* Return the list of control node exiting an interval. Note that if a
   node of the control list is in fact the exit_note of a unstructure,
   it is really an exit node at an upper level. */
static list
interval_exit_nodes(vertex interval, control exit_node)
{
    list exit_controls = NIL;
    
    pips_debug(6, "Interval %p with controls ", interval);
    ifdebug(6)
	display_address_of_control_nodes(interval_vertex_label_controls(vertex_vertex_label(interval)));
    MAP(CONTROL, c, {
	pips_debug(7, "\n\tControl %p:\n", c);   
	MAP(CONTROL, successor, {	    
	    pips_debug(7, "\t\tControl successor %p:\n", successor);   
	    if (!gen_in_list_p(successor,
			       interval_vertex_label_controls(vertex_vertex_label(interval)))) {
		/* A successor that is not in the interval is an exit
                   node... Add it to the exit nodes list if not
                   already in it: */
		if (!gen_in_list_p(successor, exit_controls))
		    exit_controls = CONS(CONTROL, successor, exit_controls);
	    }
	}, control_successors(c));
	
	if (c == exit_node)
	    /* The current exit_node of the unstructured is clearly an
               exit node even if it does not have any successor: */
	    exit_controls = CONS(CONTROL, c, exit_controls);
    }, interval_vertex_label_controls(vertex_vertex_label(interval)));

    ifdebug(6) {
	pips_debug(0, "Interval exit node list: ");
	display_address_of_control_nodes(exit_controls);
	pips_debug(0, "\n");
    }
    
    return exit_controls;
}


void
control_list_patch(list l,
		   control c_old,
		   control c_new)
{
    gen_list_patch(l, (gen_chunk *) c_old, (gen_chunk *) c_new);
}


/* Transfer the control node c as a predecessor from old_node to
   new_node: */
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


/* Transfer the control node c as a successor from old_node to
   new_node: */
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


/* Replace all the references to old_node by new_node in the
   successors & predecessors of the controls in the control node
   list: */
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


/* Put all the controls in their own unstructured to hierarchize the
   graph and link the unstructured to the outer unstructured.
   
   The exit_node is the exit control node, either control_undefined if
   it there is no exit : it is a true endless loop (assume it is not
   the exit node). */
static void
hierarchize_control_list(vertex interval,
			 list controls,
			 control exit_node)
{
    /* The list of all reachable controls of the new unstructured: */
    list new_controls = gen_copy_seq(controls);

    control entry_node = CONTROL(CAR(controls));
    /* Create the new control nodes with the new unstructured: */
    control new_entry_node = make_control(control_statement(entry_node),
					  NIL, NIL);
    control new_exit_node = make_control(make_nop_statement(),
					 NIL, NIL);
    unstructured new_unstructured = make_unstructured(new_entry_node,
						      new_exit_node);
    control_statement(entry_node) =
	make_stmt_of_instr(make_instruction(is_instruction_unstructured,
					    new_unstructured));
    ifdebug(6) {
	pips_debug(0, "List of controls: ");
	display_address_of_control_nodes(controls);
	if (exit_node != control_undefined) {
            pips_debug(0, "\nExit node %p\n", exit_node);
        }
	pips_debug(0, "New unstructured %p: new_entry_node = %p, new_exit_node = %p\n",
		   new_unstructured, new_entry_node, new_exit_node);
    }
    /* Now the hard work: replace carefully the old control nodes by
       new one in the spaghetti plate... */

    /* First clone the graph structure: */
    control_successors(new_entry_node) =
	gen_copy_seq(control_successors(entry_node));
    control_predecessors(new_entry_node) =
	gen_copy_seq(control_predecessors(entry_node));
    control_list_patch(new_controls, entry_node, new_entry_node);
    if (exit_node != control_undefined) {
	control_successors(new_exit_node) =
	    gen_copy_seq(control_successors(exit_node));
	control_predecessors(new_exit_node) =
	    gen_copy_seq(control_predecessors(exit_node));
	/* Add the new_exit_node in the new_controls list: */
	new_controls = CONS(CONTROL, new_exit_node, new_controls);
    }
    ifdebug(6) {
	pips_debug(0, "new_controls list: ");
	display_address_of_control_nodes(new_controls);
	}

    /* Correct the nodes to reflect the new nodes: */
    MAP(CONTROL, c, {
	control_list_patch(control_successors(c), entry_node, new_entry_node);
	control_list_patch(control_predecessors(c), entry_node, new_entry_node);
	if (exit_node != control_undefined) {
	    control_list_patch(control_successors(c), exit_node, new_exit_node);
	    control_list_patch(control_predecessors(c), exit_node, new_exit_node);
	}
    }, new_controls);

    /* Detach the new unstructured from the old one: */
    MAP(CONTROL, c, {
	gen_list_and(&control_successors(c), new_controls);
	gen_list_and(&control_predecessors(c), new_controls);	
    }, new_controls);

    /* If there was a goto from exit_node to entry_node, there is an
       artefact one between new_exit_node and new_entry_node. Remove
       it: */
    unlink_2_control_nodes(new_exit_node, new_entry_node);
    
    /* Detach the old unstructured from the new one: */
    gen_list_and_not(&control_successors(entry_node), new_controls);
    gen_list_and_not(&control_predecessors(entry_node), new_controls);	

    /* Unlink an eventual loop around entry_node that has been
       captured anyway in the new unstructured: */
    unlink_2_control_nodes(entry_node, entry_node);
    
    if (exit_node != control_undefined) {
	gen_list_and_not(&control_successors(exit_node), new_controls);
	gen_list_and_not(&control_predecessors(exit_node), new_controls);
    }

    if (exit_node != control_undefined
	&& control_successors(entry_node) == NIL /* When a single node
	loop has been hierarchize, this link already exist */
	) {
	/* Now the exit_node becomes a successor of the entry_node: */
	link_2_control_nodes(entry_node, exit_node);
    }

    /* Update the control list of the interval that owns only
       entry_node after hierarchization: */
    gen_free_list(interval_vertex_label_controls(vertex_vertex_label(interval)));
    interval_vertex_label_controls(vertex_vertex_label(interval)) =
	CONS(CONTROL, entry_node, NIL);

    gen_free_list(new_controls);

    ifdebug(5) {
	pips_debug(0, "Nodes from entry_node: ");
	display_linked_control_nodes(entry_node);
	if (exit_node != control_undefined) {
	    pips_debug(0, "\nNodes from exit_node: ");
	    display_linked_control_nodes(exit_node);
	}
	pips_debug(0, "\nNodes from new_entry_node: ");
	display_linked_control_nodes(new_entry_node);
	pips_debug(0, "\nNodes from new_exit_node: ");
	display_linked_control_nodes(new_exit_node);
    }
    ifdebug(1) {
	check_control_coherency(new_entry_node);
	check_control_coherency(new_exit_node);
	check_control_coherency(entry_node);
	if (exit_node != control_undefined)
	    check_control_coherency(exit_node);
	pips_assert("Control should be consistent from entry_node)...",
		    control_consistent_p(entry_node));
    }
    pips_assert("new_exit_node cannot have a successor.",
		control_successors(new_exit_node) == NIL);
    ifdebug(7)
	print_statement(control_statement(entry_node));
}


/* Use an interval graph partitionning method to recursively
   decompose the control graph: */
void
control_graph_recursive_decomposition(unstructured u)
{
    /* An interval graph is represented by a graph with
       interval_vertex_label decorations embedding control nodes. The
       first interval of the graph is the entry interval of the
       interval graph and the first control node of an interval is the
       entry control node of the interval: */
    bool modified;
    control entry_node, exit_node;
    graph intervals;
    
    debug_on("RECURSIVE_DECOMPOSITION_DEBUG_LEVEL");
    
    entry_node = unstructured_control(u);
    exit_node = unstructured_exit(u);

    /* The seed interval graph is indeed the control graph itself: */
    intervals = control_graph_to_interval_graph_format(entry_node);

    pips_debug(3, "Entering with unstructured %p (%p, %p)\n",
	       u, entry_node, exit_node);
    ifdebug(5) {
	pips_debug(0, "Nodes from entry_node: ");
	display_linked_control_nodes(entry_node);
	pips_debug(0, "\nNodes from exit_node: ");
	display_linked_control_nodes(exit_node);
    }
    ifdebug(6)
	display_interval_graph(intervals);

    /* Apply recursively interval graph decomposition: */
    do {
	/* For all intervals of the graph: */
	MAP(VERTEX, interval, {
	    list controls = interval_vertex_label_controls(vertex_vertex_label(interval));
	    control interval_entry = CONTROL(CAR(controls));
	    list interval_exits = interval_exit_nodes(interval, exit_node);
	    if (gen_length(interval_exits) <= 1
		&& /* Useless to restructure the exit node... */
		interval_entry != exit_node
		&& /* If a single node, only useful if there is a loop
		      inside (in order to detect these loops,
		      hierarchize_control_list() is called before
		      interval_graph() but then we need to add more
		      guards...): */
		(gen_length(controls) > 1
		 || (gen_length(control_successors(interval_entry)) == 2
		     && (CONTROL(CAR(control_successors(interval_entry))) == interval_entry
			 || CONTROL(CAR(CDR(control_successors(interval_entry)))) == interval_entry)))
		) {
		/* If an interval has at most one exit, the
		   underlaying control graph can be hierachized by an
		   unstructured. */
		/* Put all the controls in their own unstructured
		   to hierarchize the graph: */
		hierarchize_control_list(interval,
					 controls,
					 interval_exits == NIL ? control_undefined : CONTROL(CAR(interval_exits)));
		
		gen_free_list(interval_exits);
	    }
	}, CDR(graph_vertices(intervals)) /* Skip the entry interval */);
	/* Stop if the interval graph does no longer changed : it is
           only one node or an irreductible graph: */
	/* Construct the interval graph from the previous one: */
	modified = interval_graph(intervals);
	pips_debug(6, "Modified = %d\n", modified);
	ifdebug(6)
	    display_interval_graph(intervals);		
    } while (modified);


    /* Do not forget to detach control nodes from interval before
       sweeping: */
    MAP(VERTEX, v, {
	gen_free_list(interval_vertex_label_controls(vertex_vertex_label(v)));
	interval_vertex_label_controls(vertex_vertex_label(v)) = NIL;
    }, graph_vertices(intervals));
    free_graph(intervals);
    
    pips_debug(3, "Exiting.\n");
    ifdebug(5) {
	pips_debug(0, "Nodes from entry_node: ");
	display_linked_control_nodes(entry_node);
	pips_debug(0, "\nNodes from exit_node: ");
	display_linked_control_nodes(exit_node);
    }
    ifdebug(1)
	pips_assert("Unstructured should be consistent here...",
		    unstructured_consistent_p(u));
    debug_off();
}
