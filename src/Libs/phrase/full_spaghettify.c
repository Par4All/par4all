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
/**
 * The spaghettifier is used in context of PHRASE project while
 * creating "Finite State Machine"-like code portions in order to synthetise
 * them in reconfigurables units.
 *
 * This phases transforms all the module in a unique
 * unstructured statement where all the control nodes are:
 * - CALL
 * - or TEST
 * - or SEQUENCE
 *
 * In fact the module becomes a sequence statement with a beginning
 * statement, the unstructured statement and a final statement
 *
 * full_spaghettify      > MODULE.code
 *       < PROGRAM.entities
 *       < MODULE.code
 *
 */

#include <stdio.h>
#include <ctype.h>

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "effects.h"

#include "resources.h"

#include "misc.h"
#include "ri-util.h"
#include "effects-util.h"
#include "pipsdbm.h"

#include "text-util.h"

#include "dg.h"

#include "control.h"
#include "callgraph.h"

#include "spaghettify.h"
#include "phrase_tools.h"

static control full_spaghettify_statement (statement stat,
					   const char* module_name,
					   unstructured u,
					   control current_control,
					   control next_control);

static control connect_unstructured (statement unstructured_statement,
				     control current_control,
				     control next_control);

static void reduce_sequence (control current_control,
				sequence seq,
				control* new_entry,
				control* new_exit);

static void flatten_unstructured (unstructured the_unstructured);

static control replace_control_with_unstructured (unstructured the_unstructured,
						  control current_control);

static statement full_spaghettify_module (statement module_statement,
					  const char* module_name)
{
  statement returned_statement;
  unstructured new_unstructured;
  instruction unstructured_instruction;
  control entry;
  control exit;
  list blocs = NIL ;
  int stat_number;
  int stat_ordering;

  entry = make_control(make_continue_statement
		       (entity_empty_label()), NIL, NIL);
  exit = make_control(make_continue_statement
		       (entity_empty_label()), NIL, NIL);
  link_2_control_nodes (entry, exit);
  new_unstructured
    = make_unstructured(entry, exit);
 
  unstructured_instruction = make_instruction(is_instruction_unstructured,
					      new_unstructured);

 
  stat_number = statement_number(module_statement);
  stat_ordering = statement_ordering(module_statement);

  full_spaghettify_statement (module_statement,
			      module_name,
			      new_unstructured,
			      entry,
			      exit);

  returned_statement = make_statement(entity_empty_label(),
				      stat_number,
				      stat_ordering,
				      empty_comments,
				      unstructured_instruction,
				      NIL, NULL,
				      statement_extensions(module_statement), make_synchronization_none());
 
  pips_assert("Statement is consistent after FULL_SPAGUETTIFY",
	      statement_consistent_p(returned_statement));
 
  ifdebug(2) {
    CONTROL_MAP (current_control, {
      pips_assert("Statement is consistent after FULL_SPAGUETTIFY",
		  statement_consistent_p(control_statement(current_control)));
      debug_control("FSM STATE: Module control =======================================",current_control, 2);
    }, entry, blocs);
  }
  return returned_statement;
}

/**
 * This function reduce the sequence seq at the position in the control
 * graph indicated by control current_control. The sequence is reduced by
 * the creation of new control nodes corresponding to all the statements
 * in the sequence.  In addition new_entry and new_exit control pointers
 * are updated according to the transformation.
 * NOTE: this function don't do recursively the job, but is generally called
 * by flatten_unstructured which do the job recursively.
 */
static void reduce_sequence (control current_control,
			     sequence seq,
			     control* new_entry,
			     control* new_exit)
{
  control first_control = NULL;
  control last_control = NULL;
  control new_control = NULL;
  bool is_first_control;
  int i;

  if(gen_length(sequence_statements(seq)) == 0)
  {
    *new_entry = control_undefined;
    *new_exit = control_undefined;
    return;
  }

  /* Conserve lists of predecessors and successors */
  list predecessors = gen_copy_seq (control_predecessors(current_control));
  list successors = gen_copy_seq (control_successors(current_control));

  pips_debug(5,"CONTROL [%p]\n", current_control);
  ifdebug(5) {
    print_statement (control_statement(current_control));
  }

  /* Deconnect all the predecessors */
  for (i=0; i<(int)gen_length(predecessors); i++) {
    pips_debug(5,"Unlink predecessor [%p-%p]\n",
	       CONTROL(gen_nth(i,predecessors)),
	       current_control);
    unlink_2_control_nodes (CONTROL(gen_nth(i,predecessors)), current_control);
  }

  /* Deconnect all the successors */
  for (i=0; i<(int)gen_length(successors); i++) {
    pips_debug(5,"Unlink successor [%p-%p]\n",
	       current_control,
	       CONTROL(gen_nth(i,successors)));
    unlink_2_control_nodes (current_control, CONTROL(gen_nth(i,successors)));
  }

  is_first_control = true;

  /* We iterate on each statement in the sequence */
  FOREACH(STATEMENT, current_stat, sequence_statements(seq))
  {
    /* We build a new control node from current statement */
    new_control = make_control (current_stat, NIL, NIL);
    ifdebug(5) {
      print_statement (current_stat);
    }
    if (is_first_control) {
      /* For the first statement... */
      first_control = new_control;
      is_first_control = false;
      pips_debug(5,"First control %p\n", first_control);
      *new_entry = first_control;
      /* Reconnect all the predecessors */
      /* ATTENTION link_2_control_nodes add to the list at the first position,
	 so we need to reconnect in the reverse order */
      for (i=gen_length(predecessors)-1; i>=0; i--) {
	pips_debug(5,"Relink predecessor [%p-%p]\n",
		   CONTROL(gen_nth(i,predecessors)),
		   first_control);
	control c = CONTROL(gen_nth(i,predecessors));
	link_2_control_nodes (c, first_control);
      }
      /* Reconnect all the successors */
      /* ATTENTION link_2_control_nodes add to the list at the first position,
	 so we need to reconnect in the reverse order */
      for (i=gen_length(successors)-1; i>=0; i--) {
	pips_debug(5,"Relink successor [%p-%p]\n",
		   first_control,
		   CONTROL(gen_nth(i,successors)));
	link_2_control_nodes (first_control, CONTROL(gen_nth(i,successors)));
      }
    }
    else {
      /* If this is not the first statement, we have to link
	 it with the previous one */
      link_2_control_nodes (last_control, new_control);
      pips_debug(5,"Other control %p [%p-%p]\n",
		 new_control, last_control, new_control);
      /* Deconnect all the OLD successors */
      for (i=0; i<(int)gen_length(successors); i++) {
	pips_debug(5,"Unlink successor [%p-%p]\n",
		   last_control,
		   CONTROL(gen_nth(i,successors)));
	unlink_2_control_nodes (last_control, CONTROL(gen_nth(i,successors)));
      }
      /* Reconnect all the NEW successors */
      /* ATTENTION link_2_control_nodes add to the list at the first position,
	 so we need to reconnect in the reverse order */
      for (i=gen_length(successors)-1; i>=0; i--) {
	pips_debug(5,"Relink successor [%p-%p]\n",
		   new_control,
		   CONTROL(gen_nth(i,successors)));
	link_2_control_nodes (new_control, CONTROL(gen_nth(i,successors)));
      }
    }
    last_control = new_control;
    *new_exit = new_control;
  }
}
			    

/**
 * This function takes as entry an unstructured and flatten it:
 *  - by reducing hierarchical unstructured
 *  - by reducing sequence statement (the sequence control node is
 *    replaced by a sequence of new control nodes formed with the
 *    statements inside the sequence)
 */
static void flatten_unstructured (unstructured the_unstructured)
{
  list blocs = NIL;
  list sequences_to_reduce = NIL;
  list unstructured_to_flatten = NIL;
  int nb_of_sequences_to_reduce;
  int nb_of_unstructured_to_flatten;
  int debug_iteration = 0;

  /* Repeat until there is no more sequences to
     reduce and unstructured to flatten */

  do {

    blocs = NIL;
    debug_iteration++;
    pips_debug(2,"New iteration %d\n", debug_iteration);

    sequences_to_reduce = NIL;
    unstructured_to_flatten = NIL;

    ifdebug(5) {
      short_debug_unstructured (the_unstructured, 2);
    }
   
    /* First, check all the sequences to reduce */
    CONTROL_MAP (current_control, {
      instruction i;
      i = statement_instruction(control_statement(current_control));
      switch (instruction_tag(i)) {
      case is_instruction_sequence:
	{
	  sequences_to_reduce
	    = CONS(CONTROL,
		   current_control,
		   sequences_to_reduce);
	  pips_debug (5, "MARKING SEQUENCE: %p\n", current_control);
	  debug_control ("SEQUENCE: ", current_control, 5);
	}
      default:
	break;
      }
    }, unstructured_entry (the_unstructured), blocs);

    nb_of_sequences_to_reduce = gen_length (sequences_to_reduce);
   
    /* Do the job on the sequences */
    MAP (CONTROL, current_control, {
      instruction i = statement_instruction(control_statement(current_control));
      control new_entry_of_imbricated;
      control new_exit_of_imbricated;
      pips_debug(2,"Imbricated sequence: REDUCING\n");
      debug_control ("REDUCE SEQUENCE: ", current_control, 5);
      reduce_sequence (current_control,
		       instruction_sequence(i),
		       &new_entry_of_imbricated,
		       &new_exit_of_imbricated);
      pips_debug(5,"Entry of imbricated sequence is %p\n",new_entry_of_imbricated);
      pips_debug(5,"Exit of imbricated sequence is %p\n",new_exit_of_imbricated);
      if (current_control == unstructured_entry (the_unstructured)) {
	pips_debug(5,"Changing entry %p for %p\n",
		   unstructured_entry (the_unstructured),
		   new_entry_of_imbricated);
	unstructured_entry (the_unstructured) = new_entry_of_imbricated;
      }
      else if (current_control == unstructured_exit (the_unstructured)) {
	pips_debug(5,"Changing exit %p for %p\n",
		   unstructured_exit (the_unstructured),
		   new_exit_of_imbricated);
	unstructured_exit (the_unstructured) = new_exit_of_imbricated;
      }
      /*free_control(current_control);*/
    }, sequences_to_reduce);

    /* Check the unstructured to flatten */
    CONTROL_MAP (current_control, {
      instruction i = statement_instruction(control_statement(current_control));
      switch (instruction_tag(i)) {
      case is_instruction_unstructured:
	{
	  unstructured_to_flatten 
	    = CONS(CONTROL,
		   current_control,
		   unstructured_to_flatten);
	  pips_debug (5, "MARKING UNSTRUCTURED: %p\n", current_control);
	  debug_control ("UNSTRUCTURED: ", current_control, 5);
	}
      default:
	break;
      }
    }, unstructured_entry (the_unstructured), blocs);

    nb_of_unstructured_to_flatten = gen_length (unstructured_to_flatten);

    /* Do the job on the unstructured */
    MAP (CONTROL, current_control, {
      instruction i = statement_instruction(control_statement(current_control));
      unstructured u = instruction_unstructured(i);
      pips_debug(2,"Imbricated unstructured: FLATTENING\n");
      pips_debug(5,"Flatten unstructured\n");
      debug_control ("FLATTEN UNSTRUCTURED: ", current_control, 5);
      replace_control_with_unstructured (u, current_control);
      /*free_control(current_control);*/
    }, unstructured_to_flatten);

  }
  while (((nb_of_sequences_to_reduce > 0)
	 || (nb_of_unstructured_to_flatten > 0)) && (debug_iteration < 3));
 
}

/**
 * This function connect the unstructured unstructured_statement to the
 * current Control Flow Graph between control nodes current_control and
 * next_control
 */
static control connect_unstructured (statement unstructured_statement,
				     control current_control,
				     control next_control)
{
  unstructured the_unstructured
    = instruction_unstructured(statement_instruction(unstructured_statement));
  control exit;

  pips_assert("Control with 1 successors in CONNECT_UNSTRUCTURED",
	      gen_length(control_successors(current_control)) == 1);
  pips_assert("Control with 1 predecessor in CONNECT_UNSTRUCTURED",
	      gen_length(control_predecessors(next_control)) == 1);
  pips_assert("Control connections in CONNECT_UNSTRUCTURED",
	      CONTROL(gen_nth(0,control_successors(current_control)))
	      == next_control);

  exit = unstructured_exit (the_unstructured);

  pips_assert("Exit with no successors in CONNECT_UNSTRUCTURED",
	      gen_length(control_successors(exit)) == 0);
  /* pips_assert("Entry with no predecessor in CONNECT_UNSTRUCTURED",
     gen_length(control_predecessors(entry)) == 0); */

  pips_debug(5,"connect_unstructured BEFORE flatten_unstructured()\n");
  ifdebug(5) {
    print_statement (unstructured_statement);
  }

  flatten_unstructured (the_unstructured);
 
  pips_debug(5,"connect_unstructured AFTER flatten_unstructured()\n");
  ifdebug(5) {
    print_statement (unstructured_statement);
  }

  exit = unstructured_exit (the_unstructured);
  unlink_2_control_nodes (current_control, next_control);
  link_2_control_nodes (current_control, unstructured_entry(the_unstructured));
  link_2_control_nodes (unstructured_exit(the_unstructured), next_control);

  return exit;
}

/**
 * This function connect the unstructured the_unstructured to the
 * current Control Flow Graph at the place of specified control
 * the_control. The predecessors and successors will be the same as
 * the_control predecessors and successors.
 * Return the new current control, which is the exit of passed unstructured
 */
static control replace_control_with_unstructured (unstructured the_unstructured,
						  control current_control)
{
  control entry, exit;
  int i;

  /* Conserve lists of predecessors and successors */
  list predecessors = gen_copy_seq (control_predecessors(current_control));
  list successors = gen_copy_seq (control_successors(current_control));

  /* Deconnect all the predecessors */
  for (i=0; i<(int)gen_length(predecessors); i++) {
    unlink_2_control_nodes (CONTROL(gen_nth(i,predecessors)), current_control);
  }

  /* Deconnect all the successors */
  for (i=0; i<(int)gen_length(successors); i++) {
    unlink_2_control_nodes (current_control, CONTROL(gen_nth(i,successors)));
  }

  flatten_unstructured (the_unstructured);

  entry = unstructured_entry (the_unstructured);
  exit = unstructured_exit (the_unstructured);

  /*pips_assert("Entry of unstructured has no predecessor",
    gen_length(control_predecessors(entry)) == 0);*/

  pips_assert("Exit of unstructured has no successor",
	      gen_length(control_successors(exit)) == 0);

  /* Reconnect all the predecessors */
  /* ATTENTION link_2_control_nodes add to the list at the first position,
     so we need to reconnect in the reverse order */
  for (i=gen_length(predecessors)-1; i>=0; i--) {
    link_2_control_nodes (CONTROL(gen_nth(i,predecessors)), entry);
  }

  /* Reconnect all the successors */
  /* ATTENTION link_2_control_nodes add to the list at the first position,
     so we need to reconnect in the reverse order */
  for (i=gen_length(successors)-1; i>=0; i--) {
    link_2_control_nodes (exit, CONTROL(gen_nth(i,successors)));
  }

  return exit;
}

/**
 * This function recursively takes the stat statement and transform the
 * Control Flow Graph module_unstructured in order to generate equivalent
 * code. The statement to transform is assumed to be executed between
 * controls current_control and next_control. 
 *
 * Generated unstructured is equivalent to a FSM where all the states are
 * the different nodes of the Control Flow Graph
 *
 * This function return the control node corresponding to the new position
 * in the Control Flow Graph
 */
static control full_spaghettify_statement (statement stat,
					   const char* module_name,
					   unstructured module_unstructured,
					   control current_control,
					   control next_control)
{

  instruction i = statement_instruction(stat);

  ifdebug(2) {
    debug_statement("FULL_SPAGHETTIFY: Module statement: =====================================", stat, 2);
   
    pips_assert("Control with 1 successors in FULL_SPAGHETTIFY",
		gen_length(control_successors(current_control)) == 1);
    pips_assert("Control with 1 predecessor in FULL_SPAGHETTIFY",
		gen_length(control_predecessors(next_control)) == 1);
    pips_assert("Control connections in FULL_SPAGHETTIFY",
		CONTROL(gen_nth(0,control_successors(current_control)))
		== next_control);
  }

  switch (instruction_tag(i)) {
  case is_instruction_test:
    {
      pips_debug(2, "full_spaghettify_statement: TEST\n");
      return
	connect_unstructured (spaghettify_test (stat, module_name),
			      current_control,
			      next_control);
      break;
    }
  case is_instruction_sequence:
    {
      sequence seq = instruction_sequence(i);
      control last_control = current_control;
      pips_debug(2, "full_spaghettify_statement: SEQUENCE\n");  
      MAP(STATEMENT, current_stat,
      {
	last_control = full_spaghettify_statement (current_stat,
						   module_name,
						   module_unstructured,
						   last_control,
						   next_control);
      }, sequence_statements(seq));
      /*free_statement(stat);*/
      return last_control;
      break;
    }
  case is_instruction_loop: {
    pips_debug(2, "full_spaghettify_statement: LOOP\n");  
    return
      connect_unstructured (spaghettify_loop (stat, module_name),
			    current_control,
			    next_control);
    break;
  }
  case is_instruction_whileloop: {
    pips_debug(2, "full_spaghettify_statement: WHILELOOP\n");  
    return
      connect_unstructured (spaghettify_whileloop (stat, module_name),
			    current_control,
			    next_control);
    break;
  }
  case is_instruction_forloop: {
    pips_debug(2, "full_spaghettify_statement: FORLOOP\n");
    return connect_unstructured (spaghettify_forloop (stat, module_name),
				 current_control,
				 next_control);
    break;
  }
  case is_instruction_call: {
    control new_control = make_control (stat, NIL, NIL); 
    pips_debug(2, "full_spaghettify_statement: CALL\n");  
    unlink_2_control_nodes (current_control, next_control);
    link_2_control_nodes (current_control, new_control);
    link_2_control_nodes (new_control, next_control);
    return new_control;
    break;
  }
  case is_instruction_unstructured: {
    pips_debug(2, "full_spaghettify_statement: UNSTRUCTURED\n");  
    ifdebug(5) {
      print_statement (stat);
    }
    return
      connect_unstructured (stat,
			    current_control,
			    next_control);
    break;
  }
  default:
    pips_user_warning("full_spaghettify_statement: UNDEFINED\n");  
    return current_control;
    break;
  }


}


/*********************************************************
 * Phase main
 *********************************************************/

bool full_spaghettify(const char* module_name)
{
   /* get the resources */
  statement stat = (statement) db_get_memory_resource(DBR_CODE,
						      module_name,
						      true);

  set_current_module_statement(stat);
  set_current_module_entity(module_name_to_entity(module_name));
 
  debug_on("SPAGUETTIFY_DEBUG_LEVEL");

  /* Now do the job */ 
  stat = full_spaghettify_module(stat, module_name);

  pips_assert("Statement is consistent after FULL_SPAGUETTIFY",
	      statement_consistent_p(stat));
 
  pips_assert("Unstructured is consistent after FULL_SPAGUETTIFY",
	      unstructured_consistent_p(statement_unstructured(stat)));

  /* Reorder the module, because new statements have been added */ 
  module_reorder(stat);
 
  ifdebug(5) {
    pips_debug(5,"====================================================\n");
    pips_debug(5,"Statement BEFORE simple_restructure_statement\n");
    print_statement (stat);
  }

  /* Restructure the module */
  simple_restructure_statement(stat);
  
  ifdebug(5) {
    pips_debug(5,"====================================================\n");
    pips_debug(5,"Statement AFTER simple_restructure_statement\n");
    print_statement (stat);
  }

  /* Reorder the module, because new statements have been added */ 
  module_reorder(stat);

  pips_assert("Statement is consistent after FULL_SPAGUETTIFY",
	      statement_consistent_p(stat));
 
  /**
   * ATTENTION
   * after simple_restructure_statement, statement stat is
   * not longer a unstructured, but may be a sequence !!!
   */
 
  DB_PUT_MEMORY_RESOURCE(DBR_CODE, module_name, stat);
  DB_PUT_MEMORY_RESOURCE(DBR_CALLEES, module_name,
			 compute_callees(stat));
 
  /* update/release resources */
  reset_current_module_statement();
  reset_current_module_entity();
 
  debug_off();
 
  return true;
}

