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
/* Clean up the sequences and their contents by fusing them or
   removing useless continues or empty instructions.

   Ronan.Keryell@cri.ensmp.fr

   */

/* 	%A% ($Date: 2004/02/16 15:24:55 $, ) version $Revision$, got on %D%, %T% [%P%].\n Copyright (c) École des Mines de Paris Proprietary.	 */

#ifndef lint
char vcid_clean_up_sequences[] = "%A% ($Date: 2004/02/16 15:24:55 $, ) version $Revision$, got on %D%, %T% [%P%].\n Copyright (c) École des Mines de Paris Proprietary.";
#endif /* lint */


#include <stdlib.h>
#include <stdio.h>
#include <stdlib.h>

#include "linear.h"

#include "genC.h"
#include "ri.h"
#include "ri-util.h"
#include "text-util.h"
#include "properties.h"
#include "misc.h"


/* To store the statistics: */

static int clean_up_empty_block_removed;
static int clean_up_fused_sequences;
static int clean_up_1_statement_sequence;

hash_table statement_to_goto_table = NULL;

void
initialize_clean_up_sequences_statistics()
{
    clean_up_empty_block_removed = 0;
    clean_up_fused_sequences = 0;
    clean_up_1_statement_sequence = 0;
}


void
display_clean_up_sequences_statistics()
{
    if (get_bool_property("CLEAN_UP_SEQUENCES_DISPLAY_STATISTICS")
	&&
	(clean_up_empty_block_removed
	+ clean_up_fused_sequences
	+ clean_up_1_statement_sequence) != 0) {
	user_log("Statistics about cleaning up sequences:\n");
	if(clean_up_empty_block_removed)
	  user_log("\t%d empty sequence%s or useless CONTINUE removed.\n",
		 clean_up_empty_block_removed,
		 clean_up_empty_block_removed==1? "" : "s");
	if(clean_up_fused_sequences)
	  user_log("\t%d %s been fused.\n",
		   clean_up_fused_sequences,
		   clean_up_fused_sequences==1? "sequence has" : "sequences have");
	if(clean_up_1_statement_sequence)
	  user_log("\t%d sequence%s of only 1 statement have been removed.\n",
		 clean_up_1_statement_sequence,
		 clean_up_1_statement_sequence==1? "" : "s");
    }
}


static bool __attribute__ ((unused))
clean_up_sequences_filter(statement s)
{
  /* Just say to recurse... */
  /* gen_true() would be as good... and gcc would not complain. */
  /* And the maintenance would be easier. */
  pips_assert("", s==s);
  return TRUE;
}


/* Add a couple (statement -> GOTO) to the map statement_to_goto_table: */
bool
compute_statement_to_goto_table_filter(instruction i)
{
    if (instruction_goto_p(i)) {
	statement s = instruction_goto(i);

	pips_debug(7, "Adding GOTO from instruction %p to statement %p.\n",
		   i, s);

	if (!hash_defined_p(statement_to_goto_table, (char *) s)) {
	    hash_put(statement_to_goto_table,
		     (char *) s,
		     (char *) CONS(INSTRUCTION, i, NIL));
	}
	else {
	    hash_update(statement_to_goto_table,
			(char *) s,
			(char *) CONS(INSTRUCTION,
				      i,
				      (list) hash_get(statement_to_goto_table,
						      (char *) s)));
	}
	/* There is nothing to recurse into: */
	return FALSE;
    }
    return TRUE;
}


/* Since clean_up_sequences() is called before the controlizer, there
   may be some GOTO. GOTOs are in fact pointer to the target
   statement, hence when target statements are moved around, some
   GOTOs need to be updated. For this purpose, we build a list of
   GOTOs per target statement.
   */
void
compute_statement_to_goto_table(statement s)
{
    pips_assert("Map statement_to_goto_table should be uninitialized.\n",
		statement_to_goto_table == NULL);

    statement_to_goto_table = hash_table_make(hash_pointer, 0);
    gen_recurse(s, instruction_domain,
		compute_statement_to_goto_table_filter,
		gen_null);
}


/* Discard the statement_to_goto_table map: */
void
discard_statement_to_goto_table()
{
    HASH_MAP(k, v, {
	/* Discard every GOTO list: */
	gen_free_list((list) v);
    }, statement_to_goto_table);
    hash_table_free(statement_to_goto_table);
    statement_to_goto_table = NULL;
}


/* Adjust all the GOTOs pointing s1 to s2: */
void
adjust_goto_from_to(statement s1,
		    statement s2)
{
    if (s1 == s2)
	/* Nothing to do: */
	return;

    if (hash_defined_p(statement_to_goto_table, (char *) s1)) {
	MAP(INSTRUCTION, i, {
	    pips_assert("The GOTO should point to s1.\n",
			instruction_goto(i) == s1);
	    instruction_goto(i) = s2;
	    pips_debug(6, "Adjusting GOTO from instruction %p -> statement %p to statement %p.\n",
		       i, s1, s2);
	}, (list) hash_get(statement_to_goto_table, (char *) s1));
    }
}


static void
deal_with_pending_comment(statement s,
			  string * the_comments)
{
    if (*the_comments != NULL) {
	/* There is a pending comment to be added on this statement: */
	insert_comments_to_statement(s, *the_comments);
	free(*the_comments);
	*the_comments = NULL;
    }
}


void clean_up_sequences_rewrite(statement s)
{
  instruction i = statement_instruction(s);
  tag t = instruction_tag(i);
  switch(t) {
  case is_instruction_sequence:
    {
      /* Delete empty instruction and fuse blocks of instructions : */
      list useful_sts, delete_sts;
      string the_comments = NULL;
      list cst = list_undefined;

      useful_sts = delete_sts = NIL;

      if(!(statement_with_empty_comment_p(s)
	   && statement_number(s) == STATEMENT_NUMBER_UNDEFINED
	   && unlabelled_statement_p(s))) {
	print_statement(s);
	user_log("Statement %s\n"
		 "Number=%d, label=\"%s\", comment=\"%s\"\n",
		 statement_identification(s),
		 statement_number(s), label_local_name(statement_label(s)),
		 empty_comments_p(statement_comments(s))? "" : statement_comments(s));
	pips_internal_error("This block statement should be labelless, numberless"
		   " and commentless.\n");
      }

      /*
	pips_assert("This statement should be labelless, numberless and commentless.",
	statement_with_empty_comment_p(s)
	&& statement_number(s) == STATEMENT_NUMBER_UNDEFINED
	&& unlabelled_statement_p(s));
      */

      pips_debug(3, "A sequence of %zd statements with %zd declarations\n",
		 gen_length(sequence_statements(instruction_sequence(i))),
		 gen_length(statement_declarations(s)));
      ifdebug(5) {
	pips_debug(5,
		   "Statement at entry:\n");
	print_statement(s);
      }

      for(cst = sequence_statements(instruction_sequence(i)); !ENDP(cst); POP(cst)) {
	statement st = STATEMENT(CAR(cst));

	ifdebug(9) {
	  fprintf(stderr, "[ The current statement in the sequence: ]\n");
	  print_statement(st);
	}

	if (empty_statement_or_labelless_continue_p(st)) {
	  ifdebug(9)
	    fprintf(stderr, "[ Empty statement or continue ]\n");

	  if (!statement_with_empty_comment_p(st)) {
	    /* Keep the comment to put it on the next
	       useful statement: */
	    if (the_comments == NULL)
	      /* No comment has been gathered up to
		 now: */
	      the_comments = statement_comments(st);
	    else {
	      string new_comments =
		strdup(concatenate(the_comments,
				   statement_comments(st),
				   NULL));;
	      free(the_comments);
	      the_comments = new_comments;
	    }
	    statement_comments(st) = string_undefined;
	  }
	  /* Unused instructions can be deleted in any
	     order... :-) */
	  delete_sts = CONS(STATEMENT, st, delete_sts);
	  pips_debug(3, "Empty block removed...\n");
	  clean_up_empty_block_removed++;
	}
	else {
	  if (instruction_sequence_p(statement_instruction(st))
	      && ENDP(statement_declarations(st))) {
	    /* A sequence without declarations in a sequence: they can be fused: */
	    list statements = sequence_statements(instruction_sequence(statement_instruction(st)));
	    statement first = STATEMENT(CAR(statements));
	    /* Unlink the innermost sequence: */
	    sequence_statements(instruction_sequence(statement_instruction(st))) = NIL;
	    /* Keep the sequence: */
	    useful_sts = gen_nconc(useful_sts, statements);
	    pips_debug(3, "2 nested sequences fused...\n");
	    clean_up_fused_sequences++;
	    /* Think to delete the sequence parent: */
	    delete_sts = CONS(STATEMENT, st, delete_sts);
	    /* To deal with a pending comment: */
	    st = first;
	  }
	  else {
	    /* By default, it should be useful: */
	    useful_sts = gen_nconc(useful_sts,
				   CONS(STATEMENT, st, NIL));
	    pips_debug(4, "Statement useful... %zd\n",
		       gen_length(useful_sts));
	  }

	  deal_with_pending_comment(st, &the_comments);
	}
      }

      /* Remove the list of unused statements with the
	 statemenents them-self: */
      gen_full_free_list(delete_sts);

      if (the_comments != NULL /* || !ENDP(statement_declarations(s))*/) {
	/* We have a pending comment we were not able to
	   attach. Create a continue statement to attach it: */
	statement st = make_plain_continue_statement();
	deal_with_pending_comment(st, &the_comments);
	useful_sts = gen_nconc(useful_sts,
			       CONS(STATEMENT, st, NIL));
	pips_debug(3, "CONTINUE created to add a pending comment...\n");
      }

      /* Keep track of declarations within an empty sequence:
	 see C_syntax/block01.c */
      /*
	if(ENDP(useful_sts) && !ENDP(statement_declarations(s))) {
	useful_sts = CONS(STATEMENT, s, NIL); // FI: maybe I should copy s?
	}
      */

      /* Remove the old list of statements without the
	 statements: */
      gen_free_list(sequence_statements(instruction_sequence(i)));
      sequence_statements(instruction_sequence(i)) = useful_sts;

      if (gen_length(useful_sts) == 1
	  && ENDP(statement_declarations(s))) {
	/* A sequence of only 1 instruction can be replaced by
	   this instruction, if it has not declarations */
	statement st = STATEMENT(CAR(useful_sts));
	/* Transfer the deeper statement in the current one: */
	statement_label(s) = statement_label(st);
	statement_label(st) = entity_undefined;
	statement_number(s) = statement_number(st);
	statement_comments(s) = statement_comments(st);
	statement_comments(st) = string_undefined;
	statement_instruction(s) = statement_instruction(st);
	statement_instruction(st) = instruction_undefined;
	statement_declarations(s) = statement_declarations(st);
	statement_declarations(st) = NIL;
	statement_extensions(s) = statement_extensions(st);
	statement_extensions(st) = extensions_undefined;
	/* Do not forget to adjust the GOTOs pointing on st: */
	adjust_goto_from_to(st, s);
	/* Discard the old statement: */
	free_instruction(i);
	pips_debug(3, "Sequence with 1 statement replaced by 1 statement...\n");
	clean_up_1_statement_sequence++;
      }

      ifdebug(5) {
	pips_debug(5, "Statement at exit:\n");
	print_statement(s);
      }
      break;
    }
  }
}


/* An entry point for internal usage, such as from
   take_out_the_exit_node_if_not_a_continue(): */
void
clean_up_sequences_internal(statement s)
{
    debug_on("CLEAN_UP_SEQUENCES_DEBUG_LEVEL");
    ifdebug(1) {
      pips_debug(1, "Statement at entry:\n");
      print_statement(s);
    }
    compute_statement_to_goto_table(s);
    gen_recurse(s, statement_domain,
		gen_true,
		clean_up_sequences_rewrite);
    discard_statement_to_goto_table();
    ifdebug(1) {
      pips_debug(1, "Statement at exit:\n");
      print_statement(s);
    }
    debug_off();
}


/* Recursively clean up the statement sequences by fusing them if possible
   and by removing useless one. Remove also empty blocs and useless
   continues. */
void
clean_up_sequences(statement s)
{
  debug_on("CLEAN_UP_SEQUENCES_DEBUG_LEVEL");
  initialize_clean_up_sequences_statistics();
  clean_up_sequences_internal(s);
  display_clean_up_sequences_statistics();
  debug_off();
}
