/* Clean up the sequences and their contents by fusing thme or
   removing useless continues or empty instructions.

   Ronan.Keryell@cri.ensmp.fr

   */

/* 	%A% ($Date: 1998/03/07 21:54:24 $, ) version $Revision$, got on %D%, %T% [%P%].\n Copyright (c) École des Mines de Paris Proprietary.	 */

#ifndef lint
char vcid_clean_up_sequences[] = "%A% ($Date: 1998/03/07 21:54:24 $, ) version $Revision$, got on %D%, %T% [%P%].\n Copyright (c) École des Mines de Paris Proprietary.";
#endif /* lint */


#include <stdlib.h> 
#include <stdio.h> 
#include <malloc.h> 
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


static bool
clean_up_sequences_filter(statement s)
{
    /* Just say to recurse... */
    return TRUE;
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


void
clean_up_sequences_rewrite(statement s)
{
    instruction i = statement_instruction(s);
    tag t = instruction_tag(i);
    switch(t) {
    case is_instruction_sequence:
	{
	    /* Delete empty instruction and fuse blocks of instructions : */
	    list useful_sts, delete_sts;
	    string the_comments = NULL;
	    useful_sts = delete_sts = NIL;

	    if(!(statement_with_empty_comment_p(s)
		 && statement_number(s) == STATEMENT_NUMBER_UNDEFINED
		 && unlabelled_statement_p(s))) {
		user_log("Statement %s\n"
			 "Number=%d, label=\"%s\", comment=\"%s\"\n",
			 statement_identification(s),
			 statement_number(s), label_local_name(statement_label(s)),
			 statement_comments(s));
		pips_error("text_statement", "This block statement should be labelless, numberless"
	
			   " and commentless.\n");
		}

	    /*
	    pips_assert("This statement should be labelless, numberless and commentless.",
			statement_with_empty_comment_p(s)
			&& statement_number(s) == STATEMENT_NUMBER_UNDEFINED
			&& unlabelled_statement_p(s));
			*/

	    pips_debug(3, "A sequence of %d statements\n",
		       gen_length(sequence_statements(instruction_sequence(i))));
	    ifdebug(5) {
		pips_debug(5,
		       "Statement at entry:\n");
		print_statement(s);
	    }

	    MAP(STATEMENT, st,
		{
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
			if (instruction_sequence_p(statement_instruction(st))) {
			    /* A sequence in a sequence: they can be fused: */
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
			    pips_debug(4, "Statement useful... %d\n",
				       gen_length(useful_sts));
			}

			deal_with_pending_comment(st, &the_comments);
		    }
		},
		sequence_statements(instruction_sequence(i)));
	    
	    /* Remove the list of unused statements with the
               statemenents them-self: */
	    gen_full_free_list(delete_sts);

	    if (the_comments != NULL) {
		/* We have a pending comment we were not able to
                   attach. Create a continue statement to attach it: */
		statement st = make_continue_statement(entity_empty_label());
		deal_with_pending_comment(st, &the_comments);
		useful_sts = gen_nconc(useful_sts,
				       CONS(STATEMENT, st, NIL));
		pips_debug(3, "CONTINUE created to add a pending comment...\n");
	    }

	    /* Remove the old list of statements without the
               statements: */
	    gen_free_list(sequence_statements(instruction_sequence(i)));
	    sequence_statements(instruction_sequence(i)) = useful_sts;
       
	    if (gen_length(useful_sts) == 1) {
		/* A sequence of only 1 instruction can be replaced by
                   this instruction: */
		statement st = STATEMENT(CAR(useful_sts));
		/* Transfer the deeper statement in the current one: */
		statement_label(s) = statement_label(st);
		statement_label(st) = entity_undefined;
		statement_number(s) = statement_number(st);
		statement_comments(s) = statement_comments(st);
		statement_comments(st) = string_undefined;
		statement_instruction(s) = statement_instruction(st);
		statement_instruction(st) = instruction_undefined;
		/* Discard the old statement: */
		free_instruction(i);
		pips_debug(3, "Sequence with 1 statement replaced by 1 statement...\n");
		clean_up_1_statement_sequence++;
	    }
	    
	    ifdebug(5) {
		pips_debug(5,
			   "Statement at exit:\n");
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
      gen_recurse(s, statement_domain,
		clean_up_sequences_filter,
		clean_up_sequences_rewrite);  
}
/* Recursively clean up the statement sequences by fusing if possible
   and by removing useless one. Remove also epty blocs and useless
   continue. */
void
clean_up_sequences(statement s)
{
    debug_on("CLEAN_UP_SEQUENCES_DEBUG_LEVEL");
    initialize_clean_up_sequences_statistics();
    clean_up_sequences_internal(s);
    display_clean_up_sequences_statistics();
    debug_off();
}
