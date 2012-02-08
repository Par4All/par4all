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
#include "pipsdbm.h"
#include "resources.h"


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

/* Clean up sequences context */

typedef struct {
  list l_ref; // references of the outermost statement
  list l_loops; // loops of the outermost statement
  list l_decl; // declarations of the outermost statement
  bool merge_sequences_with_declarations;
} cusq_context;

static bool
store_reference(reference ref, cusq_context * p_ctxt)
{
  p_ctxt->l_ref = CONS(REFERENCE, ref, p_ctxt->l_ref);
  return true;
}

static bool
store_loops(loop l, cusq_context * p_ctxt)
{
  p_ctxt->l_loops = CONS(LOOP, l,  p_ctxt->l_loops);
  return true;
}

static void
cusq_ctxt_init(statement s, cusq_context * p_ctxt)
{
  p_ctxt->merge_sequences_with_declarations = get_bool_property("CLEAN_UP_SEQUENCES_WITH_DECLARATIONS");
  p_ctxt->l_ref = NIL;
  p_ctxt->l_loops = NIL;
  p_ctxt->l_decl = NIL;

  if (p_ctxt->merge_sequences_with_declarations)
    {
      gen_context_multi_recurse(s, p_ctxt,
				reference_domain, store_reference, gen_null,
				loop_domain, store_loops, gen_null,
				NULL);
      p_ctxt->l_decl = statement_to_declarations(s);// gen_copy_seq(statement_declarations(s));
    }

  ifdebug(1)
    {
      pips_debug(1, "outermost declarations:");
      print_entities(p_ctxt->l_decl);
      fprintf(stderr, "\n");
      pips_debug(1, "%s merge sequences with declarations\n", p_ctxt->merge_sequences_with_declarations? "":"do not");
    }
}

static void
cusq_ctxt_reset(cusq_context * p_ctxt)
{
  gen_free_list(p_ctxt->l_loops);
  p_ctxt->l_loops = NIL;

  gen_free_list(p_ctxt->l_ref);
  p_ctxt->l_ref = NIL;

  gen_free_list(p_ctxt->l_decl);
  p_ctxt->l_decl = NIL;
}

/**
 *  replace entities in the module references and loop indices and loop locals
 *  stored in the input cusq_context using the input old/new entities mapping
 */
static void
replace_entities_in_cusq_context(cusq_context * p_ctxt, hash_table ht)
{
  FOREACH(REFERENCE, ref, p_ctxt->l_ref)
    {
      entity new_ent = hash_get(ht, reference_variable(ref));
      if (new_ent !=  HASH_UNDEFINED_VALUE)
	reference_variable(ref) = new_ent;
    }

  FOREACH(LOOP, l, p_ctxt->l_loops)
    {
      entity new_index = hash_get(ht, loop_index(l));
      if (new_index != entity_undefined)
	loop_index(l) = new_index;

      for(list l_locals = loop_locals(l); !ENDP(l_locals); POP(l_locals))
	{
	  entity new_local = hash_get(ht, ENTITY(CAR(l_locals)));
	  if (new_local != entity_undefined)
	    CAR(l_locals).p = (gen_chunkp) new_local;
	}
    }
}


/* End: clean up sequences context */

static bool __attribute__ ((unused))
clean_up_sequences_filter(statement s)
{
  /* Just say to recurse... */
  /* gen_true() would be as good... and gcc would not complain. */
  /* And the maintenance would be easier. */
  pips_assert("", s==s);
  return true;
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
	return false;
    }
    return true;
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

/**
 * change the declarations of declaration statements directly
 * contained in input sequence according to the input old/new entity mapping
 *
 * @param seq is the input sequence
 * @param renamings is an old/new entity mapping
 *
 */
static void
sequence_proper_declarations_rename_in_place(sequence seq, hash_table renamings)
{
  FOREACH(STATEMENT, stmt, sequence_statements(seq))
    {
      if (declaration_statement_p(stmt))
	{
	  ifdebug(4) {
	    pips_debug(4,
		       "current declaration statement:\n");
	    print_statement(stmt);
	  }

	  list l_decl = statement_declarations(stmt);
	  while (!ENDP(l_decl))
	    {
	      entity decl = ENTITY(CAR(l_decl));
	      entity new_decl = hash_get(renamings, decl);
	      if(new_decl != HASH_UNDEFINED_VALUE) // we have found the old declaration
		{
		  pips_debug(5, "replacing entity %s by %s\n", entity_name(decl), entity_name(new_decl));
		  CAR(l_decl).p = (gen_chunkp) new_decl;
		  hash_del(renamings, decl);
		}
	      POP(l_decl);
	    }
	}
    }
}


static void
clean_up_sequences_rewrite(statement s, cusq_context * p_ctxt)
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

      list l_new_decls = NIL; // newly declared entities if need be.
      entity module = get_current_module_entity();

      FOREACH(STATEMENT, st, sequence_statements(instruction_sequence(i))) {

	ifdebug(8) {
	  fprintf(stderr, "[ The current statement in the sequence: ]\n");
	  print_statement(st);
	}

	if (empty_statement_or_labelless_continue_p(st)) {
	  ifdebug(8)
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
	  // take care of declarations from the statements directly in the internal sequence
	  list l_internal_proper_decls = statement_to_direct_declarations(st);

	  ifdebug(4){
	    pips_debug(4, "l_internal_proper_decls = \n");
	    print_entities(l_internal_proper_decls);
	    fprintf(stderr, "\n");
	    pips_debug(4, "internal statement declarations");
	    print_entities(statement_declarations(st));
	    fprintf(stderr, "\n");
	  }

	  if (statement_sequence_p(st)
	      && !empty_statement_p(st)
	      // The next two tests should be useless
	      // The sequence can be cleaned up even when declarations
	      // and extensions are present
	      // FI: the second part of this test is awful but the
	      // controlizer seems to rely on it... See flatten_code01
	      && ( p_ctxt->merge_sequences_with_declarations || ENDP(l_internal_proper_decls) || ENDP(statement_declarations(st)))
	      && ENDP(extensions_extension(statement_extensions(st)))) {

	    /* A sequence with or without declarations in a sequence: they can be fused: */
	    sequence internal_sequence = statement_sequence(st);
	    list l_stmt_to_be_merged = sequence_statements(internal_sequence);

	    if (p_ctxt->merge_sequences_with_declarations && !ENDP(l_internal_proper_decls))
	      {
		hash_table renamings = hash_table_make(hash_pointer, HASH_DEFAULT_SIZE);
		FOREACH(ENTITY, decl, l_internal_proper_decls)
		  {
		    pips_debug(3, "dealing with declaration: %s\n", entity_name(decl));

		    // check if there is already a variable with the same user name
		    // used in the outermost sequence
		    const char * decl_user_name = entity_user_name(decl);
		    bool parent_decl_with_same_name = false;
		    FOREACH(ENTITY, parent_decl, p_ctxt->l_decl)
		      {
			// arghh, this is costly, we should at least memoize user_names...
			if (parent_decl != decl
			    && strcmp(entity_user_name(parent_decl), decl_user_name) == 0)
			  {
			    parent_decl_with_same_name = true;
			    break;
			  }
		      }

		    // find another non-conflicting name and take care of scope
		    bool top_level_entity_with_same_name_p = (FindEntity(TOP_LEVEL_MODULE_NAME, decl_user_name) != entity_undefined);
		    string decl_name = entity_name(decl);
		    string decl_without_module_and_scope = strrchr(decl_name, BLOCK_SEP_CHAR);
		    string new_name = string_undefined;
		    entity new_decl = entity_undefined;

		    pips_debug(6, "decl name without module+scope: %s\n", decl_without_module_and_scope);

		    if (decl_without_module_and_scope ==NULL)
		      {
			// this should generate an error; however, names are sometimes
			// generated by PIPS phases without scope
			pips_debug(6, "non consistant variable scope\n");
			if (top_level_entity_with_same_name_p)
			  new_decl = make_entity_copy_with_new_name_and_suffix(decl, decl_name, true);
		      }
		    else if (decl_without_module_and_scope[-1]=='0' && decl_without_module_and_scope[-2]== MODULE_SEP_STRING[0])
		      {
			pips_debug(6, "Entity already at the uppermost scope level\n");
			if (top_level_entity_with_same_name_p)
			  new_decl = make_entity_copy_with_new_name_and_suffix(decl, decl_name, true);
		      }
		    else
		      {
			// duplicate the scope, minus the last scope
			string module_and_new_scope = string_undefined;

			int i = 2;
			while (decl_without_module_and_scope[-i] != BLOCK_SEP_CHAR)
			  {
			    if (decl_without_module_and_scope[-i] == MODULE_SEP_STRING[0])
			       pips_internal_error("unexpected MODULE_SEP_STRING, no BLOCK_SEP_CHAR FOUND\n");
			    i++;
			  }
			if (decl_without_module_and_scope[-i] == BLOCK_SEP_CHAR)
			   module_and_new_scope = strndup(decl_name, decl_without_module_and_scope-decl_name-i+1);

			pips_debug(6, "module+scope of new decl: %s\n", module_and_new_scope);
			new_name = strdup(concatenate(module_and_new_scope, decl_user_name, NULL));
			free(module_and_new_scope);

			pips_debug(6, "new decl name: %s\n", new_name);

			if (parent_decl_with_same_name || top_level_entity_with_same_name_p )
			  new_decl = make_entity_copy_with_new_name_and_suffix(decl, new_name, true);
			else
			  new_decl = make_entity_copy_with_new_name(decl, new_name, true);

			pips_debug(3, "new declaration: %s\n", entity_name(new_decl));
			hash_put(renamings, decl, new_decl); // add the correspondance in renamings table for further actual renaming
			l_new_decls = CONS(ENTITY, new_decl, l_new_decls);
		      }

		    if (!entity_undefined_p(new_decl))
		      {
			// after the merge, the new entity is used in the outermost statement:
			p_ctxt->l_decl = CONS(ENTITY, new_decl, p_ctxt->l_decl);
			// the old entity is not used anymore in the module,  in the current statement and in the outermost statement
			gen_remove(&entity_declarations(module), decl);
			gen_remove(&statement_declarations(s), decl);
			gen_remove(&p_ctxt->l_decl, decl);
		      }
		  } // FOREACH(ENTITY, decl, l_internal_proper_decls)

		// replace the old entity by the new one in the references of the outermost statement
		// as well as the loop indices and loop locals
		replace_entities_in_cusq_context(p_ctxt, renamings); // ! does not change the declarations themselves
		// change the declarations in place with new names
		sequence_proper_declarations_rename_in_place(internal_sequence, renamings);
		ifdebug(4) {
		  pips_debug(4, "stmt after declaration changes:\n");
		  print_statement(st);
		}
		hash_table_free(renamings);
	      } // if (p_ctxt->merge_sequences_with_declarations && !ENDP(l_internal_proper_decls))

	    statement first = STATEMENT(CAR(l_stmt_to_be_merged));
	    /* Unlink the innermost sequence: */
	    sequence_statements(internal_sequence) = NIL;
	    /* Keep the sequence: */
	    useful_sts = gen_nconc(useful_sts, l_stmt_to_be_merged);

	    pips_debug(3, "2 nested sequences fused...\n");
	    clean_up_fused_sequences++;
	    /* Think to delete the sequence parent: */
	    delete_sts = CONS(STATEMENT, st, delete_sts);
	    /* To deal with a pending comment: */
	    st = first;

	  } // if (statement_sequence_p(st) && !empty_statement_p(st)...)
	  else {
	    /* By default, it should be useful: */
	    useful_sts = gen_nconc(useful_sts,
				   CONS(STATEMENT, st, NIL));
	    pips_debug(4, "Statement useful... %zd\n",
		       gen_length(useful_sts));
	  }
	  deal_with_pending_comment(st, &the_comments);
	  // free the spine of the list of internal declarations
	  gen_free_list(l_internal_proper_decls);
	}
      } // FOREACH(STATEMENT, st, sequence_statements(instruction_sequence(i)))

      /* Remove the list of unused statements with the
	 statements themselves: */
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

      // for consistency: add new entities to module and current_statement declarations
      if (p_ctxt->merge_sequences_with_declarations)
	{
	  FOREACH(entity, decl, l_new_decls)
	    {
	      pips_debug(3, "adding new declaration for %s\n", entity_name(decl));
	      AddLocalEntityToDeclarationsOnly(decl, module, s);
	    }
	}

      if (gen_length(useful_sts) == 1
	  && ENDP(statement_declarations(s))
	  && ENDP(extensions_extension(statement_extensions(s)))) {
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

    cusq_context ctxt;
    cusq_ctxt_init(s, &ctxt);

    gen_context_multi_recurse(s, &ctxt,
			      statement_domain, gen_true, clean_up_sequences_rewrite,
			      NULL);
    cusq_ctxt_reset(&ctxt);
    discard_statement_to_goto_table();
    ifdebug(1) {
      pips_debug(1, "Statement at exit:\n");
      print_statement(s);
      statement_consistent_p(s);
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
