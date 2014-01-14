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

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "linear.h"

#include "genC.h"
#include "text.h"
#include "text-util.h"
#include "ri.h"
#include "ri-util.h"
#include "misc.h"
#include "properties.h"
/* Used in word_to_attachments: */
typedef char * void_star;
#include "word_attachment.h"

#include "top-level.h"

/* To store the fact a prettyprinter ask for Emacs attachments: */
static bool is_emacs_pretty_print_asked;

enum {POSITION_UNDEFINED = -1};

#if 0
/* Mapping from an object to a name and from a name to an object: */
static hash_table names_of_almost_everything_in_a_module = NULL;
static hash_table names_to_almost_everything_in_a_module = NULL;

/* The unique (well, I hope so...) integer to name an object in PIPS
   from the extern world. It is not reset since it could be useful to
   name various instance of the same module for example. I hope that
   the 2^32 wrap around will not occur too quickly... :-( */
static int current_name_of_something = 1;
#endif

/* To store the attachment before sorting: */
list attachments_before_sorting = NIL;

/* Declare the various mapping between the words and attachments: */
GENERIC_LOCAL_FUNCTION(word_to_attachments_begin, word_to_attachments)
GENERIC_LOCAL_FUNCTION(word_to_attachments_end, word_to_attachments)


#if 0
bool
name_something(chunk * something)
{
    if (hash_defined_p(names_of_almost_everything_in_a_module,
		       (char *) something))
	/* Well, we have already named this object. Return the fact
           for gen_recurse: */
	return false;
    
    hash_put(names_of_almost_everything_in_a_module,
	     (char *) something,
	     (char *) current_name_of_something);
    
    hash_put(names_to_almost_everything_in_a_module,
	     (char *) current_name_of_something,
	     (char *) something);

    current_name_of_something ++;

    pips_assert("current_name_of_something should not wrap up to 0 !",
		current_name_of_something != 0);
    
    /* Go on in the RI: */
    return true;
}


void
name_almost_everything_in_a_module(statement module)
{
   pips_assert("names_of_almost_everything_in_a_module not NULL",
               names_of_almost_everything_in_a_module == NULL);
   pips_assert("names_to_almost_everything_in_a_module not NULL",
               names_to_almost_everything_in_a_module == NULL);

   names_of_almost_everything_in_a_module = hash_table_make(hash_pointer, 0);
   names_to_almost_everything_in_a_module = hash_table_make(hash_int, 0);
   
   gen_multi_recurse(module,
                     statement_domain, name_something, gen_null,
                     NULL);
}


void
free_names_of_almost_everything_in_a_module()
{
   hash_table_free(names_of_almost_everything_in_a_module);
   names_of_almost_everything_in_a_module = NULL;
   hash_table_free(names_to_almost_everything_in_a_module);
   names_to_almost_everything_in_a_module = NULL;
}
#endif


/* The translation functions between unique names and objects: */

/* Initialize some things related with the attachment of propertie. To
   be conservative, if a prettyprinter does not call
   begin_attachment_prettyprint(), all the attachment stuff is
   disable. Thus, old prettyprinter can go on without Emacs mode: */
void
begin_attachment_prettyprint()
{
    if (get_bool_property("PRETTYPRINT_ADD_EMACS_PROPERTIES")) {
	is_emacs_pretty_print_asked = true;
	/*word_text_attachment_mapping = hash_table_make(hash_pointer, 0);*/
	/* Initialize the local mapings: */
	init_word_to_attachments_begin();
	init_word_to_attachments_end();
	/*name_almost_everything_in_a_module(mod_stat);*/
    }
}


/* Clean the things related with the attachment of properties: */
void
end_attachment_prettyprint()
{
    if (is_emacs_pretty_print_asked) {
	is_emacs_pretty_print_asked = false;

	/* Strings in attachments should already have been freed by
	   print_sentence and attachement it-self (witout "s") by
	   output_an_attachment(): */
	close_word_to_attachments_begin();
	/* Should be OK since output_the_attachments_for_emacs() has
	   already unlinked the attachment from
	   ord_to_attachments_begin: */
	close_word_to_attachments_end();

	/*free_names_of_almost_everything_in_a_module();*/
    }
}


/* Use to create or extend the attachment list of a character: */
#define ADD_AN_ATTACHMENT_TO_A_MAPPING(a,			\
				       a_char,			\
				       a_mapping)		\
if (bound_##a_mapping##_p(a_char)) {				\
    attachments ats = load_##a_mapping(a_char);         	\
    list l = attachments_attachment(ats);			\
    l = CONS(ATTACHMENT, a, l);					\
    attachments_attachment(ats) = l;				\
}								\
else {								\
    store_##a_mapping(a_char,	        			\
		      make_attachments(CONS(ATTACHMENT,		\
					    a,			\
					    NIL)));		\
}


/* Attach something from begin_char up to end_char: */
static void
attach_to_character_region(char * begin_char,
			   char * end_char,
			   attachee at) 
{
    /* We do not know the position of the attachement in the output
       file yet: */
    attachment a = make_attachment(at,
				   POSITION_UNDEFINED,
				   POSITION_UNDEFINED);

    debug_on("ATTACHMENT_DEBUG_LEVEL");
    debug(6, "attach_to_character_region",
	  "Attach attachment %p (attachee %p, tag %d) from \"%c\" (%p) to \"%c\" (%p)\n",
	  a, at, attachee_tag(at),
	  *begin_char, begin_char,
	  *end_char, end_char);

    /* Attach the begin to the first character: */
    ADD_AN_ATTACHMENT_TO_A_MAPPING(a,
				   begin_char,
				   word_to_attachments_begin);

    /* Attach the end to the last character: */
    ADD_AN_ATTACHMENT_TO_A_MAPPING(a,
				   end_char,
				   word_to_attachments_end);
    debug_off();
}


/* Attach something to a word list, from begin_word to end_word: */
static void
attach_to_word_list(string begin_word,
		    string end_word,
		    attachee at) 
{
    debug_on("ATTACHMENT_DEBUG_LEVEL");
    debug(6, "attach_to_word_list",
	  "Attach attachee %p from \"%s\" (%p) to \"%s\" (%p)\n",
	  at,
	  begin_word, begin_word,
	  end_word, end_word);

    /* Attach from the first character of the first word up to the
       last character of the last word: */
    attach_to_character_region(begin_word,
			       end_word + strlen(end_word) - 1,
			       at);
    debug_off();
}


/* Attach something to all the words of the list given in argument: */
static void
attach_to_words(list l,
		attachee a)
{
    attach_to_word_list(STRING(CAR(l)),
			STRING(CAR(gen_last(l))),
			a);
}


/* Attach something to a sentence list: */
static void
attach_to_sentence_list(sentence begin,
			sentence end,
			attachee a)
{
    attach_to_word_list(first_word_of_sentence(begin),
			last_word_of_sentence(end),
			a);
}


/* Attach something to a sentence: */
static void
attach_to_sentence(sentence s,
		   attachee a)
{
    attach_to_sentence_list(s, s, a);
}


/* Attach something to a sentence up to the end of the given text: */
static void
attach_to_sentence_up_to_end_of_text(sentence s,
				    text t,
				    attachee a)
{
    attach_to_sentence_list(s, SENTENCE(CAR(gen_last(text_sentences(t)))), a);
}


/* Attach something to all a text: */
static void
attach_to_text(text t,
	       attachee a)
{
    attach_to_sentence_list(SENTENCE(CAR(text_sentences(t))),
			    SENTENCE(CAR(gen_last(text_sentences(t)))),
			    a);
}



/* The user interface: */

/* Attach a loop: */
void
attach_loop_to_sentence_up_to_end_of_text(sentence s,
				    text t,
				    loop l)
{
    if (is_emacs_pretty_print_asked)
	attach_to_sentence_up_to_end_of_text(s, t,
					     make_attachee(is_attachee_loop,
							   l));
}


/* Attach the PROGRAM/FUNCTION head: */
sentence
attach_head_to_sentence(sentence s,
			entity module)
{
    if (is_emacs_pretty_print_asked)
	attach_to_sentence(s, make_attachee(is_attachee_module_head, module));

    return s;
}


/* Attach a module usage (CALL or function call): */
void
attach_reference_to_word_list(string begin_word,
			      string end_word,
			      reference r)
{
    if (is_emacs_pretty_print_asked)
	attach_to_word_list(begin_word,
			    end_word,
			    make_attachee(is_attachee_reference, r));
}


/* Attach a reference: */
void
attach_regular_call_to_word(string word, call c)
{
  if (is_emacs_pretty_print_asked)
    attach_to_word_list(word,
			word,
			make_attachee(is_attachee_call, c));
}



/* Attach a declaration to all the words of the given list: */
void
attach_declaration_to_words(list l,
			    entity e)
{
    if (is_emacs_pretty_print_asked)
	attach_to_words(l,
			make_attachee(is_attachee_declaration, e));
}


/* Attach a declaration type to all the words of the given list. No
   need to use strdup(). May accept an empty list: */
void
attach_declaration_type_to_words(list l,
				 string declaration_type)
{
    if (is_emacs_pretty_print_asked)
	if (l != NIL)
	    attach_to_words(l,
			    make_attachee(is_attachee_type,
					  strdup(declaration_type)));
}


/* Attach a declaration type with its size to all the words of the
   given list. No need to use strdup(): */
void
attach_declaration_size_type_to_words(list l,
				      string declaration_type,
				      int size)
{
    if (is_emacs_pretty_print_asked) {
	char * size_char = i2a(size);
	attach_declaration_type_to_words(l,
					 concatenate(declaration_type,
						     "*",
						     size_char,
						     NIL));
	free(size_char);
    }
}


/* Attach some statement information to text: */
void
attach_statement_information_to_text(text t,
				     statement s)
{
    if (is_emacs_pretty_print_asked && text_sentences(t) != NIL)
	/* Some prettyprinters such as effects generate NULL
           text... Just ignore. */
	attach_to_text(t, make_attachee(is_attachee_statement_line_number,
					(void*) statement_number(s)));
}


/* Attach a decoration: */
void
attach_decoration_to_text(text t)
{
    if (is_emacs_pretty_print_asked && text_sentences(t) != NIL)
	/* Some prettyprinters such as effects generate NULL
           text... Just ignore. */
	attach_to_text(t, make_attachee(is_attachee_decoration, UU));
}


/* Attach a cumulated effects decoration: */
void
attach_cumulated_effects_decoration_to_text(text t)
{
    if (is_emacs_pretty_print_asked)
	attach_to_text(t, make_attachee(is_attachee_cumulated_effects, UU));
}


/* Attach a proper effects decoration: */
void
attach_proper_effects_decoration_to_text(text t)
{
    if (is_emacs_pretty_print_asked)
	attach_to_text(t, make_attachee(is_attachee_proper_effects, UU));
}


/* Attach a preconditions decoration: */
void
attach_preconditions_decoration_to_text(text t)
{
  if (is_emacs_pretty_print_asked)
    attach_to_text(t, make_attachee(is_attachee_preconditions, UU));
}

/* Attach a total preconditions decoration: */
void
attach_total_preconditions_decoration_to_text(text __attribute__ ((unused)) t)
{
  if (is_emacs_pretty_print_asked)
    pips_internal_error("not implemented yet");
}


/* Attach a transformers decoration: */
void
attach_transformers_decoration_to_text(text t)
{
  if (is_emacs_pretty_print_asked)
    attach_to_text(t, make_attachee(is_attachee_transformers, UU));
}


/* Try to find some attachments in the given character. If any, note
   the boundary position, that is a start or an end according to the
   functions given in parameters: */
static void
deal_with_attachment_boundary(char * a_character,
			      int position_in_the_output,
			      attachments (* load_word_to_attachments_boundary)(void_star),
			      bool (* bound_word_to_attachments_boundary_p)(void_star))
{
  debug(8, "deal_with_attachment_boundary",
	"Looking for \"%c\" (%p) at %d\n",
	*a_character,a_character, position_in_the_output);

  if (bound_word_to_attachments_boundary_p(a_character)) {
    /* Well, this word is an attachment boundary: */
    list some_attachments =
      attachments_attachment(load_word_to_attachments_boundary(a_character));
    MAP(ATTACHMENT, an_attachment,
    {
      debug(4, "deal_with_attachment_boundary",
	    "*** Found attachment = %p for \"%c\" (%p) at %d\n",
	    an_attachment, *a_character,
	    a_character, position_in_the_output);

      /* Remind the position of the boundary of the
	 attachment: */
      if (load_word_to_attachments_boundary
	  == load_word_to_attachments_begin)
	attachment_begin(an_attachment) = position_in_the_output;
      else
	attachment_end(an_attachment) = position_in_the_output;
    },
	some_attachments);
  };
}


/* Try to find some attachments in the given character that are printed out. If
   any, note the boundary position. */
void
deal_with_attachments_at_this_character(char * a_character,
					int position_in_the_output)
{
    if (is_emacs_pretty_print_asked) {
	debug_on("ATTACHMENT_DEBUG_LEVEL");

	/* Look for attachment starts: */
	debug(8, "deal_with_attachment_boundaries",
	      "Looking for attachment starts\n");
	deal_with_attachment_boundary(a_character,
				      position_in_the_output,
				      load_word_to_attachments_begin,
				      bound_word_to_attachments_begin_p);
	/* Look for attachment ends: */
	debug(8, "deal_with_attachment_boundaries",
	      "Looking for attachment ends\n");
	deal_with_attachment_boundary(a_character,
				      position_in_the_output,
				      load_word_to_attachments_end,
				      bound_word_to_attachments_end_p);    
	debug_off();
    }
}


/* Try to find some attachments in the given string. If any, note the
   boundary position. */
void
deal_with_attachments_in_this_string(string a_string,
				     int position_in_the_output)
{
    int i;
    
    for(i = 0; a_string[i] != '\0'; i++)
	deal_with_attachments_at_this_character(&a_string[i],
						position_in_the_output + i);
}


/* Try to find some attachments in the a_length first characters of
   the given string. If any, note the boundary position. */
void
deal_with_attachments_in_this_string_length(string a_string,
					    int position_in_the_output,
					    int a_length)
{
    int i;
    
    for(i = 0; i < a_length; i++) {
	pips_assert("Length is too big since end of string encountered",
		    a_string[i] != '\0');
	deal_with_attachments_at_this_character(&a_string[i],
						position_in_the_output + i);
    }
}


/* Many pretty-printers format their own pseudo-comment by their own
   and move in memory words that have attachments on them. To be able
   to track them up to the output, we need to overload the movement
   function to keep track of these. */
void
relocate_attachments(char * source,
		     char * new_position)
{
    int i;
    int source_length = strlen(source);

    debug_on("ATTACHMENT_DEBUG_LEVEL");

    pips_debug(5, "source = \"%s\" (%p), new_position = \"%s\" (%p)\n",
	       source, source,
	       new_position, new_position);
    
    for(i = 0; i < source_length; i++) {
	if (bound_word_to_attachments_begin_p(source + i)) {
	    pips_debug(4, "Relocating begin of attachment from %p to %p\n",
		       source + i,
		       new_position + i);
	    pips_assert("There is already an attachment on the target",
			!bound_word_to_attachments_begin_p(new_position + i));
	    store_word_to_attachments_begin(new_position + i,
					    load_word_to_attachments_begin(source + i));
	    delete_word_to_attachments_begin(source + i);
	}
	if (bound_word_to_attachments_end_p(source + i)) {
	    pips_debug(4, "Relocating end of attachment from %p to %p\n",
		       source + i,
		       new_position + i);
	    pips_assert("There is already an attachment on the target",
			!bound_word_to_attachments_end_p(new_position + i));
	    store_word_to_attachments_end(new_position + i,
					  load_word_to_attachments_end(source + i));
	    delete_word_to_attachments_end(source + i);
	}
    }
    debug_off();
}


/* Concatenate source to target and update the source attachments to
   point to the new location: */
char *
strcat_word_and_migrate_attachments(char * target,
				    const char * source)
{
    char * new_string;
    
    if (is_emacs_pretty_print_asked) {
	int target_length = strlen(target);

	/* The actual copy: */
	new_string = strcat(target, source);
	relocate_attachments((char*)source, target + target_length);
    }
    else
	/* The actual copy: */
	new_string = strcat(target, source);
	
    return new_string;
}


/* Duplicate a string and update the attachments to point to the new
   returned string: */
char *
strdup_and_migrate_attachments(char * a_string)
{
    char * new_string = strdup(a_string);
    
    if (new_string != NULL && is_emacs_pretty_print_asked)
	relocate_attachments(a_string, new_string);
    return new_string;
}


/* Output an attachment to the output file: */
static void
output_an_attachment(FILE * output_file,
		     attachment a)
{
    attachee at = attachment_attachee(a);
    int begin = attachment_begin(a);
    int end = attachment_end(a);
    
    pips_debug(5, "begin: %d, end: %d, attachment %p (attachee %p)\n",
	       begin, end, a, at);

    if (begin == POSITION_UNDEFINED || end == POSITION_UNDEFINED)
    {
	pips_user_warning("begin and end should be initialized.\n");
	return;
    }
		
    /* Begin an Emacs Lisp properties. + 1 since in a buffer (and not
       a string). (end + 1) + 1 since the property is set between
       start and end in Emacs.Use backquoting to evaluate keymaps
       later: */
    fprintf(output_file, "(add-text-properties %d %d `(", begin + 1, end + 2);

    switch(attachee_tag(at))
    {
    case is_attachee_reference:
	{	    
	    reference r = attachee_reference(at);
	    pips_debug(5, "\treference %p\n", r);
	    /* Evaluate the keymap: */
	    fprintf(output_file, "face epips-face-reference mouse-face epips-mouse-face-reference local-map ,epips-reference-keymap epips-property-reference \"%p\" epips-property-reference-variable \"%p\"",
		    r, reference_variable(r));
	    break;
	}
	
    case is_attachee_call:
	{	    
	    call c = attachee_call(at);
	    pips_debug(5, "\treference %p\n", c);
	    /* Evaluate the keymap: */
	    fprintf(output_file, "face epips-face-call mouse-face epips-mouse-face-call local-map ,epips-call-keymap epips-property-call \"%p\"",
		    c);
	    break;
	}
	
    case is_attachee_declaration:
	{	    
	    entity e = attachee_declaration(at);
	    pips_debug(5, "\tdeclaration %p\n", e);
	    /* Output the address as a string because Emacs cannot
               store 32-bit numbers: */
	    fprintf(output_file, "face epips-face-declaration epips-property-declaration \"%p\"",
		    e);
	    break;
	}      
	
    case is_attachee_type: 
	{	    
	    string s = attachee_type(at);
	    pips_debug(5, "\ttype \"%s\"\n", s);
	    fprintf(output_file, "epips-property-type \"%s\"", s);
	    break;
	}
	
    case is_attachee_loop: 
  	{	    
 	    loop l = attachee_loop(at);
 	    pips_debug(5, "\tloop %p\n", l);
 	    if (execution_parallel_p(loop_execution(l)))
 		fprintf(output_file, "face epips-face-parallel-loop ");
	    fprintf(output_file, "epips-property-loop \"%p\"", l);
  	    break;
  	}
	
    case is_attachee_module_head:
	{
	    entity head = attachee_module_head(at);
	    pips_debug(5, "\tmodule_head %p\n", head);
	    fprintf(output_file,
		    "face epips-face-module-head epips-module-head-name \"%s\"",
		    module_local_name(head));
	    break;
	}

    case is_attachee_decoration:
	{
	    pips_debug(5, "\tdecoration\n");
	    fprintf(output_file,
		    "invisible epips-invisible-decoration");
	    break;
	}

    case is_attachee_preconditions:
	{
	    pips_debug(5, "\tpreconditions\n");
	    fprintf(output_file,
		    "face epips-face-preconditions invisible epips-invisible-preconditions");
	    break;
	}

    case is_attachee_transformers:
	{
	    pips_debug(5, "\ttransformers\n");
	    fprintf(output_file,
		    "face epips-face-transformers invisible epips-invisible-transformers");
	    break;
	}
	
    case is_attachee_cumulated_effects:
	{
	    pips_debug(5, "\tcumulated-effect\n");
	    fprintf(output_file,
		    "face epips-face-cumulated-effect invisible epips-invisible-cumulated-effect");
	    break;
	}
	
    case is_attachee_proper_effects:
	{
	    pips_debug(5, "\tproper-effect\n");
	    fprintf(output_file,
		    "face epips-face-proper-effect invisible epips-invisible-proper-effect");
	    break;
	}

    case is_attachee_statement_line_number:
	{
	    int line_number = attachee_statement_line_number(at);

	    pips_debug(5, "\tstatement_line_number %d\n", line_number);
	    fprintf(output_file, "epips-line-number %d", line_number);
	    break;
	}

    default:
	pips_internal_error("attachee_tag inconsistent");
    }

    /* End an Emacs Lisp properties: */
    fprintf(output_file, "))\n");
}


/* The function used by qsort to compare 2 attachment structures: */
static int
compare_attachment_for_qsort(const void *xp,
			     const void *yp)
{
    attachment x = *(attachment *) xp;
    attachment y = *(attachment *) yp;

    if (attachment_begin(x) != attachment_begin(y))
	/* Sort according attachment x begins before y: */
	return attachment_begin(x) - attachment_begin(y);

    if (attachment_end(x) != attachment_end(y))
	/* Sort according attachment x ends after y: */
	return attachment_end(y) - attachment_begin(x);

    /* If they have the same range, sort according to the type: */
    return attachee_tag(attachment_attachee(x))
	- attachee_tag(attachment_attachee(y));
}


/* Output the attachment in a sorted order with less precise property
   first: */
static void
output_the_attachments_in_a_sorted_order(FILE * output_file)
{
    attachment * as;
    int i;
    
    int number_of_attachments = gen_length(attachments_before_sorting);
    
    as = (attachment *) malloc(number_of_attachments*sizeof(attachment));
    i = 0;
    MAP(ATTACHMENT, a, {
	as[i++] = a;
    }, attachments_before_sorting);

    qsort((char *)as,
	  number_of_attachments,
	  sizeof(attachment),
	  compare_attachment_for_qsort);

    for(i = 0; i < number_of_attachments; i++) {
	output_an_attachment(output_file, as[i]);
    }

    /* The attachments them self will be removed later: */
    gen_free_list(attachments_before_sorting);
    attachments_before_sorting = NIL;
}


/* Add the attachment to the intermediate list: */
static bool
put_an_attachment_in_the_list(attachment a)
{
    attachments_before_sorting = CONS(ATTACHMENT,
				      a,
				      attachments_before_sorting);

    /* We do not want to go on the recursion: */
    return false;
}


/* Nothing to do... */
static void
rewrite_an_attachment(attachment __attribute__ ((unused)) a)
{
    return;
}


/* Begin the Emacs Lisp file: */
static void
init_output_the_attachments_for_emacs(FILE * output_file)
{
    /* Begin a string with Emacs Lisp properties: */
    fprintf(output_file, "(insert \"");
}


/* Output the list of all the attachments found in the text file with
   Emacs Lisp syntax: */
static void
output_the_attachments_for_emacs(FILE * output_file)
{
    debug_on("ATTACHMENT_DEBUG_LEVEL");

    /* End the string part: */
    fprintf(output_file, "\")\n");

    /* Enumerate all the attachments: */
    gen_multi_recurse(get_word_to_attachments_begin(),
		      attachment_domain,
		      put_an_attachment_in_the_list,
		      rewrite_an_attachment,
		      NULL);

    /* Now we try to output the stuff in a decent order: */
    output_the_attachments_in_a_sorted_order(output_file);

    /* Just for fun, the previous gen_recurse will also recurse
       through the words of the mapping but it is not deep and thus it
       does not worth using a gen_multi_recurse with a string_domain
       and a filter returning always stop to avoid this unnecessary
       recursion. */

    WORD_TO_ATTACHMENTS_MAP(word_pointer, attachment_list,
    {
	pips_debug(6, "Key %p, value %p\n", word_pointer, attachment_list);
	/* Unlink the attachment from
	   attachments to avoid double free
	   later since referenced both by
	   word_to_attachments_begin and
	   word_to_attachments_end: */
	MAPL(ats,
	     {
		 ATTACHMENT_(CAR(ats)) = attachment_undefined;
	     },
	     attachments_attachment(attachment_list));
    },
			    get_word_to_attachments_begin());

    /* End the property part: */
    /* fprintf(output_file, "\n\t)\n)\n"); */

    debug_off();
}


    /* Add the attachment in Emacs mode by creating a twin file that
       is decorated with Emacs properties: */
void
write_an_attachment_file(string file_name)
{
    if (is_emacs_pretty_print_asked) {
	FILE * file_stream;
	char * emacs_file_name = strdup(concatenate(file_name, EMACS_FILE_EXT, NULL));
	FILE * emacs_file_stream = safe_fopen(emacs_file_name, "w");
	init_output_the_attachments_for_emacs(emacs_file_stream);
	/* Now include the original plain file: */
	file_stream = safe_fopen(file_name, "r");
	for(;;) {
	    char c = getc(file_stream);
	    /* Strange semantics: must have read the character first: */
	    if (feof(file_stream))
		break;
	    
	    /* Just backslashify the '"' and '\': */
	    if (c == '"' || c == '\\')
		(void) putc('\\', emacs_file_stream);
	    
	    (void) putc(c, emacs_file_stream);	  
	}
	safe_fclose(file_stream, file_name);
	/* Actually add the interesting stuff: */
	output_the_attachments_for_emacs(emacs_file_stream);
	safe_fclose(emacs_file_stream, emacs_file_name);
	free(emacs_file_name);
    }
}
