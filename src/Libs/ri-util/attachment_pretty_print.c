#include <stdlib.h>
#include <stdio.h>

#include "genC.h"
#include "ri.h"
#include "misc.h"
#include "word_attachment.h"

/* By default, do not pretty print with emacs (no properties attached
   to the output text code, etc.) */
bool is_emacs_prettyprint = FALSE;
bool is_attachment_prettyprint = FALSE;


/* The hash table to associate an Emacs property list to some output
   text: */
static hash_table word_text_attachment_mapping;


/* Mapping from an object to a name and from a name to an object: */
static hash_table names_of_almost_everything_in_a_module = NULL;
static hash_table names_to_almost_everything_in_a_module = NULL;


/* The unique (well, I hope so...) integer to name an object in PIPS
   from the extern world. It is not reset since it could be useful to
   name various instance of the same module for example. I hope that
   the 2^32 wrap around will not occur too quickly... :-( */
static int current_name_of_something = 0;

bool
name_something(chunk * something)
{
   hash_put(names_of_almost_everything_in_a_module,
            (char *) something,
            (char *) current_name_of_something);

   hash_put(names_to_almost_everything_in_a_module,
            (char *) current_name_of_something,
            (char *) something);

   current_name_of_something ++;
   
   /* Go on in the RI: */
   return TRUE;
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


/* Initialize some things related with the attachment of properties: */
void
begin_attachment_prettyprint()
{
   if (is_attachment_prettyprint) {
      word_text_attachment_mapping = hash_table_make(hash_pointer, 0);
   }
}


/* Clean the things related with the attachment of properties: */
void
end_attachment_prettyprint()
{
   if (is_attachment_prettyprint) {
      hash_table_free(word_text_attachment_mapping);
   }
}


/* Attach with an hash table a property list to a pointer (assumed a
   pointer to a text word): */
static void
attach_a_property_list_to_a_word_text_pointer(void * a_pointer,
                                              attachments * an_attachment_list)
{
   if (is_attachment_prettyprint) {
      hash_put(word_text_attachment_mapping,
               (char *) a_pointer,
               (char *) an_attachment_list);
   }
}


/* Attach with an hash table a property to a pointer (assumed a
   pointer to a text word): */
static void
attach_a_property_to_a_word_text_pointer(void * a_pointer,
                                         attachment * an_attachment)
{
   if (is_attachment_prettyprint) {
      attach_a_property_list_to_a_word_text_pointer(a_pointer,
                                                    make_attachments(CONS(ATTACHMENT,
                                                                          an_attachment,
                                                                          NIL)));
   }
}


