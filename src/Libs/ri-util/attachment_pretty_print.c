#include <stdlib.h>
#include <stdio.h>

#include "genC.h"
#include "ri.h"
#include "word_attachment.h"

/* By default, do not pretty print with emacs (no properties attached
   to the output text code, etc.) */
bool is_emacs_prettyprint = FALSE;
bool is_attachment_prettyprint = FALSE;


/* The hash table to associate an Emacs property list to some output
   text: */
static hash_table word_text_attachment_mapping;


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
                                                    CONS(ATTACHMENT,
                                                         an_attachment,
                                                         NIL));
   }
}


