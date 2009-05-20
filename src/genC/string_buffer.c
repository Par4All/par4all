/* $Id$

   Implement an a la java StringBuffer, a string-like object with
   efficient modification methods.

   The idea here is to speed-up concatenation of strings by keeping a
   stack of string and delaying the final build-up of the global string up
   to an explicit call to the string_buffer_to_string() method.

   In this way, if we have s strings of c characters, the concatenation
   complexity is in O(sc) with string_buffer_append() instead of O(s^2 c)
   with concatenate().

   Fabien Coelho, Mines ParisTech/CRI.
 */

#include <stdlib.h>
#include "genC.h"

/* internally defined structure.
 */
typedef struct __string_buffer_head
{
  stack ins;
}
  _string_buffer_head;

/* allocate a new string buffer
 */
string_buffer string_buffer_make(void)
{
  string_buffer n = (string_buffer) malloc(sizeof(_string_buffer_head));
  message_assert("allocated", n!=NULL);
  n->ins = stack_make(0, 0, 0);
  return n;
}

/* free string buffer structure
 * @arg free_strings also free string contents
 */
void string_buffer_free(string_buffer *psb, bool free_strings)
{
  if (free_strings)
    STACK_MAP_X(s, string, free(s), (*psb)->ins, 0);
  stack_free(&((*psb)->ins));
  free(*psb);
  *psb = NULL;
}

/* return malloc'ed string from string buffer sb
 */
string string_buffer_to_string(string_buffer sb)
{
  int bufsize = 0, current = 0;
  char * buf = NULL;

  STACK_MAP_X(s, string, bufsize+=strlen(s), sb->ins, 0);

  buf = (char*) malloc(sizeof(char)*(bufsize+1));
  message_assert("allocated", buf!=NULL);
  buf[current] = '\0';

  STACK_MAP_X(s, string,
  {
    int len = strlen(s);
    (void) memcpy(&buf[current], s, len);
    current += len;
    buf[current] = '\0';
  },
	      sb->ins, 0);

  return buf;
}

/* put string buffer into file.
 */
void string_buffer_to_file(string_buffer sb, FILE * out)
{
  STACK_MAP_X(s, string, fputs(s, out), sb->ins, 0);
}

/* append string s to string buffer sb, without duplication
 */
void string_buffer_append(string_buffer sb, string s)
{
  stack_push(s, sb->ins);
}

/* append string buffer sb2 to string buffer sb.
 */
void string_buffer_append_sb(string_buffer sb, string_buffer sb2)
{
  STACK_MAP_X(s, string, string_buffer_append(sb, s), sb2->ins, 0);
}

/* append string s to string buffer sb, with duplication
 */
void string_buffer_append_dup(string_buffer sb, string s)
{
  string_buffer_append_dup(sb, strdup(s));
}

/* append string buffer sb2 to string buffer sb with string duplication.
 */
void string_buffer_append_sb_dup(string_buffer sb, string_buffer sb2)
{
  STACK_MAP_X(s, string, string_buffer_append_dup(sb, s), sb2->ins, 0);
}
