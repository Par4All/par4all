/*

  $Id$

  Copyright 1989-2009 MINES ParisTech

  This file is part of NewGen.

  NewGen is free software: you can redistribute it and/or modify it under the
  terms of the GNU General Public License as published by the Free Software
  Foundation, either version 3 of the License, or any later version.

  NewGen is distributed in the hope that it will be useful, but WITHOUT ANY
  WARRANTY; without even the implied warranty of MERCHANTABILITY or
  FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
  License for more details.

  You should have received a copy of the GNU General Public License along with
  NewGen.  If not, see <http://www.gnu.org/licenses/>.

*/
/*
   Implement an a la java StringBuffer, a string-like object with
   efficient modification methods.

   The idea here is to speed-up concatenation of strings by keeping a
   stack of string and delaying the final build-up of the global string up
   to an explicit call to the string_buffer_to_string() method.

   In this way, if we have s strings of c characters, the concatenation
   complexity is in O(sc) with string_buffer_append() instead of O(s^2 c)
   with concatenate().

   Fabien Coelho
*/

#include <stdlib.h>
#include "genC.h"

/* internally defined structure.
 */
typedef struct __string_buffer_head
{
  stack ins;
  boolean dup; // whether to duplicate strings
}
  _string_buffer_head;

/* allocate a new string buffer
 */
string_buffer string_buffer_make (bool dup)
{
  string_buffer n = (string_buffer) malloc(sizeof(_string_buffer_head));
  message_assert("allocated", n!=NULL);
  n->ins = stack_make(0, 0, 0);
  n->dup = dup;
  return n;
}

/* @brief free string buffer structure, also free string contents
 * according to the dup field
 * @arg psb the string_buffer to free
 */
void string_buffer_free (string_buffer *psb)
{
  if ((*psb)->dup)
    STACK_MAP_X(s, string, free(s), (*psb)->ins, 0);
  stack_free(&((*psb)->ins));
  free(*psb);
  *psb = NULL;
}

/* free string buffer structure and force string freeing
 * @arg psb the string_buffer to free
 */
void string_buffer_free_all (string_buffer *psb)
{
  message_assert("null pointer", (*psb) != NULL);
  (*psb)->dup = TRUE;
  string_buffer_free (psb);
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

/* return malloc'ed string from string buffer sb going from bottom to top
 */
string string_buffer_to_string_reverse (string_buffer sb)
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
	      sb->ins, 1);

  return buf;
}

/* put string buffer into file.
 */
void string_buffer_to_file(string_buffer sb, FILE * out)
{
  STACK_MAP_X(s, string, fputs(s, out), sb->ins, 0);
}

/* append string s to string buffer sb, the duplication
 * is done if needed according to the dup field.
 */
void string_buffer_append(string_buffer sb, string s)
{
  stack_push(sb->dup? strdup(s): s, sb->ins);
}

/* @brief append the string buffer sb2 to string buffer sb.
 * @return void
 * @param sb, the string buffer where to append the second string buffer
 * @param sb2, the string buffer to append to the fisrt string buffer
 */
void string_buffer_append_sb(string_buffer sb, string_buffer sb2)
{
  STACK_MAP_X(s, string, string_buffer_append(sb, s), sb2->ins, 0);
}

/* @brief append a list of string to a string buffer. Note that each element
 * of the list is duplicated or not according to the dup field.
 * @return void
 * @param sb, the string buffer where to append the whole list
 * @param l, the list of string to append to the string buffer
 * @param flg, set to TRUE if the list need to be added in the revere order
 */
void string_buffer_append_list(string_buffer sb, list l, bool flg)
{
  if (flg == TRUE) l = gen_nreverse (l);
  FOREACH (STRING, s, l) {
    string_buffer_append(sb, s);
  }
}

#include <stdarg.h>

/* append a NULL terminated list of string to sb.
 */
void string_buffer_cat(string_buffer sb, string next, ...)
{
  va_list args;
  va_start(args, next);
  while (next)
  {
    string_buffer_append(sb, next);
    next = va_arg(args, string);
  }
  va_end(args);
}
