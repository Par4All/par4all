/*

  $Id$

  Copyright 1989-2014 MINES ParisTech

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
   stack of strings and delaying the final build-up of the global string up
   to an explicit call to the string_buffer_to_string() method.

   In this way, if we have s strings of c characters, the concatenation
   complexity is in O(sc) with string_buffer_append() instead of O(s^2 c)
   with concatenate().

   Fabien Coelho
*/
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <stdlib.h>
#include "genC.h"

/* internally defined structure.
 * the string_buffer pointer type is defined in "newgen_string_buffer.h"
 */
struct __string_buffer_head
{
  stack ins;
  bool dup; // whether to duplicate all strings appended to the buffer.
  // we could keep track of the current buffer size on the fly:
  // size_t size;
};

/* allocate a new string buffer
 * @param dup tell whether to string duplicate appended strings
 * if so, the strings will be freed later.
 */
string_buffer string_buffer_make(bool dup)
{
  string_buffer n =
    (string_buffer) malloc(sizeof(struct __string_buffer_head));
  message_assert("string_buffer is allocated", n!=NULL);
  n->ins = stack_make(0, 0, 0);
  n->dup = dup;
  return n;
}

/* remove stack contents
 */
void string_buffer_reset(string_buffer sb)
{
  while (!stack_empty_p(sb->ins)) {
    string s = (string) stack_pop(sb->ins);
    if (sb->dup) free(s);
  }
}

/* @brief free string buffer structure, also free string contents
 * according to the dup field
 * @arg psb the string_buffer to free
 */
void string_buffer_free(string_buffer *psb)
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
  message_assert("not null pointer", (*psb) != NULL);
  (*psb)->dup = true;
  string_buffer_free (psb);
}

/* return the size of the string in string_buffer sb
 */
size_t string_buffer_size(const string_buffer sb)
{
  size_t size = 0;
  STACK_MAP_X(s, string, size+=strlen(s), sb->ins, 0);
  return size;
}

/* return whether string_buffer sb is empty.
 */
bool string_buffer_empty_p(const string_buffer sb)
{
  return string_buffer_size(sb)==0;
}

/* convert to a malloced string, maybe in rev-ersed order of the appends
 */
static string
  string_buffer_to_string_internal(const string_buffer sb, bool rev)
{
  string buf = (string) malloc(string_buffer_size(sb)+1);
  message_assert("allocated", buf!=NULL);

  int current = 0;
  buf[current] = '\0';

  STACK_MAP_X(s, string,
  {
    int len = strlen(s);
    (void) memcpy(&buf[current], s, len);
    current += len;
    buf[current] = '\0';
  },
	      sb->ins, rev);

  return buf;
}
/* return malloc'ed string from string buffer sb
 */
string string_buffer_to_string(const string_buffer sb)
{
  return string_buffer_to_string_internal(sb, false);
}

/* return malloc'ed string from string buffer sb going from bottom to top
 */
string string_buffer_to_string_reverse (const string_buffer sb)
{
  return string_buffer_to_string_internal(sb, true);
}

/* put string buffer into file.
 */
void string_buffer_to_file(const string_buffer sb, FILE * out)
{
  STACK_MAP_X(s, string, fputs(s, out), sb->ins, 0);
}

#define BUFFER_SIZE 512

/* put string buffer as a C-string definition of the string buffer...
 */
void string_buffer_append_c_string_buffer(
  string_buffer sb, const string_buffer src, int indent)
{
  message_assert("distinct string buffers", sb!=src);
  // internal buffer...
  char buffer[BUFFER_SIZE];
  int i = 0, j;
  // start string
  buffer[i++] = '"';
  bool in_string = true;
  STACK_MAP_X(s, string,
    for (char * c = s; *c; c++)
    {
      if (i>BUFFER_SIZE-10-indent)
      {
        buffer[i++] = '\0';
        string_buffer_append(sb, buffer);
        i = 0;
      }
      if (!in_string)
      {
        buffer[i++] = '\n';
        for (j=0; j<indent; j++)
          buffer[i++] = ' ';
        buffer[i++] = '"';
        in_string = true;
      }
      switch (*c)
      {
      case '\n': // New line
        buffer[i++] = '\\';
        buffer[i++] = 'n';
        // the string is closed on newlines anyway
        buffer[i++] = '"';
        in_string = false;
        break;
      case '\a': // Audible bell
        buffer[i++] = '\\';
        buffer[i++] = 'a';
        break;
      case '\b': // Backspace
        buffer[i++] = '\\';
        buffer[i++] = 'b';
        break;
      case '\f': // Form feed
        buffer[i++] = '\\';
        buffer[i++] = 'f';
        break;
      case '\r': // carriage Return
        buffer[i++] = '\\';
        buffer[i++] = 'r';
        break;
      case '\t': // horizontal tab
        buffer[i++] = '\\';
        buffer[i++] = 't';
        break;
      case '\v': // Vertical tab
        buffer[i++] = '\\';
        buffer[i++] = 'v';
        break;
      case '\0': // null character
        buffer[i++] = '\\';
        buffer[i++] = '0';
        break;
      case '\\': // backslash
        buffer[i++] = '\\';
        buffer[i++] = '\\';
        break;
      case '"': // double quote
        buffer[i++] = '\\';
        buffer[i++] = '"';
        break;
      default:
        buffer[i++] = *c;
      }
    },
              src->ins, 0);
  // end string if needed
  if (in_string)
    buffer[i++] = '"';
  buffer[i++] = '\0';
  string_buffer_append(sb, buffer);
}

/* append string s (if non empty) to string buffer sb, the duplication
 * is done if needed according to the dup field.
 */
void string_buffer_append(string_buffer sb, const string s)
{
  if (*s) stack_push(sb->dup? strdup(s): s, sb->ins);
}

/* @brief append the string buffer sb2 to string buffer sb.
 * @return void
 * @param sb, the string buffer where to append the second string buffer
 * @param sb2, the string buffer to append to the fisrt string buffer
 */
void string_buffer_append_sb(string_buffer sb, const string_buffer sb2)
{
  STACK_MAP_X(s, string, string_buffer_append(sb, s), sb2->ins, 0);
}

/* @brief append a list of string to a string buffer. Note that each element
 * of the list is duplicated or not according to the dup field.
 * @return void
 * @param sb, the string buffer where to append the whole list
 * @param l, the list of string to append to the string buffer
 */
void string_buffer_append_list(string_buffer sb, const list l)
{
  FOREACH (string, s, l)
    string_buffer_append(sb, s);
}

#include <stdarg.h>

/* append a NULL terminated list of string to sb.
 * @param sb string buffer to be appended to
 * @param first... appended strings
 */
void string_buffer_cat(string_buffer sb, const string first, ...)
{
  va_list args;
  va_start(args, first);
  string next = first;
  while (next)
  {
    string_buffer_append(sb, next);
    next = va_arg(args, string);
  }
  va_end(args);
}
/* append a formatted string to sb
 * @param sb string buffer to be appended to
 * @param format printf format string
 * @param ... values
 */
void string_buffer_printf(string_buffer sb, const string format, ...)
{
  message_assert("duplicate string_buffer", sb->dup);
  va_list args;
  va_start(args, format);
  string str;
  int err = vasprintf(&str, format, args);
  message_assert("vasprintf ok", err >= 0);
  string_buffer_append(sb, str);
  va_end(args);
}
