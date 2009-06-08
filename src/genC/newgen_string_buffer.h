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
/* minimal a la java StringBuffer... */

#ifndef STRING_BUFFER_INCLUDED
#define STRING_BUFFER_INCLUDED

struct __string_buffer_head;
typedef struct __string_buffer_head * string_buffer;

string_buffer string_buffer_make(bool dup);
void string_buffer_free(string_buffer *);
void string_buffer_free_all(string_buffer *);
string string_buffer_to_string(string_buffer);
string string_buffer_to_string_reverse(string_buffer);
void string_buffer_to_file(string_buffer, FILE *);
void string_buffer_append(string_buffer, string);
void string_buffer_cat(string_buffer, string, ...);
void string_buffer_append_sb(string_buffer, string_buffer);
void string_buffer_append_list(string_buffer, list);

#endif /* STRING_BUFFER_INCLUDED */
