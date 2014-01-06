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
/* minimal a la java StringBuffer... */

#ifndef STRING_BUFFER_INCLUDED
#define STRING_BUFFER_INCLUDED

typedef struct __string_buffer_head * string_buffer;

// CONSTRUCTOR
string_buffer string_buffer_make(bool dup);
// DESTRUCTORS
void string_buffer_free(string_buffer *);
void string_buffer_free_all(string_buffer *);
// TEST
size_t string_buffer_size(const string_buffer);
bool string_buffer_empty_p(const string_buffer);
// OPERATIONS
void string_buffer_reset(string_buffer);
void string_buffer_append(string_buffer, const string);
void string_buffer_cat(string_buffer, const string, ...);
void string_buffer_append_sb(string_buffer, const string_buffer);
void string_buffer_append_list(string_buffer, const list);
void string_buffer_printf(string_buffer, const string, ...);
// CONVERSIONS
string string_buffer_to_string(const string_buffer);
string string_buffer_to_string_reverse(const string_buffer);
void string_buffer_to_file(const string_buffer, FILE *);
void string_buffer_append_c_string_buffer(const string_buffer, string_buffer, int);

#endif /* STRING_BUFFER_INCLUDED */
