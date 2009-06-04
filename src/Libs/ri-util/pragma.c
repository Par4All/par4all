/*

  $Id$

  Copyright 1989-2009 MINES ParisTech

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
/*
  $Id$

  This file define methods to deal with objects extensions and pragma
  used as extensions to statements in the PIPS internal representation.

  A middle term, extensions method could go in another file.

  It is a trivial inplementation based on strings for a proof of concept.

  Pierre.Villalon@hpc-project.com
  Ronan.Keryell@hpc-project.com
*/

#include "linear.h"
#include "genC.h"
#include "ri.h"

///@return an empty extensions
extensions empty_extensions (void) {
  return make_extensions (NIL);
}

///@return TRUE if the extensions field is empty
///@param es the extensions to test
bool empty_extensions_p (extensions es) {
  return (extensions_extension (es) == NIL);
}

/** Return a new allocated string with the pragma textual representation.

 */
string
pragma_to_string (pragma p) {
  string s = pragma_string(p);
  if (s != string_undefined)
    s = strdup(s);
  return s;
}

/** Return a new allocated string with the extension textual representation.

 */
string
extension_to_string(extension e) {
  string s;
  /* For later other extensions:
  switch (extension_tag(e)) {
  case is_extension_pragma:
  */
  s = pragma_to_string(extension_pragma(e));
  /*
  default:
    pips_internal_error("Unknown extension type\n");
  }
  */
  return s;
}

/** Return a new allocated string with the textual representation of the
    extensions.

    Assume that all the extension from the extensions (note the presence
    or not of the "s"...) are defined below.

    @return string_undefined if es is extension_undefined, an malloc()ed
    textual string either.
 */
string
extensions_to_string(extensions es) {
  string s = string_undefined;

  if (empty_extensions_p (es) == FALSE) {
    /* Use a string_buffer for efficient string concatenation: */
    string_buffer sb = string_buffer_make(FALSE);

    list el = extensions_extension(es);
    FOREACH(EXTENSION, e, el) {
      s = extension_to_string(e);
      string_buffer_append(sb, s);
    }
    s = string_buffer_to_string(sb);
    /* Free the buffer with its strings: */
    string_buffer_free_all(&sb);
  }

  return s;
}


/** Add a pragma to a statement.

    @param stat, the statement which we want to add a pragma
    @param s, the string pragma.
    @param copy_flag, to be set to true to duplicate the string
 */
void
add_pragma_to_statement(statement st, string s, bool copy_flg) {
  extensions es = statement_extensions(st);
  /* Make a new pragma: */
  pragma p = pragma_undefined;
  if (copy_flg == TRUE) p = make_pragma(strdup(s));
  else p = make_pragma(s);
  extension e = make_extension(p);
  /* Add the new pragma to the extension list: */
  list el = extensions_extension(es);
  el = gen_extension_cons(e, el);
  extensions_extension(es) = el;
  /* Update the statement in case there was to extension first: */
  statement_extensions(st) = es;
}
