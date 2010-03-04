/*

  Copyright 1989-2010 HPC Project

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
  Pierre.Villalon@hpc-project.com
  Ronan.Keryell@hpc-project.com
*/
#ifdef HAVE_CONFIG_H
    #include "pips_config.h"
#endif

#include "linear.h"
#include "genC.h"
#include "misc.h"
#include "ri.h"
#include "ri-util.h"
#include "text-util.h"
#include "outlining_private.h"
#include "step_private.h"
#include "step.h"
#include "properties.h"

/*****************************************************A CONSTRUCTOR LIKE PART
 */

/** @return an empty extensions
 */
extensions empty_extensions (void) {
  return make_extensions (NIL);
}

/** @return TRUE if the extensions field is empty
    @param es the extensions to test
*/
bool empty_extensions_p (extensions es) {
  return (extensions_extension (es) == NIL);
}

/***************************************************** PRETTYPRINT PART
 */

/** @return a new allocated string to close the extension.
 *  @param es, the extension to be closed
 *
 *  Today we only generate omp parallel do pragma so the close is pretty
 *  easy. Later we will have to analyze the extension to generate the
 *  close string accordingly.
 */
string close_extension (extension e) {
  return close_pragma (extension_pragma(e));
}

/** @return a new allocated string to close the extensions.
 *  @param es, the extensions to be closed
 *  @param nl, set to TRUE to get the string with a final new line character
 */
string
close_extensions (extensions es, bool nl) {
  string s = string_undefined;

  if (empty_extensions_p (es) == FALSE) {
    /* Use a string_buffer for efficient string concatenation: */
    string_buffer sb = string_buffer_make(FALSE);

    list el = extensions_extension(es);
    FOREACH(EXTENSION, e, el) {
      s = close_extension(e);
      if (s != string_undefined) {
	string_buffer_append(sb, s);
	if (nl == TRUE) string_buffer_append(sb, strdup ("\n"));
	nl = TRUE;
      }
    }
    s = string_buffer_to_string(sb);
    /* Free the buffer with its strings: */
    string_buffer_free_all(&sb);
  }

  return s;
}


/** @return a new allocated string with the extension textual representation.
 */
string
extension_to_string(extension e) {
  string s;
  /* to be added later when extension can be something else than pragma
  switch (extension_tag(e)) {
  case is_extension_pragma:
  */
    s = pragma_to_string(extension_pragma(e));
    /*  default:
    pips_internal_error("Unknown extension type\n");
    }*/
  return s;
}

/** @brief return a new allocated string with the string representation of the
 *  extensions. Basically you'll get one extension per line
 *
 *  Assume that all the extension from the extensions (note the presence
 *  or not of the "s"...) are defined below.
 *
 *  @return string_undefined if es is extension_undefined, an malloc()ed
 *  textual string either.
 *  @param es, the extensions to translate to strings
 *  @param nl, set to TRUE to get the string with a final new line character
 */
string
extensions_to_string(extensions es, bool nl) {
  string s = string_undefined;

  /* Prettyprint in the correct language: */
  set_prettyprint_is_fortran_p(!get_bool_property("PRETTYPRINT_C_CODE"));

  if (empty_extensions_p (es) == FALSE) {
    /* Use a string_buffer for efficient string concatenation: */
    string_buffer sb = string_buffer_make(FALSE);

    list el = extensions_extension(es);
    FOREACH(EXTENSION, e, el) {
      s = extension_to_string(e);
      string_buffer_append(sb, s);
      if (nl == TRUE) string_buffer_append(sb, strdup ("\n"));
      nl = TRUE;
    }
    s = string_buffer_to_string(sb);
    /* Free the buffer with its strings: */
    string_buffer_free_all(&sb);
  }

  return s;
}
