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
 /* include file for transformer library
  */

/* prefix used for value entity names; no conflict should occur with user
 * function names as long as they are restricted to 6 characters
 */
#define SEMANTICS_MODULE_NAME "*SEMANTICS*"

/* Must be used in suffixes and prefixes below */
#define SEMANTICS_SEPARATOR '#'

/* internal entity names (FI: I should have used suffixes to be consistent with external
 * suffixes */
#define OLD_VALUE_PREFIX "o#"
#define INTERMEDIATE_VALUE_PREFIX "i#"
#define TEMPORARY_VALUE_PREFIX "t#"

/* external suffixes (NEW_VALUE_SUFFIX is not used, new values are represented
 * by the variable itself, i.e. new value suffix is the empty string "")
 */
#define NEW_VALUE_SUFFIX "#new"
#define OLD_VALUE_SUFFIX "#init"
#define INTERMEDIATE_VALUE_SUFFIX "#int"

#define ADDRESS_OF_SUFFIX "#addressof"

#define SIZEOF_SUFFIX "#sizeof"
