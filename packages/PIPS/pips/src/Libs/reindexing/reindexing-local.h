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
/*#define INFINITY 2147483647*/ /* 2^31 - 1 */
#define INFINITY 32767 /* 2^15 - 1 */

#define RE_MODULE_NAME "REINDEX"

#define STAT_SYM "S"

#define IS_TEMP 1

#define IS_NEW_ARRAY 2

/* FI: moved into ri-util */
/*
#define MIN_OPERATOR_NAME "MIN"
#define MAX_OPERATOR_NAME "MAX"

#define ENTITY_MIN_P(e) (strcmp(entity_local_name(e), \
				MIN_OPERATOR_NAME) == 0)
#define ENTITY_MAX_P(e) (strcmp(entity_local_name(e), \
				MAX_OPERATOR_NAME) == 0)
#define ENTITY_MIN_OR_MAX_P(e) (ENTITY_MIN_P(e) || \
				 ENTITY_MAX_P(e) )
*/

#define IS_LOOP_BOUNDS 0
#define IS_ARRAY_BOUNDS 1

#define SAI "INS_"
#define SAT "TEMP_"
#define SA_MODULE_NAME "SA"
#define SPACE " "
#define COMA  ","
#define LEPA  "("
#define RIPA  ")"
