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

#include "c_parser_private.h"
extern FILE * c_in; /* the file read in by the c_lexer */
extern int c_lineno ;
extern int c_lex();
extern int c_parse();


/* The labels in C have function scope... but beware of
   inlining... and conflict with user labels */
/* To disambiguate labels, in case inlining is performed later and to
   suppress the potential for conflicts with user labels.

   Temporary entities have to be generated to be replaced later by the
   final labels. The temporary entities should be eliminated from the
   symbol table...
 */
#define get_label_prefix() "-"  // a character that cannot be used in a correct

