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
/* Legal characters to start a comment line
 *
 * '\n' is added to cope with empty lines
 * Empty lines with SPACE and TAB characters 
 * are be preprocessed and reduced to an empty line by GetChar().
 */
#define START_COMMENT_LINE "CcDd*!#\n"

/* lex yacc interface */
extern FILE * syn_in; 
extern int syn_lex();
extern void syn_reset_lex();
extern int syn_parse();
extern void syn_error(const char*);

/* definition of implementation dependent constants */

#include "constants.h"

#define HASH_SIZE 1013
#define FORMATLENGTH (4096)
#define LOCAL static

#ifndef abs
#define abs(v) (((v) < 0) ? -(v) : (v))
#endif

/* extern char * getenv(); */

#define Warning(f,m) \
(user_warning(f,"Warning between lines %d and %d\n%s\n",line_b_I,line_e_I,m) )

#define FatalError(f,m) \
(pips_error(f,"Fatal error between lines %d and %d\n%s\n",line_b_I,line_e_I,m))

#include "parser_private.h"
