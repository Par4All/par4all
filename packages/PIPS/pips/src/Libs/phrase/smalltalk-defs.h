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
#ifndef SMALLTALK_DEFS
#define SMALLTALK_DEFS

#define EMPTY         ""
#define NL            "\n"
#define STSEMICOLON   "." NL
#define SPACE         " "

#define OPENBRACKET   "["
#define CLOSEBRACKET  "]"

#define OPENPAREN     "("
#define CLOSEPAREN    ")"

#define OPENBRACE     "{"
#define CLOSEBRACE    "}"

#define BEGINTEMPVAR  "|"
#define ENDTEMPVAR    "|"

#define SETVALUE      ":="
#define RETURNVALUE   "^"

#define COMMENT       "\""

#define ST_LE         "<="
#define ST_PLUS       "+"
#define ST_MINUS      "-"

#define ST_WHILETRUE  "whileTrue:"

#define ST_IFTRUE     "ifTrue:"
#define ST_IFFALSE    "ifFalse:"

#define ARRAY             "Array"
#define ARRAY_NEW         "new:"
#define ARRAY_AT          "at:"
#define ARRAY_AT_PUT_1    "at:"
#define ARRAY_AT_PUT_2    "put:"

#define ARRAY2D               "Array2D"
#define ARRAY2D_NEW1          "width:"
#define ARRAY2D_NEW2          "height:"
#define ARRAY2D_AT_AT_1       "at:"
#define ARRAY2D_AT_AT_2       "at:"
#define ARRAY2D_AT_AT_PUT_1   "at:"
#define ARRAY2D_AT_AT_PUT_2   "at:"
#define ARRAY2D_AT_AT_PUT_3   "put:"

#endif
