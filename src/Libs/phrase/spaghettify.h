/*

  $Id$

  Copyright 1989-2010 MINES ParisTech

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
#ifndef SPAGHETTIFY_DEFS
#define SPAGHETTIFY_DEFS

#define INDEX_VARIABLE_NAME "DO%d_INDEX"
#define BEGIN_VARIABLE_NAME "DO%d_BEGIN"
#define END_VARIABLE_NAME "DO%d_END"
#define INCREMENT_VARIABLE_NAME "DO%d_INCREMENT"

statement spaghettify_loop (statement stat, string module_name);

statement spaghettify_whileloop (statement stat, string module_name);

statement spaghettify_forloop (statement stat, string module_name);

statement spaghettify_test (statement stat, string module_name);

statement spaghettify_statement (statement stat, string module_name);

#endif