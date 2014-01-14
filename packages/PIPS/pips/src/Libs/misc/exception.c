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
#ifdef HAVE_CONFIG_H
    #include "pips_config.h"
#endif
/* Stack management for PIPS contexts
 *
 * A PIPS context contains a pointer in the PIPS debug level stack.
 * When a PIPS context is popped, the debug level stack is restored.
 * It may be the current level!
 *
 * Example: voir catch_user_error() in library pipmake, or a PIPS user
 * interface such as wpips, tpips or pips
 */

#include <stdio.h>
#include <genC.h>
#include "misc.h"

static stack debug_stack = NULL;

void push_pips_context(char const * file, char const * function, int line)
{
  pips_debug(9, "%s %s:%d\n", function, file, line);
  if (!debug_stack) debug_stack = stack_make(0, 50, 0);
  stack_push((void *) get_debug_stack_pointer(), debug_stack);
}

void pop_pips_context(char const * file, char const * function, int line)
{
  pips_debug(9, "%s %s:%d\n", function, file, line);
  if (!debug_stack) 
    pips_internal_error("unexpected pop without push %s %s:%d",
			function, file, line);
  set_debug_stack_pointer((_int) stack_pop(debug_stack));
}

/* That's all */
