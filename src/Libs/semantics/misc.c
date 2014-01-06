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
 /* low level routines which should be added to other packages
  *
  * Francois Irigoin, April 1990
  *
  * Modifications:
  *  - passage a l'utilisation de database
  */

#include <stdio.h>

#include "genC.h"
#include "misc.h"
#include "linear.h"
#include "ri.h"
#include "effects.h"
#include "ri-util.h"
#include "effects-util.h"
#include "semantics.h"

/* probably to add to the future callgraph library */

int call_site_count(m)
entity m;
{
    /* number of "call to m" sites within the current program;
       they are all in m's CALLERS */
    pips_assert("call_site_count", value_code_p(entity_initial(m)));
    /* I do not know yet; let's return 1 to please the semantic analysis */
    user_warning("call_site_count",
		 "not implemented yet, always returns 1\n");
    return 1;
}

int caller_count(m)
entity m;
{
    /* number of modules calling m within the current program;
       i.e. number of modules containing at least one call site to m */
    pips_assert("caller_count", value_code_p(entity_initial(m)));
    /* I do not know yet; let's return 1 to please the semantic analysis */
    user_warning("caller_count",
		 "not implemented yet, always returns 1\n");
    return 1;
}

int dynamic_call_count(m)
entity m;
{
    /* number of call to m during the current program execution;
       return 0 if m is never called, either because it's a call
       graph root or because it was linked by mistake;
       return -1 if the dynamic call count is unknow, for instance
       because one of m's call site is located in a loop of unknown
       bounds;
       return k when it can be evaluated */
    pips_assert("dynamic_call_count", value_code_p(entity_initial(m)));
    /* I do not know yet; let's return 1 to please the semantic analysis */
    user_warning("dynamic_call_count",
		 "not implemented yet, always returns -1\n");
    return -1;
}
