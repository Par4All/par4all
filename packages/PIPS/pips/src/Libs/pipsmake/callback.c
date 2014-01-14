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
/*
 */

#include <stdio.h>
#include <stdlib.h>
#include "genC.h"
#include "linear.h"
#include "ri.h"

#include "database.h"
#include "pipsmake.h"

static pipsmake_callback_handler_type callback =
    (pipsmake_callback_handler_type) NULL ;
static bool callback_set_p = false;

void set_pipsmake_callback(pipsmake_callback_handler_type p)
{
    message_assert("callback is already set", callback_set_p == false);
    
    callback_set_p = true;
    callback = p;
}

void reset_pipsmake_callback()
{
    message_assert("callback not set", callback_set_p == true);
    
    callback_set_p = false;
    callback = (pipsmake_callback_handler_type) NULL;
}

bool run_pipsmake_callback()
{
    bool result = true;

    if (callback_set_p)
	result = (*callback)();

    return result;
}
