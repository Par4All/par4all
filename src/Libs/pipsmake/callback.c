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
#ifdef HAVE_CONFIG_H
    #include "pips_config.h"
#endif
/*
 */

#include <stdio.h>
#include <stdlib.h>
#include "genC.h"

#include "database.h"
#include "makefile.h"
#include "pipsmake.h"

static pipsmake_callback_handler_type callback =
    (pipsmake_callback_handler_type) NULL ;
static bool callback_set_p = FALSE;

void set_pipsmake_callback(pipsmake_callback_handler_type p)
{
    message_assert("callback is already set", callback_set_p == FALSE);
    
    callback_set_p = TRUE;
    callback = p;
}

void reset_pipsmake_callback()
{
    message_assert("callback not set", callback_set_p == TRUE);
    
    callback_set_p = FALSE;
    callback = (pipsmake_callback_handler_type) NULL;
}

bool run_pipsmake_callback()
{
    bool result = TRUE;

    if (callback_set_p)
	result = (*callback)();

    return result;
}
