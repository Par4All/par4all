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
