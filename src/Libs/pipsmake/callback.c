#include <stdio.h>
#include "genC.h"

typedef bool (*pipsmake_callback_handler_type)();

static pipsmake_callback_handler_type callback = (pipsmake_callback_handler_type) NULL ;
static bool set = FALSE;

void set_pipsmake_callback(pipsmake_callback_handler_type p)
{
    message_assert("callback is already set", set == FALSE);
    
    set = TRUE;
    callback = p;
}

void reset_pipsmake_callback()
{
    message_assert("callback not set", set == TRUE);
    
    set = FALSE;
    callback = (pipsmake_callback_handler_type) NULL 
}

bool run_pipsmake_callback()
{
    bool result = TRUE;

    if (set)
	result = (*callback)();

    return result;
}
