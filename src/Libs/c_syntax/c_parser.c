/* $id$
   $Log: c_parser.c,v $
   Revision 1.1  2003/06/24 07:23:20  nguyen
   Initial revision
 */

#include <stdlib.h>
#include <stdio.h>
#include <malloc.h>
#include <string.h>

#include "genC.h"
#include "linear.h"
#include "misc.h"
#include "ri.h"
#include "ri-util.h"

#include "c_syntax.h"

#include "resources.h"
#include "database.h"
#include "makefile.h"

#include "pipsdbm.h"

/* name of the current file */
char * file_name = NULL;

bool c_parser(string module)
{
    string dir = db_get_current_workspace_directory();
 
    debug_on("C_SYNTAX_DEBUG_LEVEL");

    file_name = strdup(concatenate(dir,"/",db_get_file_resource(DBR_SOURCE_FILE,module,TRUE),0));
    free(dir);

    /* yacc parser is called */
    c_in = safe_fopen(file_name, "r");
    c_parse();
    safe_fclose(c_in, file_name);
    free(file_name);
    file_name = NULL;

    debug_off();

    return TRUE;
}


