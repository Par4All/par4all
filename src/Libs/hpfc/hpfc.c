/*
 * HPFC module by Fabien COELHO
 *
 * SCCS stuff:
 * $RCSfile: hpfc.c,v $ ($Date: 1995/03/13 11:56:22 $, ) version $Revision$,
 * got on %D%, %T%
 */
 
#include <stdio.h>
#include <string.h> 
extern fprintf();
extern system();

#include "boolean.h"
#include "vecteur.h"
#include "contrainte.h"
#include "sc.h"

#include "genC.h"

#include "ri.h" 
#include "database.h"
#include "hpf.h" 
#include "hpf_private.h"

#include "ri-util.h" 
#include "misc.h" 
#include "regions.h"
#include "semantics.h"
#include "effects.h"
#include "properties.h"
#include "resources.h"
#include "pipsdbm.h"

#include "hpfc.h"
#include "defines-local.h"

#define NO_FILE "no file name"

/* the source code is transformed with hpfc_directives
 * into something that can be parsed with a standard f77 compiler.
 */
void hpfc_directives_filter(name)
string name;
{
    string file_name = db_get_resource(DBR_SOURCE_FILE, name, TRUE);

    debug_on("HPFC_DEBUG_LEVEL");
    debug(1, "hpfc_directives_filter", "considering module %s\n", name);

    system(concatenate("mv ", file_name, " ", file_name, "- ; ",
		       "$HPFC_TOOLS/hpfc_directives", 
		       " < ", file_name, "-", 
		       " > ", file_name, " ;",
		       NULL));

    /*  I put some fake file as a created resource
     */
    DB_PUT_FILE_RESOURCE(DBR_HPFC_FILTERED, strdup(name), NO_FILE);

    debug_off();
}

void hpfc_init(name)
string name;
{
    /* struct DirectiveHandler *x = handlers; */
    /* entity e; */

    debug_on("HPFC_DEBUG_LEVEL");
    debug(1, "hpfc_init", "considering workspace %s\n", name);
    debug(1, "hpfc_init", "not implemented yet\n");

    /*   hpfc special entities are created as instrinsics...
     */
    /*
    for(; x->name!=(string) 0; x++)
    {
	e = FindOrCreateEntity(TOP_LEVEL_MODULE_NAME, x->name);
    }
    */

    /*  I put some fake file as a created resource
     */ 
   DB_PUT_FILE_RESOURCE(DBR_HPFC_STATUS, strdup(name), NO_FILE);

    debug_off();
}

void hpfc_directives(name)
string name;
{
    statement s = (statement) db_get_resource(DBR_CODE, name, FALSE);

    debug_on("HPFC_DEBUG_LEVEL");
    debug(1, "hpfc_directives", "considering module %s\n", name);

    handle_hpf_directives(s);

    /*  I put some fake file as a created resource
     */
    DB_PUT_FILE_RESOURCE(DBR_HPFC_DIRECTIVES, name, NO_FILE);
    DB_PUT_MEMORY_RESOURCE(DBR_CODE, name, s);

    debug_off();
}

/* should compile MODULE name.
 */
void hpfc_compile(name)
string name;
{
    debug_on("HPFC_DEBUG_LEVEL");
    debug(1, "hpfc_compile", "considering module %s\n", name);
    debug(1, "hpfc_compile", "not implemented yet\n");
    debug_off();
}

/* close the hpf compiler execution.
 * must deal with the commons, which are global to the program.
 */
void hpfc_close(name)
string name;
{
    debug_on("HPFC_DEBUG_LEVEL");
    debug(1, "hpfc_close", "considering %s\n", name);
    debug(1, "hpfc_close", "not implemented yet\n");
    debug_off();
}

/* that's all
 */
