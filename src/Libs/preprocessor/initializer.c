/* stdio is necessary for misc.h */
#include <stdio.h>
#include <string.h>

#include "genC.h"
#include "misc.h"

/* there is no real rule to produce source or user files; it was introduced
 * by Remi to deal with source and user files as with any other kind
 * of resources
 */
void initializer(module_name)
string module_name;
{
    /* FI: strdup a cause de problemes lies aux varargs */
    user_error("initializer", "no source file for %s (%s might be an ENTRY point)\n",
	       strdup(module_name), strdup(module_name), 0);
}
