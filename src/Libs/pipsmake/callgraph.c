#include <stdio.h>
extern int fprintf();
#include <string.h>

#include "genC.h"
#include "misc.h"
#include "ri.h"
#include "database.h"
#include "pipsdbm.h"
#include "resources.h"

/* callgraph compute the caller list of each module, using the callees list
 * of them
 *
 * Note: although callgraph is directly and internally used by pipsmake
 * which needs to know about the callers of a module to chain its rules,
 * it might be better to store it in a future (new) callgraph library (FI)
 *
 * Argument "name" is not used. It is instantiated as a specific module
 * by make() but this routine process the whole program
 */
void callgraph(name)
string name;
{
    int nmodules = 0;
    char *module_list[ARGS_LENGTH];
    callees module_callers[ARGS_LENGTH];
    int i;

    db_get_module_list(&nmodules, module_list);
    pips_assert("callgraph", nmodules>0);

    for(i=0; i<nmodules; i++) {
	module_callers[i] = make_callees(NIL);
    }

    for(i=0; i<nmodules; i++) {
	pips_assert("callgraph",
		    callees_callees(module_callers[i])==NIL);
    }

    for(i=0; i<nmodules; i++) {
	string module_name = module_list[i];
	callees module_callees = callees_undefined;
	list cm;

	module_callees = (callees)
	    db_get_memory_resource(DBR_CALLEES, module_name, TRUE);

	for(cm=callees_callees(module_callees); cm!=NIL; POP(cm)) {
	    string module_called = STRING(CAR(cm));
	    int r;

	    for(r=0; r<nmodules; r++) {
		if(strcmp(module_called, module_list[r])==0)
		    break;
	    }
	    if(r==nmodules)
		user_error("callgraph", "no source file for module %s\n", module_called);
	    pips_assert("callgraph",0<=r && r<nmodules);

	    callees_callees(module_callers[r]) =
		gen_nconc(callees_callees(module_callers[r]), 
			  CONS(STRING,strdup(module_name), NIL));
	}
    }
    
    for(i=0; i<nmodules; i++) {
	string module_name = module_list[i];

	DB_PUT_MEMORY_RESOURCE(DBR_CALLERS,
			       strdup(module_name), 
			       (char*) module_callers[i]);

    }
}
