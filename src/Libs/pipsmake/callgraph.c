#include <stdio.h>
#include <string.h>

#include "genC.h"
#include "misc.h"
#include "ri.h"
#include "database.h"

#include "ri-util.h"
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
bool callgraph(string name)
{
    gen_array_t modules = db_get_module_list();
    int nmodules = gen_array_nitems(modules), i;
    gen_array_t module_callers = gen_array_make(nmodules);

    pips_assert("some modules", nmodules>0);

    for(i=0; i<nmodules; i++)
	gen_array_addto(module_callers, i, (char*) make_callees(NIL));

    for(i=0; i<nmodules; i++) {
	callees c = (callees)gen_array_item(module_callers, i);
	pips_assert("no callees", callees_callees(c)==NIL);
    }

    for(i=0; i<nmodules; i++) {
	string module_name = gen_array_item(modules, i);
	callees module_callees = callees_undefined;
	list cm;

	module_callees = (callees)
	    db_get_memory_resource(DBR_CALLEES, module_name, TRUE);

	for(cm=callees_callees(module_callees); cm!=NIL; POP(cm))
	{
	    string module_called = STRING(CAR(cm));
	    callees c;
	    int r;

	    for(r=0; r<nmodules; r++) {
		string rname = gen_array_item(modules, r);
		if(strcmp(module_called, rname)==0)
		    break;
	    }
	    if(r==nmodules)
		user_error("callgraph",
			   "no source file for module %s\n",
			   module_called);
	    pips_assert("valid module",0<=r && r<nmodules);

	    c = (callees) gen_array_item(module_callers, r);
	    callees_callees(c) =
		gen_nconc(callees_callees(c), 
			  CONS(STRING,strdup(module_name), NIL));
	}
    }
    
    for(i=0; i<nmodules; i++) 
    {
	string module_name = gen_array_item(modules, i);
	DB_PUT_MEMORY_RESOURCE(DBR_CALLERS, module_name,
			       (char*) gen_array_item(module_callers,i));

    }
    return TRUE;
}
