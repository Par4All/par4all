#include <stdio.h>
extern int fprintf();
#include <string.h>
#include <sys/types.h>

#include "genC.h"
#include "list.h"
#include "database.h"
#include "makefile.h"
#include "ri.h"
#include "pipsdbm.h"
#include "pipsmake.h"

#include "misc.h"


bool active_phase_p(phase)
string phase;
{
    makefile current_makefile = parse_makefile();

	MAPL(pa, {
	    string s = STRING(CAR(pa));
	    
	    if (strcmp(s, phase) == 0)
		return (TRUE);
	}, makefile_active_phases(current_makefile));
    return (FALSE);
}


void fprint_activated(fd)
FILE *fd;
{
    makefile m = parse_makefile();

    MAPL(pa, {
	string s = STRING(CAR(pa));

	fprintf(fd, "%s\n", s);

    }, makefile_active_phases(m));
}


string activate(phase)
string phase;
{
    rule r;
    virtual_resource res;
    string vrn;
    string old_phase;
    makefile current_makefile = parse_makefile();

    /* find rule that describes phase */
    r = find_rule_by_phase(phase);
    if(r == rule_undefined) {
	user_warning( "activate", "Rule `%s' undefined\n", phase);
	return(NULL);
    }

    /* complete simple cases */
    if (active_phase_p(phase))
	return (phase);
    if (gen_length(rule_produced(r))!=1)
	if (gen_length(rule_produced(r))>1){
		pips_error("activate", 
			   "Phase %s : rule that produces more than one resource : inconsistency risks \n", phase);
	}
	else
	    pips_error("activate", 
		       "Phase %s produces no resource\n", phase);

    /* find resource res that is produced by phase */
    res = VIRTUAL_RESOURCE(CAR(rule_produced(r)));
    vrn = virtual_resource_name(res);

    /* Check if selected phase produces a resource it uses */
    MAPL(pvr, {
        virtual_resource 	vr = VIRTUAL_RESOURCE(CAR(pvr));
        string 			vrn2 = virtual_resource_name(vr);
        owner 			vro = virtual_resource_owner(vr);

        /* We do not check callers and callees */
        if ( owner_callers_p(vro) || owner_callees_p(vro) ) {}

        /* We can't select a phase that uses a resource it produces */
        else if (strcmp(vrn, vrn2) == 0) {
            pips_error("activate",
                      "Selected phase %s requires a resource it produces\n",
                      phase);
        }
        else debug(9, "activate",
             "OK : Required resource %s is not produced by selected phase %s\n",
	      vrn2, phase);
    }, (list) rule_required( r ) );


    /* find current active phase old_phase that produces res */
    old_phase = rule_phase(find_rule_by_resource(vrn));

    /* replace old_phase by phase in active phase list */
    if (old_phase != NULL) {

	MAPL(pa, {
	    string s = STRING(CAR(pa));
	    
	    if (strcmp(s, old_phase) == 0) {
		free(STRING(CAR(pa)));
		STRING(CAR(pa)) = strdup(phase);
	    }
	}, makefile_active_phases(current_makefile));
    }

    if (db_get_current_program() != database_undefined) {
	/* remove resources with the same name as res to maintain 
	   consistency in the database */
	db_unput_resources(vrn);
    }

    return (phase);
}



