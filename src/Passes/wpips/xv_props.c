#include <stdio.h>
extern char *getenv();
extern int sscanf();

#include <sys/time.h>
#include <sys/resource.h>
#include <xview/xview.h>
#include <xview/panel.h>

#include "genC.h"
#include "ri.h"
#include "graph.h"
#include "makefile.h"
#include "database.h"

#include "misc.h"
#include "ri-util.h"
#include "pipsdbm.h"
#include "pipsmake.h"
#include "complexity_ri.h"

#include "constants.h"
#include "resources.h"
#include "properties.h"

#include "wpips.h"



static Menu smenu_props;

static hash_table aliases;



/* returns the first key which value is svp. when not found, returns NULL */
string hash_get_key_by_value(htp, svp)
hash_table htp;
string svp;
{
    HASH_MAP(kp, vp, {
	if (strcmp(vp, svp) == 0)
	    return kp;
    }, htp);

    return HASH_UNDEFINED_VALUE;
}

void update_props()
{
    string res_alias_n, res_true_n ,phase_alias_n ,phase_true_n;
    Menu menu_props, special_prop_m;
    Menu_item props_mi, special_prop_mi;
    int i, j;

    debug_on("WPIPS_DEBUG_LEVEL");

    menu_props = smenu_props;

    /* walk through items of menu_props */
    for (i=(int)xv_get(menu_props, MENU_NITEMS); i>0; i--) {

	/* find resource corresponding to the item */
	props_mi = (Menu_item) xv_get(menu_props, MENU_NTH_ITEM, i);
	res_alias_n = (string) xv_get(props_mi, MENU_STRING);
	if( (res_true_n = (string) hash_get(aliases, res_alias_n))
	   == HASH_UNDEFINED_VALUE)
	    pips_error("update_props", 
		       "Hash table aliases no more consistent\n");

	/* find special prop menu containing item corresponding to active 
	   phase*/
	special_prop_m = (Menu) xv_get (props_mi, MENU_PULLRIGHT);

	/* find active phase which produces resource res_true_n */
	phase_true_n = rule_phase(find_rule_by_resource(res_true_n));
	if ( (phase_alias_n=hash_get_key_by_value(aliases, phase_true_n)) 
	    == HASH_UNDEFINED_VALUE ) {
	    /* There is no alias for currently selected phase: phase_true_n
	       We have to activate another phase
	       */
	    special_prop_mi = xv_get(special_prop_m, MENU_NTH_ITEM, 1);
	    phase_alias_n = (string) xv_get(special_prop_mi, MENU_STRING);
	    user_warning("update_props",
			 "No alias available for selected phase `%s'; selecting `%s'\n",
			 phase_true_n, phase_alias_n);
	    phase_true_n = hash_get(aliases, phase_alias_n);
	    activate(phase_true_n);
	    xv_set(special_prop_mi, MENU_SELECTED, TRUE, NULL);
	    debug(1, "update_props", 
		  "Rule `%s' selected to produce resource `%s'\n",
		  phase_alias_n, res_alias_n);
	}
	else {
	    /* walk through items of special_prop_m to selecte the activated 
	       one */
	    for (j=(int)xv_get(special_prop_m, MENU_NITEMS); j>0; j--) {
		special_prop_mi = xv_get(special_prop_m, MENU_NTH_ITEM, j);
		debug(9, "update_props", "Menu item tested:\"%s\"\n", 
		      (string) xv_get(special_prop_mi, MENU_STRING));
		if ( strcmp( (string)xv_get(special_prop_mi, MENU_STRING),
			    phase_alias_n ) ==0 ) {
		    xv_set(special_prop_mi, MENU_SELECTED, TRUE, NULL);

		    user_log("Props: phase %s set on.\n", phase_alias_n);
		    debug(1, "update_props", 
			  "Rule `%s' selected to produce resource `%s'\n",
			  phase_alias_n, res_alias_n);
		}
	    }
	}
    }

    debug_off();
}

void clear_props()
{
    /* string res_alias_n, res_true_n ,phase_alias_n ,phase_true_n; */
    Menu menu_props, special_prop_m;
    Menu_item props_mi, special_prop_mi;
    int i, j;

    menu_props = smenu_props;

    /* walk through items of menu_props */
    for (i=(int)xv_get(menu_props, MENU_NITEMS); i>0; i--) {

	props_mi = (Menu_item) xv_get(menu_props, MENU_NTH_ITEM, i);
	special_prop_m = (Menu) xv_get (props_mi, MENU_PULLRIGHT);

	/* walk through items of special_prop_m */
	for (j=(int)xv_get(special_prop_m, MENU_NITEMS); j>0; j--) {
	    special_prop_mi = xv_get(special_prop_m, MENU_NTH_ITEM, j);
	    xv_set(special_prop_mi, MENU_SELECTED, FALSE);
	}
    }
}

void props_notify(menu, menu_item)
Menu menu;
Menu_item menu_item;
{
    string aliased_phase = (char *) xv_get(menu_item, MENU_STRING);
    string phase = hash_get(aliases, aliased_phase);

    if (phase == (string) HASH_UNDEFINED_VALUE)
	pips_error("props_notify", "aliases badly managed !!!\n");
    else {
	if ( db_get_current_program()==database_undefined ) {
	    prompt_user("No workspace opened. Props not accounted.\n");
	}
	else {
	    debug_on("WPIPS_DEBUG_LEVEL");
	    activate(phase);
	    debug_off();
	}

	user_log("Props: phase %s set on.\n", aliased_phase);
    }
}


void build_props(menu_props)
Menu menu_props;
{
    makefile m = parse_makefile();
    hash_table phase_by_made_htp = hash_table_make( hash_string, 0 );

    smenu_props = menu_props;

    /* walking thru rules */
    MAPL(pr, {
	rule r = RULE(CAR(pr));


	/* walking thru resources made by this particular rule */
	MAPL(pvr, {
	    virtual_resource vr = VIRTUAL_RESOURCE(CAR(pvr));
	    string vrn = virtual_resource_name(vr);
	    cons *p = CONS(STRING, rule_phase(r), NIL);
	    cons* l = (cons *) hash_get(phase_by_made_htp, vrn);

	    if ( l == (list) HASH_UNDEFINED_VALUE ) {
		hash_put(phase_by_made_htp, vrn, p);
	    }
	    else {
		(void) gen_nconc(l, p);
	    }
	}, rule_produced(r));

    }, makefile_rules(m));

    /* walking thru phase_by_made_htp */
    HASH_MAP(k, v, {
	string alias1 = hash_get_key_by_value(aliases, k);
	list l = (cons*) v;

	if ((alias1 != HASH_UNDEFINED_VALUE) && (gen_length(l) >= 2)) {

	    Menu_item mi_props;
	    Menu menu_special_prop;

	    menu_special_prop=(Menu)xv_create(NULL, MENU_CHOICE_MENU,
					      MENU_NOTIFY_PROC, props_notify,
					      NULL);

	    MAPL(vrn, {
		string alias2 = hash_get_key_by_value(aliases, 
						      STRING(CAR(vrn)));
		Menu_item mi_special_prop;

		if (alias2!=HASH_UNDEFINED_VALUE) {
		    mi_special_prop = (Menu_item)xv_create(NULL, MENUITEM,
							   MENU_STRING, 
							   alias2,
							   MENU_RELEASE,
							   NULL);
		    xv_set(menu_special_prop, MENU_APPEND_ITEM, 
			   mi_special_prop, NULL);
		}
	    }, l);

	    mi_props = (Menu_item)xv_create(NULL, MENUITEM,
					    MENU_STRING, alias1,
					    MENU_PULLRIGHT, menu_special_prop,
					    MENU_RELEASE,
					    NULL);
	    xv_set(menu_props, MENU_APPEND_ITEM, mi_props, NULL);
/*	    xv_set(menu_special_prop, 
		   MENU_SELECTED_ITEM, (Menu_item)xv_get(menu_special_prop, 
							 MENU_DEFAULT_ITEM),
		   NULL);
*/
	}
    }, phase_by_made_htp);
}



void build_aliases()
{
    char buffer[128];
    char true_name[128], alias_name[128];
    FILE *fd;
    char * wpips_rc = WPIPS_RC;

    aliases = hash_table_make(hash_string, 0);

    if(wpips_rc == NULL)
	user_error("build_aliases", "Shell variable LIBDIR is undefined. Have you run pipsrc?\n",
		   0 );
    fd = safe_fopen(wpips_rc, "r");

    while (fgets(buffer, 128, fd) != NULL) {
	if (buffer[0] == '-')
	    continue;

	sscanf(buffer, "alias%s '%[^']", true_name, alias_name);

	if (hash_get(aliases, alias_name) != HASH_UNDEFINED_VALUE) {
	    pips_error("build_aliases", "Aliases must not be ambiguous\n");
	}
	else {
	    char upper[128];

	    hash_put(aliases, 
		     strdup(alias_name), 
		     strdup(strupper(upper, true_name)));
	}
    }
}

void create_props_menu()
{
    Menu menu;

    menu = xv_create(XV_NULL, MENU_COMMAND_MENU, 
		     NULL);

    build_aliases();
    build_props(menu);

    (void) xv_create(main_panel, PANEL_BUTTON,
		     PANEL_LABEL_STRING, "Props",
		     PANEL_ITEM_MENU, menu,
		     NULL);
}
