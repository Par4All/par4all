#include <stdio.h>

#include <sys/time.h>
#include <sys/resource.h>
#include <xview/xview.h>
#include <xview/panel.h>
#include <xview/text.h>
#include <types.h>

#include "genC.h"

#include "constants.h"
#include "misc.h"
#include "top-level.h"
#include "database.h"
#include "pipsdbm.h"
#include "wpips.h"

#include "resources.h"
#include "phases.h"


void transform_notify(menu, menu_item)
Menu menu;
Menu_item menu_item;
{
    char *label = (char *) xv_get(menu_item, MENU_STRING);

    char *modulename = db_get_current_module_name();

    /* FI: borrowed from edit_notify() */
    if (modulename == NULL) {
	prompt_user("No module selected");
	return;
    }

    if (modulename != NULL) {
	if (strcmp(label, PRIVATIZE_TRANSFORM) == 0) {
	    safe_apply(BUILDER_PRIVATIZER, modulename);
	}
	else if (strcmp(label, DISTRIBUTE_TRANSFORM) == 0) {
	    safe_apply(BUILDER_DISTRIBUTER, modulename);
	}
	else if (strcmp(label, PARTIAL_EVAL_TRANSFORM) == 0) {
	    safe_apply(BUILDER_PARTIAL_EVAL, modulename);
	}
	else if (strcmp(label, UNROLL_TRANSFORM) == 0) {
	    safe_apply(BUILDER_UNROLL, modulename);
	}
	else if (strcmp(label,STRIP_MINE_TRANSFORM) == 0) {
	    safe_apply(BUILDER_STRIP_MINE, modulename);
	}
	else if (strcmp(label,LOOP_INTERCHANGE_TRANSFORM) == 0) {
	    safe_apply(BUILDER_LOOP_INTERCHANGE, modulename);
	}
	else if (strcmp(label, REDUCTIONS_TRANSFORM) == 0) {
	    safe_apply(BUILDER_REDUCTIONS, modulename);
	}
	else {
	    pips_error("transform_notify", "Bad choice");
	}
    }
}

void create_transform_menu()
{
    Menu menu;

    menu = xv_create(XV_NULL, MENU_COMMAND_MENU, 
			 MENU_GEN_PIN_WINDOW, main_frame, "Transform Menu",
		     MENU_ACTION_ITEM, PRIVATIZE_TRANSFORM, transform_notify,
		     MENU_ACTION_ITEM, DISTRIBUTE_TRANSFORM, transform_notify,
		     MENU_ACTION_ITEM, PARTIAL_EVAL_TRANSFORM, transform_notify,
		     MENU_ACTION_ITEM, UNROLL_TRANSFORM, transform_notify,
		     MENU_ACTION_ITEM, STRIP_MINE_TRANSFORM, transform_notify,
		     MENU_ACTION_ITEM, LOOP_INTERCHANGE_TRANSFORM, transform_notify,
		     MENU_ACTION_ITEM, REDUCTIONS_TRANSFORM, transform_notify,
		     NULL);

    (void) xv_create(main_panel, PANEL_BUTTON,
		     PANEL_LABEL_STRING, "Transform",
		     PANEL_ITEM_MENU, menu,
		     0);
}
