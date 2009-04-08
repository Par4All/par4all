#include <stdio.h>

#include <sys/time.h>
#include <sys/resource.h>
#include <xview/xview.h>
#include <xview/panel.h>
#include <xview/text.h>

#include "genC.h"
#include "misc.h"

#include "wpips.h"

/* Include the label names: */
#include "wpips-labels.h"

#include "constants.h"

void analyze_notify(menu, menu_item)
	Menu menu;Menu_item menu_item; {
	char *label = (char *) xv_get(menu_item, MENU_STRING);

	if (strcmp(label, SEMANTICS_ANALYZE) == 0) {
		prompt_user("Not Implemented");
	} else if (strcmp(label, CALLGRAPH_ANALYZE) == 0) {
		prompt_user("Not Implemented");
	} else {
		pips_error("analyze_notify", "Bad choice");
	}
}

void create_analyze_menu() {
	Menu menu;

	menu = xv_create(XV_NULL, MENU_COMMAND_MENU, MENU_ACTION_ITEM,
			SEMANTICS_ANALYZE, analyze_notify, MENU_ACTION_ITEM,
			CALLGRAPH_ANALYZE, analyze_notify, NULL);

	(void) xv_create(main_panel, PANEL_BUTTON, PANEL_LABEL_STRING, "Analyze",
			PANEL_ITEM_MENU, menu, 0);
}
