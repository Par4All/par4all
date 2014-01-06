/*

  $Id$

  Copyright 1989-2014 MINES ParisTech

  This file is part of PIPS.

  PIPS is free software: you can redistribute it and/or modify it
  under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  any later version.

  PIPS is distributed in the hope that it will be useful, but WITHOUT ANY
  WARRANTY; without even the implied warranty of MERCHANTABILITY or
  FITNESS FOR A PARTICULAR PURPOSE.

  See the GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with PIPS.  If not, see <http://www.gnu.org/licenses/>.

*/
#ifdef HAVE_CONFIG_H
    #include "pips_config.h"
#endif
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
