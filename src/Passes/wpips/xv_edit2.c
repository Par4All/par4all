#include <stdio.h>
#include <sys/param.h>

#include <sys/time.h>
#include <sys/resource.h>
#include <xview/xview.h>
#include <xview/panel.h>
#include <xview/text.h>
#include <types.h>

#include "genC.h"
#include "ri.h"
#include "makefile.h"
#include "database.h"

#include "misc.h"
#include "pipsdbm.h"

#include "resources.h"
#include "constants.h"
#include "top-level.h"

#include "wpips.h"

static Textsw edit_textsw[2];
static Menu_item current_selection_mi, 
                 edit, 
                 close;


void current_selection_notify(menu, menu_item)
Menu menu;
Menu_item menu_item;
{
    unhide_window(edit_frame[0]);
    unhide_window(edit_frame[1]);
}

#define NO_TEXTSW_AVAILABLE -1 /* cannot be 0 or 1 */
static bool fixed[2] = {FALSE, FALSE};

int alloc_first_initialized_window()
{
    static int next=0;
    int other;

    next=next % 2;
    other=(next+1) % 2;

    if (! ( (bool)xv_get(edit_textsw[next], TEXTSW_MODIFIED) 
	|| fixed[next]) ) {
	return (next++);
    }
    else if (! ( (bool)xv_get(edit_textsw[other], TEXTSW_MODIFIED) 
	|| fixed[other]) )
	return(other);
    else return(NO_TEXTSW_AVAILABLE);
}    

void edit_notify(menu, menu_item)
Menu menu;
Menu_item menu_item;
{
    char *modulename = db_get_current_module_name();
    char *filename;
    int win_nb;

    if (modulename == NULL) {
	prompt_user("No module selected");
	return;
    }

    /* Is there an available edit_textsw ? */
    if ( (win_nb=alloc_first_initialized_window()) == NO_TEXTSW_AVAILABLE ) {
	prompt_user("None of the 2 text-windows is available");
	return;
    }

    filename = db_get_file_resource(DBR_SOURCE_FILE, modulename, TRUE);

    xv_set(edit_frame[win_nb], FRAME_LABEL, "Pips Edit Facility", 0);

    xv_set(edit_textsw[win_nb], 
	   TEXTSW_FILE, filename,
	   TEXTSW_BROWSING, FALSE,
	   TEXTSW_FIRST, 0,
	   0);

    xv_set(current_selection_mi, 
	   MENU_STRING, "Lasts",
	   MENU_INACTIVE, FALSE,
	   0);
    xv_set(close, MENU_INACTIVE, FALSE, 0);

    unhide_window(edit_frame[win_nb]);
}


Xv_opaque view_notify(menu, menu_item)
Menu menu;
Menu_item menu_item;
{
    char *print_type, *print_type_2 = NULL;
    char *label = (char *) xv_get(menu_item, MENU_STRING);
    char *modulename = db_get_current_module_name();
    int win1, win2;

    if (modulename == NULL) {
	prompt_user("No module selected");
	return;
    }

    /* Is there an available edit_textsw ? */
    if ( (win1=alloc_first_initialized_window()) == NO_TEXTSW_AVAILABLE ) {
	prompt_user("None of the 2 text-windows is available");
	return;
    }

    xv_set(edit_frame[win1], FRAME_LABEL, "Xview Pips Display Facility", 0);

    if (strcmp(label, SEQUENTIAL_VIEW) == 0) {
	print_type = DBR_PRINTED_FILE;
    }
    else if (strcmp(label, PARALLEL_VIEW) == 0) {
	print_type = DBR_PARALLELPRINTED_FILE;
    }
    else if (strcmp(label, CALLGRAPH_VIEW) == 0) {
	print_type = DBR_CALLGRAPH_FILE;
    }
    else if (strcmp(label, ICFG_VIEW) == 0) {
	print_type = DBR_ICFG_FILE;
    }
    else if (strcmp(label, DISTRIBUTED_VIEW) == 0) {
	print_type = DBR_WP65_COMPUTE_FILE;
	print_type_2 = DBR_WP65_BANK_FILE;
    }
    else if (strcmp(label, DEPENDENCE_GRAPH_VIEW) == 0) {
	print_type = DBR_DG_FILE;
    }
    else if (strcmp(label, FLINT_VIEW) == 0) {
	print_type = DBR_FLINTED;
    }
    else {
	pips_error("view_notify", "bad label : %s\n", label);
    }

    xv_set(edit_textsw[win1], 
	   TEXTSW_FILE, build_view_file(print_type),
	   TEXTSW_BROWSING, TRUE,
	   TEXTSW_FIRST, 0,
	   0);

    if ( print_type_2 != NULL ) {
	/* Is there an available edit_textsw ? */
	if ( (win2=alloc_first_initialized_window()) 
	    == NO_TEXTSW_AVAILABLE ) {
	    prompt_user("None of the 2 text-windows is available");
	    return;
	}

	xv_set(edit_frame[win2], FRAME_LABEL, 
	       "Xview Pips Display Facility", 0);

	xv_set(edit_textsw[win2], 
	       TEXTSW_FILE, build_view_file(print_type_2),
	       TEXTSW_BROWSING, TRUE,
	       TEXTSW_FIRST, 0,
	       0);

    }

    xv_set(current_selection_mi, 
	   MENU_STRING, "Lasts",
	   MENU_INACTIVE, FALSE, 0);
    xv_set(close, MENU_INACTIVE, FALSE, 0);

    unhide_window(edit_frame[win1]);
    if ( print_type_2 != NULL ) {
	unhide_window(edit_frame[win2]);
    }
}

void edit_close_notify(menu, menu_item)
Menu menu;
Menu_item menu_item;
{
    if (! (bool)xv_get(edit_textsw[0], TEXTSW_MODIFIED))
	hide_window(edit_frame[0]);
    if (! (bool)xv_get(edit_textsw[1], TEXTSW_MODIFIED))
	hide_window(edit_frame[1]);

    if ((bool)xv_get(edit_textsw[0], TEXTSW_MODIFIED)
	||(bool)xv_get(edit_textsw[1], TEXTSW_MODIFIED) ) {
	prompt_user("File not saved in editor");
	return;
    }

    xv_set(current_selection_mi, 
	   MENU_STRING, "No Selection",
	   MENU_INACTIVE, TRUE, 0);

    xv_set(close, MENU_INACTIVE, TRUE, 0);

    hide_window(edit_frame[0]);
    hide_window(edit_frame[1]);
}



void create_edit_window()
{
    /* Xv_Window window; */

    edit_textsw[0] = xv_create(edit_frame[0], TEXTSW, 
			    TEXTSW_DISABLE_CD, TRUE,
			    TEXTSW_DISABLE_LOAD, TRUE,
			    0);

    edit_textsw[1] = xv_create(edit_frame[1], TEXTSW, 
			    TEXTSW_DISABLE_CD, TRUE,
			    TEXTSW_DISABLE_LOAD, TRUE,
			    0);

/*    window = (Xv_Window) xv_find(edit_frame, WINDOW, 0);

    xv_set(window, 
	   WIN_EVENT_PROC, default_win_interpose, 
	   NULL);*/
}



void create_edit_menu()
{
    Menu menu;

    current_selection_mi = 
	xv_create(NULL, MENUITEM, 
		  MENU_STRING, "No Selection",
		  MENU_NOTIFY_PROC, current_selection_notify,
		  MENU_INACTIVE, TRUE,
		  MENU_RELEASE,
		  NULL);
    edit = 
	xv_create(NULL, MENUITEM, 
		  MENU_STRING, "Edit",
		  MENU_NOTIFY_PROC, edit_notify,
		  MENU_RELEASE,
		  NULL);

    close = 
	xv_create(NULL, MENUITEM, 
		  MENU_STRING, "Close",
		  MENU_NOTIFY_PROC, edit_close_notify,
		  MENU_INACTIVE, TRUE,
		  MENU_RELEASE,
		  NULL);

    menu = 
	xv_create(XV_NULL, MENU_COMMAND_MENU, 
		  MENU_APPEND_ITEM, current_selection_mi,
		  MENU_APPEND_ITEM, edit,
		  MENU_ACTION_ITEM, SEQUENTIAL_VIEW, view_notify,
		  MENU_ACTION_ITEM, CALLGRAPH_VIEW, view_notify,
		  MENU_ACTION_ITEM, ICFG_VIEW, view_notify,
		  MENU_ACTION_ITEM, PARALLEL_VIEW, view_notify,
		  MENU_ACTION_ITEM, DISTRIBUTED_VIEW, view_notify,
		  MENU_ACTION_ITEM, DEPENDENCE_GRAPH_VIEW, view_notify,
		  MENU_ACTION_ITEM, FLINT_VIEW, view_notify,
		  MENU_APPEND_ITEM, close,
		  NULL);

    (void) xv_create(main_panel, PANEL_BUTTON,
		     PANEL_LABEL_STRING, "Edit/View",
		     PANEL_ITEM_MENU, menu,
		     NULL);

}
