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

/* Include the label names: */
#include "wpips-labels.h"

#define NO_TEXTSW_AVAILABLE -1 /* cannot be positive (i.e. a window number. */
static Textsw edit_textsw[MAX_NUMBER_OF_WPIPS_WINDOWS];
static Panel_item check_box[MAX_NUMBER_OF_WPIPS_WINDOWS];
static bool dont_touch_window[MAX_NUMBER_OF_WPIPS_WINDOWS];
int number_of_wpips_windows = INITIAL_NUMBER_OF_WPIPS_WINDOWS;

static Menu_item current_selection_mi, 
                 edit, 
                 close;


void current_selection_notify(menu, menu_item)
Menu menu;
Menu_item menu_item;
{
  int i;
  
  for(i = 0; i < number_of_wpips_windows; i++)
    unhide_window(edit_frame[i]);
}


#define DONT_TOUCH_WINDOW_ADDRESS 0

void dont_touch_window_notify(Panel_item item, int value, Event *event)
{
  bool *dont_touch_window = (bool *) xv_get(item, XV_KEY_DATA,
					    DONT_TOUCH_WINDOW_ADDRESS);
  *dont_touch_window = (bool) xv_get(item, PANEL_VALUE);
}


int alloc_first_initialized_window()
{
   static int next = 0;
   int i, candidate;

   for(i = next; i < next + number_of_wpips_windows; i++) {
      candidate = i % number_of_wpips_windows;
      if (! wpips_emacs_mode) {
         /* Make sence only without Emacs yet: */
         /* Skip windows with modified text inside : */
         if ((bool)xv_get(edit_textsw[candidate], TEXTSW_MODIFIED))
            continue;
         /* Skip windows with a retain attribute : */
         /*
           check_box = (Panel_item) xv_get(edit_textsw[candidate],
           XV_KEY_DATA, PANEL_CHECK_BOX,
           NULL);
           */
         if ((bool)xv_get(check_box[candidate], PANEL_VALUE))
            continue;
      }
    
      next = candidate + 1;
      return candidate;
   }
   return(NO_TEXTSW_AVAILABLE);
}    

void edit_notify(menu, menu_item)
Menu menu;
Menu_item menu_item;
{
	char string_filename[SMALL_BUFFER_LENGTH], string_modulename[SMALL_BUFFER_LENGTH];
    char *modulename = db_get_current_module_name();
    char *filename;
    int win_nb;

    if (modulename == NULL) {
		prompt_user("No module selected");
		return;
    }

    /* Is there an available edit_textsw ? */
    if ( (win_nb=alloc_first_initialized_window()) == NO_TEXTSW_AVAILABLE ) {
		prompt_user("None of the text-windows is available");
		return;
    }

    filename = db_get_file_resource(DBR_SOURCE_FILE, modulename, TRUE);
	sprintf(string_filename, "File: %s", filename);
	sprintf(string_modulename, "Module: %s", modulename);

		/* Display the file name and the module name. RK, 2/06/1993 : */
    xv_set(edit_frame[win_nb], FRAME_LABEL, "Pips Edit Facility",
		FRAME_SHOW_FOOTER, TRUE,
		FRAME_LEFT_FOOTER, string_filename,
		FRAME_RIGHT_FOOTER, string_modulename,
		NULL);

    xv_set(edit_textsw[win_nb], 
	   TEXTSW_FILE, filename,
	   TEXTSW_BROWSING, FALSE,
	   TEXTSW_FIRST, 0,
	   NULL);

    xv_set(current_selection_mi, 
	   MENU_STRING, "Lasts",
	   MENU_INACTIVE, FALSE,
	   NULL);

    xv_set(close, MENU_INACTIVE, FALSE, NULL);

    unhide_window(edit_frame[win_nb]);
}


char *compute_title_string(int window_number)
{
  char title_string_beginning[] = "Xview Pips Display Facility # ";
  static char title_string[sizeof(title_string_beginning) + 4];

  (void) sprintf(title_string, "%s%d",
		 title_string_beginning, window_number + 1);
  /* xv_set will copy the string. */
  return title_string;
}


/* To display some Pips output with wpips or epips: */
void
wpips_execute_and_display_something(char * label)
{
    char string_modulename[SMALL_BUFFER_LENGTH], bank_view_name[SMALL_BUFFER_LENGTH];
    char busy_label[SMALL_BUFFER_LENGTH];
    char *busy_label_format = "*Computing %s * ...";
    char *print_type, *print_type_2 = NULL;
    char *modulename = db_get_current_module_name();
    int win1, win2;
    Icon icon_number, icon_number2;

    if (modulename == NULL) {
	prompt_user("No module selected");
	return;
    }

    /* Is there an available edit_textsw ? */
    if ( (win1=alloc_first_initialized_window()) == NO_TEXTSW_AVAILABLE ) {
	prompt_user("None of the text-windows is available");
	return;
    }
    icon_number = icon_number2 = -1;
    if (strcmp(label, USER_VIEW) == 0) {
	print_type = DBR_PARSED_PRINTED_FILE;
	icon_number = user_ICON;
    }
    else if (strcmp(label, SEQUENTIAL_VIEW) == 0) {
	print_type = DBR_PRINTED_FILE;
	icon_number = sequential_ICON;
    }
    else if (strcmp(label, SEQUENTIAL_EMACS_VIEW) == 0) {
	print_type = DBR_EMACS_PRINTED_FILE;
	icon_number = sequential_ICON;
    }
    else if (strcmp(label, PARALLEL_VIEW) == 0) {
	print_type = DBR_PARALLELPRINTED_FILE;
	icon_number = parallel_ICON;
    }
    else if (strcmp(label, CALLGRAPH_VIEW) == 0) {
	print_type = DBR_CALLGRAPH_FILE;
	icon_number = callgraph_ICON;
    }
    else if (strcmp(label, ICFG_VIEW) == 0) {
	print_type = DBR_ICFG_FILE;
	icon_number = ICFG_ICON;
    }
    else if (strcmp(label, DISTRIBUTED_VIEW) == 0) {
	print_type = DBR_WP65_COMPUTE_FILE;
	icon_number = WP65_PE_ICON;
	print_type_2 = DBR_WP65_BANK_FILE;
	icon_number2 = WP65_bank_ICON;
    }
    else if (strcmp(label, DEPENDENCE_GRAPH_VIEW) == 0) {
	print_type = DBR_DG_FILE;
    }
    else if (strcmp(label, FLINT_VIEW) == 0) {
	print_type = DBR_FLINTED;
    }
    else if (strcmp(label, ARRAY_DFG_VIEW) == 0) {
	print_type = DBR_ADFG_FILE;
    }
    else if (strcmp(label, TIME_BASE_VIEW) == 0) {
	print_type = DBR_BDT_FILE;
    }
    else if (strcmp(label, PLACEMENT_VIEW) == 0) {
	print_type = DBR_PLC_FILE;
    }
    else {
	pips_error("view_notify", "bad label : %s\n", label);
    }

    if (! wpips_emacs_mode) {
	(void) sprintf(busy_label, busy_label_format, label);
	/* Display the file name and the module name. RK, 2/06/1993 : */
	sprintf(string_modulename, "Module: %s", modulename);
	xv_set(edit_frame[win1], FRAME_LABEL, compute_title_string(win1),
	       FRAME_SHOW_FOOTER, TRUE,
	       FRAME_LEFT_FOOTER, busy_label,
	       FRAME_RIGHT_FOOTER, string_modulename,
	       FRAME_BUSY, TRUE,
	       NULL);
	
	set_pips_icon(edit_frame[win1], icon_number, modulename);

	xv_set(edit_textsw[win1], 
	       TEXTSW_FILE, build_view_file(print_type),
	       TEXTSW_BROWSING, TRUE,
	       TEXTSW_FIRST, 0,
	       NULL);

	xv_set(edit_frame[win1],
	       FRAME_LEFT_FOOTER, label,
	       FRAME_BUSY, FALSE,
	       NULL);
    }
    else {
	/* The Emacs mode equivalent: */
	send_window_number_to_emacs(win1);
	send_module_name_to_emacs(modulename);
	/* send_icon_name_to_emacs(icon_number); */
	send_view_to_emacs(label, build_view_file(print_type));
    }
   
  
    if ( print_type_2 != NULL ) {
	/* Is there an available edit_textsw ? */
	if ( (win2=alloc_first_initialized_window()) 
	    == NO_TEXTSW_AVAILABLE ) {
	    prompt_user("None of the text-windows is available");
	    return;
	}

	if (! wpips_emacs_mode) {
	    /* Display the file name and the module name. RK, 2/06/1993 : */
	    (void) sprintf(bank_view_name, "%s (bank view)", label);
	    (void) sprintf(busy_label, busy_label_format, bank_view_name);
	    xv_set(edit_frame[win2], FRAME_LABEL, compute_title_string(win2),
		   FRAME_SHOW_FOOTER, TRUE,
		   FRAME_LEFT_FOOTER, busy_label,
		   FRAME_RIGHT_FOOTER, string_modulename,
		   FRAME_BUSY, TRUE,
		   NULL);
    
	    set_pips_icon(edit_frame[win2], icon_number2, modulename);

	    xv_set(edit_textsw[win2], 
		   TEXTSW_FILE, build_view_file(print_type_2),
		   TEXTSW_BROWSING, TRUE,
		   TEXTSW_FIRST, 0,
		   NULL);
    
	    xv_set(edit_frame[win2],
		   FRAME_LEFT_FOOTER, bank_view_name,
		   FRAME_BUSY, FALSE,
		   NULL);
	}
	else {
	    /* The Emacs mode equivalent: */
	    send_window_number_to_emacs(win2);
	    /* Should be the same, nevertheless...: */
	    send_module_name_to_emacs(modulename);
	    /* send_icon_name_to_emacs(icon_number2); */
	    send_view_to_emacs("BANK", build_view_file(print_type_2));
	}
    }

    xv_set(current_selection_mi, 
	   MENU_STRING, "Lasts",
	   MENU_INACTIVE, FALSE, NULL);
    xv_set(close, MENU_INACTIVE, FALSE, NULL);

    if (! wpips_emacs_mode) {
	unhide_window(edit_frame[win1]);
	if ( print_type_2 != NULL ) {
	    unhide_window(edit_frame[win2]);
	}
    }
}


void view_notify(menu, menu_item)
Menu menu;
Menu_item menu_item;
{
   char *label = (char *) xv_get(menu_item, MENU_STRING);

   wpips_execute_and_display_something(label);
}

void edit_close_notify(menu, menu_item)
     Menu menu;
Menu_item menu_item;
{
  int i;
  
  for(i = 0; i < MAX_NUMBER_OF_WPIPS_WINDOWS; i++)
    if (! (bool)xv_get(edit_textsw[i], TEXTSW_MODIFIED))
      hide_window(edit_frame[i]);

  for(i = 0; i < MAX_NUMBER_OF_WPIPS_WINDOWS; i++)
    if ((bool)xv_get(edit_textsw[i], TEXTSW_MODIFIED)) {
      unhide_window(edit_frame[i]);
      prompt_user("File not saved in editor");
      return;
    }
  
  xv_set(current_selection_mi, 
	 MENU_STRING, "No Selection",
	 MENU_INACTIVE, TRUE, NULL);

  xv_set(close, MENU_INACTIVE, TRUE, NULL);

  for(i = 0; i < MAX_NUMBER_OF_WPIPS_WINDOWS; i++)
    hide_window(edit_frame[i]);
}



void create_edit_window()
{
  /* Xv_Window window; */
  int i;
  
  for(i = 0; i < MAX_NUMBER_OF_WPIPS_WINDOWS; i++) {
    Panel panel;
    
    edit_textsw[i] = xv_create(edit_frame[i], TEXTSW, 
			       TEXTSW_DISABLE_CD, TRUE,
			       TEXTSW_DISABLE_LOAD, TRUE,
			       0);
    window_fit(edit_textsw[i]);
    panel = xv_create(edit_frame[i], PANEL,
		      WIN_ROW_GAP, 1,
		      WIN_COLUMN_GAP, 1,
		      NULL);
    dont_touch_window[i] = FALSE;
    check_box[i] = xv_create(panel, PANEL_CHECK_BOX,
			     PANEL_CHOICE_STRINGS, "Retain this window", NULL,
			     PANEL_VALUE, dont_touch_window[i],
			     PANEL_ITEM_X_GAP, 1,
			     PANEL_ITEM_Y_GAP, 1,
			     PANEL_NOTIFY_PROC, dont_touch_window_notify,
			     XV_KEY_DATA, DONT_TOUCH_WINDOW_ADDRESS, &dont_touch_window[i],
			     NULL);
    window_fit_height(panel);
    window_fit(edit_frame[i]);
  }
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
		  MENU_GEN_PIN_WINDOW, main_frame, "View & Edit Menu",
		  MENU_APPEND_ITEM, current_selection_mi,
		  MENU_APPEND_ITEM, edit,
		  MENU_ACTION_ITEM, USER_VIEW, view_notify,
		  MENU_ACTION_ITEM, SEQUENTIAL_VIEW, view_notify,
		  MENU_ACTION_ITEM, CALLGRAPH_VIEW, view_notify,
		  MENU_ACTION_ITEM, ICFG_VIEW, view_notify,
		  MENU_ACTION_ITEM, PARALLEL_VIEW, view_notify,
		  MENU_ACTION_ITEM, DISTRIBUTED_VIEW, view_notify,
		  /* MENU_ACTION_ITEM, HPFC_VIEW, view_notify,*/
		  MENU_ACTION_ITEM, DEPENDENCE_GRAPH_VIEW, view_notify,
		  MENU_ACTION_ITEM, FLINT_VIEW, view_notify,
		  MENU_ACTION_ITEM, SEQUENTIAL_EMACS_VIEW, view_notify,
		  MENU_ACTION_ITEM, ARRAY_DFG_VIEW, view_notify,
		  MENU_ACTION_ITEM, TIME_BASE_VIEW, view_notify,
		  MENU_ACTION_ITEM, PLACEMENT_VIEW, view_notify,
		  MENU_APPEND_ITEM, close,
		  NULL);

    (void) xv_create(main_panel, PANEL_BUTTON,
		     PANEL_LABEL_STRING, "Edit/View",
		     PANEL_ITEM_MENU, menu,
		     NULL);

}
