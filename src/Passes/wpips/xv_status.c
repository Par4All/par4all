#include <stdio.h>
#include <varargs.h>

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

#include "wpips.h"

static Panel_item directory_name, program_name, module_name, memory_name, message;

void show_directory()
{
    xv_set(directory_name, PANEL_VALUE, get_cwd(), 0);
}



void show_program()
{
    static char *none = "none";
    char *name = db_get_current_program_name();

    if (name == NULL)
	name = none;

    xv_set(program_name, PANEL_VALUE, name, 0);
}



void show_module()
{
    static char *none = "none";
    char *name = db_get_current_module_name();

    if (name == NULL)
	name = none;

    xv_set(module_name, PANEL_VALUE, name, 0);
}



/*VARARGS0*/
/*
void show_message(va_alist)
va_dcl
{
    static char message_buffer[SMALL_BUFFER_LENGTH];
    va_list args;
    char *fmt;

    va_start(args, char *);

    fmt = va_arg(args, char *);

    (void) vsprintf(message_buffer, fmt, args);

    va_end(args);

    xv_set(message, PANEL_VALUE, message_buffer, 0);
}
*/

void show_message(m)
string m;
{
    static char message_buffer[SMALL_BUFFER_LENGTH];

    (void) vsprintf(message_buffer, m);

    xv_set(message, PANEL_VALUE, message_buffer, 0);
}


void create_status_subwindow()
{
  window_fit(main_panel);

  status_panel = (Panel) xv_create(main_frame, PANEL,
				   WIN_X, 0,
				   WIN_BELOW, main_panel,
				   /* Corrige un bug de peinture en
				      couleur. */
				   PANEL_ITEM_X_GAP, 200,
				   /*				     PANEL_LAYOUT, PANEL_VERTICAL,*/
				   NULL);

  /* PANEL_VALUE_Y used to be unset before we shifted to xview.3, Apr. 92 */
  message = 
    xv_create(status_panel, PANEL_TEXT, 
	      PANEL_LABEL_STRING, "Message:",
	      PANEL_READ_ONLY, TRUE,
	      PANEL_VALUE_DISPLAY_LENGTH, 64,
	      NULL);

  directory_name = 
    xv_create(status_panel, PANEL_TEXT,
	      PANEL_NEXT_ROW, -1,
	      PANEL_LABEL_STRING, "Directory:",
	      PANEL_READ_ONLY, TRUE,
	      PANEL_VALUE_DISPLAY_LENGTH, 64,
	      NULL);

  program_name = 
    xv_create(status_panel, PANEL_TEXT,
	      PANEL_NEXT_ROW, -1,
	      PANEL_LABEL_STRING, "Workspace:",
	      PANEL_VALUE_DISPLAY_LENGTH, 10,
	      PANEL_READ_ONLY, TRUE, 
	      NULL);

  module_name = 
    xv_create(status_panel, PANEL_TEXT, 
	      PANEL_NEXT_ROW, -1,
	      PANEL_LABEL_STRING, "Module:",
	      PANEL_READ_ONLY, TRUE,
	      PANEL_VALUE_DISPLAY_LENGTH, 10,
	      NULL);

  memory_name = 
    xv_create(status_panel, PANEL_TEXT, 
	      /*		  PANEL_VALUE_X, xv_col(status_panel, 2), */
	      PANEL_LABEL_STRING, "Memory:",
	      PANEL_READ_ONLY, TRUE,
	      PANEL_VALUE_DISPLAY_LENGTH, 10,
	      NULL);

  window_fit(status_panel);
  window_fit(main_frame);

  show_directory();
  show_program();
  show_module();

}
