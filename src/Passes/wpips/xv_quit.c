#include <stdio.h>

#include <sys/time.h>
#include <sys/resource.h>
#include <xview/xview.h>
#include <xview/panel.h>
#include <xview/notice.h>

#include "genC.h"
#include "database.h"
#include "misc.h"
#include "linear.h"
#include "ri.h"

#include "ri-util.h"
#include "pipsdbm.h"
#include "top-level.h"

#include "wpips.h"

#define QUICK_QUIT "Quit without saving"
#define CLOSE_QUIT "Close (save) the Workspace & Quit"
#define DELETE_QUIT "Delete the Workspace & Quit"
#define CD_HACK_QUIT "Change Directory (tcl/tk hack)"

Panel_item quit_button;


void
cd_notify(Menu menu, Menu_item menu_item)
{
    direct_change_directory();
}

void
quit_notify(Menu menu,
            Menu_item menu_item)
{
   Event e;
   int result;
   string pn;

   if ((pn = db_get_current_workspace_name())) {
      string fmt="Workspace %s not closed";
      char str[SMALL_BUFFER_LENGTH];
      string str1, str2, menu_string;

      str2 = "Do you really want to quit PIPS?";
      menu_string=(string) xv_get(menu_item, MENU_STRING);
      if (strcmp(menu_string,CLOSE_QUIT)==0)
         str1=" ";
      else 
         str1="-=< Resources can get lost! >=-";

      sprintf(str, fmt , pn);

      /* Send to emacs if we are in the emacs mode: */
      if (wpips_emacs_mode)
         send_notice_prompt_to_emacs(str, str1, str2, NULL);
      result =  notice_prompt(xv_find(main_frame, WINDOW, 0), 
                              &e,
                              NOTICE_MESSAGE_STRINGS,
                              str, str1, str2,
                              NULL,
                              NOTICE_BUTTON_YES,	menu_string,
                              NOTICE_BUTTON_NO,	"Cancel",
                              NULL);
      if (result == NOTICE_NO)
         return;
      else if (strcmp(menu_string, CLOSE_QUIT) == 0)
         close_workspace();
      else if (strcmp(menu_string, DELETE_QUIT) == 0)
         delete_workspace(pn);
   }

   /* Clear the log window to avoid the message about the edited
      state: 
      clear_log_subwindow(NULL, NULL);
      Does not work...
      Quit:
      xv_destroy[_safe](main_frame);
      */
   /* Exit xv_main_loop() at top level: */
   notify_stop();
}


void
create_quit_button()
{
   Menu menu;

   menu = xv_create(XV_NULL, MENU_COMMAND_MENU, 
                    MENU_ACTION_ITEM, CLOSE_QUIT, quit_notify,
                    MENU_ACTION_ITEM, QUICK_QUIT, quit_notify,
                    MENU_ACTION_ITEM, DELETE_QUIT, quit_notify,
		    MENU_ACTION_ITEM, CD_HACK_QUIT, cd_notify,
                    NULL);

   quit_button = xv_create(main_panel, PANEL_BUTTON,
                           PANEL_LABEL_STRING, "Quit ",
                           PANEL_ITEM_MENU, menu,
                           NULL);
}
