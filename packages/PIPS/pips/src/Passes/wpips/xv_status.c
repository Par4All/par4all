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
#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <sys/time.h>
#include <sys/resource.h>

#include <sys/time.h>
#include <sys/resource.h>
#include <xview/xview.h>
#include <xview/panel.h>
#include <xview/text.h>
#include <xview/svrimage.h>

#if (defined(TEXT))
#undef TEXT
#endif

#if (defined(TEXT_TYPE))
#undef TEXT_TYPE
#endif

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "database.h"

#include "misc.h"
#include "ri-util.h"
#include "pipsdbm.h"

#include "wpips.h"

enum {DECALAGE_STATUS = 100
};

/* Max number of digits displayed in the status panel: */
enum {CPU_USAGE_LENGTH = 8
};


Panel_item directory_name_panel_item;
Panel_item workspace_name_panel_item,
memory_name, message, window_number;
Panel_item module_name_panel_item;
Panel_item cpu_usage_item;

Server_image status_window_pips_image;

/* Strange, "man end" says that end is a function! */
extern etext, edata, end;

void
display_memory_usage()
{
   char memory_string[17];
   char cpu_string[CPU_USAGE_LENGTH + 1];
   struct rusage an_rusage;

   /* etext, edata and end are only address symbols... */
   debug(6, "display_memory_usage",
	 "_text %#x, _data %#x, _end %#x, _brk %#x\n",
	 &etext, &edata, &end, sbrk(0));
   
   sprintf(memory_string, "%10.3f", (sbrk(0) - (int) &etext)/(double)(1 << 20));

   xv_set(memory_name,
          PANEL_VALUE, memory_string,
          NULL);

   if (getrusage(RUSAGE_SELF, &an_rusage) == 0) {
      double the_cpu_time =
         an_rusage.ru_utime.tv_sec + an_rusage.ru_stime.tv_sec
         + (an_rusage.ru_utime.tv_usec + an_rusage.ru_stime.tv_usec)*1e-6;
      sprintf(cpu_string, "%*.*g",
              CPU_USAGE_LENGTH, CPU_USAGE_LENGTH - 2, the_cpu_time);
   }
   else
      /* getrusage() failed: */
      sprintf(cpu_string, "%*s", CPU_USAGE_LENGTH, "?");
   
   xv_set(cpu_usage_item,
          PANEL_VALUE, cpu_string,
          NULL);
}


void
window_number_notify(Panel_item item, int value, Event *event)
{
   number_of_wpips_windows = (int) xv_get(item, PANEL_VALUE);
   
   if (wpips_emacs_mode)
      send_window_number_to_emacs(number_of_wpips_windows);
   
   display_memory_usage();
}


void
show_directory()
{
   xv_set(directory_name_panel_item, PANEL_VALUE, get_cwd(), NULL);
   display_memory_usage();
}



void
show_workspace()
{
   static char *none = "(* none *)";
   char *name = db_get_current_workspace_name();

   if (name == NULL)
      name = none;

   xv_set(workspace_name_panel_item, PANEL_VALUE, name, 0);
   display_memory_usage();
}


void
show_module()
{
   static char *none = "(* none *)";
   char *module_name_content = db_get_current_module_name();

   if (module_name_content == NULL)
      module_name_content = none;

   xv_set(module_name_panel_item, PANEL_VALUE, module_name_content, 0);
   display_memory_usage();
}


void
wpips_interrupt_button_blink()
{
   if ((Server_image) xv_get(status_window_pips_image, PANEL_LABEL_IMAGE) ==
       wpips_negative_server_image)
      xv_set(status_window_pips_image,
             PANEL_LABEL_IMAGE, wpips_positive_server_image,
             NULL);
   else
      xv_set(status_window_pips_image,
             PANEL_LABEL_IMAGE, wpips_negative_server_image,
             NULL);
      
}


void
wpips_interrupt_button_restore()
{
   xv_set(status_window_pips_image,
          PANEL_LABEL_IMAGE, wpips_positive_server_image,
          NULL);
}


void
show_message(string message_buffer /*, ...*/)
{
   /* va_list some_arguments;
   static char message_buffer[SMALL_BUFFER_LENGTH]; */

   /* va_start(some_arguments, a_printf_format); */

   /* (void) vsprintf(message_buffer, a_printf_format, some_arguments);*/

   xv_set(message, PANEL_VALUE, message_buffer, 0);
   display_memory_usage();
}


void
create_status_subwindow()
{
   /* Maintenant on n'utilise plus qu'un seul panel pour la fene^tre
      principale.  En effet, sinon il y a des proble`mes de retrac,age
      sur e'cran couleur. RK, 15/03/1994. */
   Server_image pips_icon_server_image;
   
   message = 
      xv_create(main_panel, PANEL_TEXT, 
                PANEL_VALUE_X, DECALAGE_STATUS,
                PANEL_VALUE_Y, xv_rows(main_panel, 1),
                PANEL_LABEL_STRING, "Message:",
                PANEL_READ_ONLY, TRUE,
                PANEL_VALUE_DISPLAY_LENGTH, 64,
                PANEL_VALUE_STORED_LENGTH, 1000,
                NULL);
/*
   directory_name_panel_item = 
      xv_create(main_panel, PANEL_TEXT, 
                PANEL_VALUE_X, DECALAGE_STATUS,
                PANEL_VALUE_Y, xv_rows(main_panel, 2),
                PANEL_LABEL_STRING, "Directory:",
                PANEL_READ_ONLY, FALSE,
                PANEL_NOTIFY_PROC, end_directory_text_notify,
                PANEL_VALUE_DISPLAY_LENGTH, 64,
                PANEL_VALUE_STORED_LENGTH, 256,
                NULL);
   */
   directory_name_panel_item =
      schoose_create_abbrev_menu_with_text(
	  main_panel,
	  "Directory:",
	  61,
	  DECALAGE_STATUS,
	  xv_rows(main_panel, 2),
	  generate_directory_menu,
	  /* Ignore the return code of end_directory_notify: */
	  (void (*)(char *)) end_directory_notify); 

   xv_set(directory_name_panel_item,
          PANEL_VALUE_STORED_LENGTH, 512,
          NULL);
   
   workspace_name_panel_item =
      schoose_create_abbrev_menu_with_text(main_panel,
                                           "Workspace:",
                                           20,
                                           DECALAGE_STATUS,
                                           xv_rows(main_panel, 3),
                                           generate_workspace_menu,
                                           open_or_create_workspace);

   module_name_panel_item =
      schoose_create_abbrev_menu_with_text(main_panel,
                                           "Module:",
                                           20,
                                           DECALAGE_STATUS,
                                           xv_rows(main_panel, 4),
                                           generate_module_menu,
                                           end_select_module_notify);

   memory_name = 
      xv_create(main_panel, PANEL_TEXT,
                PANEL_ITEM_X_GAP, DECALAGE_STATUS,
                PANEL_VALUE_X, xv_col(main_panel, 50),
                PANEL_VALUE_Y, xv_rows(main_panel, 3),
                PANEL_LABEL_STRING, "Memory (MB):",
                PANEL_VALUE_DISPLAY_LENGTH, 9,
                PANEL_READ_ONLY, TRUE,
                NULL);

   cpu_usage_item = xv_create(main_panel, PANEL_TEXT,
                             PANEL_LABEL_STRING, "CPU (s):", 
                              PANEL_VALUE_DISPLAY_LENGTH, CPU_USAGE_LENGTH,
                              PANEL_READ_ONLY, TRUE,
                              PANEL_VALUE_X, xv_col(main_panel, 68),
                              PANEL_VALUE_Y, xv_rows(main_panel, 3),
                              NULL);
   
   window_number = 
      xv_create(main_panel, PANEL_NUMERIC_TEXT,
                /*PANEL_ITEM_X_GAP, DECALAGE_STATUS,
                  PANEL_VALUE_Y, xv_rows(main_panel, 4),*/
                PANEL_VALUE_X, xv_col(main_panel, 50),
                PANEL_VALUE_Y, xv_rows(main_panel, 4),
                PANEL_LABEL_STRING, "# windows:",
                PANEL_MIN_VALUE, 1,
                PANEL_MAX_VALUE, MAX_NUMBER_OF_WPIPS_WINDOWS,
                PANEL_VALUE, number_of_wpips_windows,
                PANEL_VALUE_DISPLAY_LENGTH, 2,
                PANEL_NOTIFY_PROC, window_number_notify,
                NULL);
   
   pips_icon_server_image = create_status_window_pips_image();
   
   status_window_pips_image =
      xv_create(main_panel, PANEL_BUTTON,
                PANEL_LABEL_IMAGE,
                pips_icon_server_image,
                PANEL_NOTIFY_PROC, wpips_interrupt_pipsmake,
                                /* Put the Pixmap above the Help button: */
                XV_X, xv_get(quit_button, XV_X) + (xv_get(quit_button, XV_WIDTH) - xv_get(pips_icon_server_image, XV_WIDTH))/2,
                XV_Y, xv_rows(main_panel, 3) + 20,
                NULL);
   
   window_fit(main_panel);
   window_fit(main_frame);

   show_directory();
   show_workspace();
   show_module();
   display_memory_usage();
}
