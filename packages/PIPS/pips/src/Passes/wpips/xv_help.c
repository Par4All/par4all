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

#include <sys/time.h>
#include <sys/resource.h>
#include <xview/xview.h>
#include <xview/panel.h>
#include <xview/text.h>

#include "genC.h"
#include "top-level.h"
#include "wpips.h"
#include "xv_sizes.h"

#include "genC.h"
#include "misc.h"

#define min(a,b) ((a) < (b) ? (a) : (b))
#define HELP_LINES 32

/* The URL of the PIPS documentation at the École des Mines de Paris: */
#define PIPS_DOCUMENTATION_URL "http://www.cri.ensmp.fr/pips"

static Panel_item lines[HELP_LINES]; /* GRRRRRRRR FC. */
static gen_array_t help_list = (gen_array_t) NULL;

void
display_help(char * topic)
{
   int i, n;

   if (!help_list) /* lazy */
       help_list = gen_array_make(0);

   get_help_topic(topic, help_list);
   n = gen_array_nitems(help_list);

   for (i = 0; i < min(HELP_LINES, n); i++) {
      xv_set(lines[i], PANEL_LABEL_STRING, 
	     gen_array_item(help_list, i), 0);
   }

   for (i = min(HELP_LINES, n); i < HELP_LINES; i++) {
      xv_set(lines[i], PANEL_LABEL_STRING, "", 0);
   }

   unhide_window(help_frame);
}



static void
close_help_notify(Panel_item item,
                  Event * event)
{
    gen_array_full_free(help_list), help_list = (gen_array_t) NULL;
    hide_window(help_frame);
}



void
create_help_window()
{
   int i;
   Panel help_panel;

   help_panel = xv_create(help_frame, PANEL,
                          NULL);

   for (i = 0; i < HELP_LINES; i++) {
      lines[i] = xv_create(help_panel, PANEL_MESSAGE, 
                           XV_X, 15,
                           XV_Y, 15*(i+1),
                           0);
   }

   (void) xv_create(help_panel, PANEL_BUTTON,
                    PANEL_LABEL_STRING, "CLOSE",
                    XV_X, HELP_WIDTH/2-28,
                    XV_Y, 15*(HELP_LINES+1),
                    PANEL_NOTIFY_PROC, close_help_notify,
                    0);


}

static void
help_notify(Menu menu,
            Menu_item menu_item)
{
   display_help((char *) xv_get(menu_item, MENU_CLIENT_DATA));
}


static void
help_launch_pips_netscape(Menu menu,
                          Menu_item menu_item)
{
   system("netscape " PIPS_DOCUMENTATION_URL " &");
}


static void
help_launch_pips_xmosaic(Menu menu,
                          Menu_item menu_item)
{
   system("xmosaic " PIPS_DOCUMENTATION_URL " &");
}


void
create_help_menu()
{
   Menu menu;

   menu = xv_create(XV_NULL, MENU_COMMAND_MENU, 
                    MENU_TITLE_ITEM,
                    "The PIPS documentation",
                    MENU_GEN_PIN_WINDOW, main_frame, "Help Menu",
                    MENU_ITEM,
                    MENU_STRING, "A few introductory words...",
                    MENU_CLIENT_DATA, strdup("Introduction"),
                    MENU_NOTIFY_PROC, help_notify,
                    NULL,
                    MENU_ITEM,
                    MENU_STRING, "A few words about \"Workspace\"...",
                    MENU_CLIENT_DATA, strdup("Workspace"),
                    MENU_NOTIFY_PROC, help_notify,
                    NULL,
                    MENU_ITEM,
                    MENU_STRING, "A few words about \"Module\"...",
                    MENU_CLIENT_DATA, strdup("Module"),
                    MENU_NOTIFY_PROC, help_notify,
                    NULL,
                    MENU_ITEM,
                    MENU_STRING, "A few words about \"Directory\"...",
                    MENU_CLIENT_DATA, strdup("Directory"),
                    MENU_NOTIFY_PROC, help_notify,
                    NULL,
                    MENU_ITEM,
                    MENU_STRING, "A few words about \"View\"...",
                    MENU_CLIENT_DATA, strdup("View"),
                    MENU_NOTIFY_PROC, help_notify,
                    NULL,
                    MENU_ITEM,
                    MENU_STRING, "A few words about \"Transform/Edit\"...",
                    MENU_CLIENT_DATA, strdup("Transform/Edit"),
                    MENU_NOTIFY_PROC, help_notify,
                    NULL,
                    MENU_ITEM,
                    MENU_STRING, "A few words about \"Compile\"...",
                    MENU_CLIENT_DATA, strdup("Compile"),
                    MENU_NOTIFY_PROC, help_notify,
                    NULL,
                    MENU_ITEM,
                    MENU_STRING, "A few words about \"Options\"...",
                    MENU_CLIENT_DATA, strdup("Options"),
                    MENU_NOTIFY_PROC, help_notify,
                    NULL,
                    MENU_ITEM,
                    MENU_STRING, "A few words about \"Log\"...",
                    MENU_CLIENT_DATA, strdup("Log"),
                    MENU_NOTIFY_PROC, help_notify,
                    NULL,
                    /* Just a separator: */
                    WPIPS_MENU_SEPARATOR,
                    MENU_ACTION_ITEM, "The PIPS documentation on Internet with Netscape...",
                    help_launch_pips_netscape,
                    MENU_ACTION_ITEM, "The PIPS documentation on Internet with XMosaic...",
                    help_launch_pips_xmosaic,
                    NULL);

   (void) xv_create(main_panel, PANEL_BUTTON,
                    PANEL_LABEL_STRING, "Help ",
                    PANEL_ITEM_MENU, menu,
                    /* Align the Help button with the Quit button:
                    XV_X, xv_get(quit_button, XV_X), */
                    NULL);
}
