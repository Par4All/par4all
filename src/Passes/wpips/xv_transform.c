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
#include "ri.h"
#include "top-level.h"
#include "database.h"
#include "pipsdbm.h"
#include "wpips.h"

/* Include the label names: */
#include "wpips-labels.h"

#include "resources.h"
#include "phases.h"

/* The transform menu: */
Menu transform_menu;

void
apply_on_each_transform_item(void (* function_to_apply_on_each_menu_item)(Menu_item))
{
   int i;
   /* Walk through items of  */
   for (i = (int) xv_get(transform_menu, MENU_NITEMS); i > 0; i--) {
      Menu_item menu_item = (Menu_item) xv_get(transform_menu,
                                               MENU_NTH_ITEM, i);
      /* Skip the title item: */
      if (!(bool) xv_get(menu_item, MENU_TITLE)
          && xv_get(menu_item, MENU_NOTIFY_PROC) != NULL)
          function_to_apply_on_each_menu_item(menu_item);
   }
}


void
disable_transform_selection()
{
   apply_on_each_transform_item(disable_menu_item);
}


void
enable_transform_selection()
{
   apply_on_each_transform_item(enable_menu_item);
}


void
transform_notify(Menu menu,
                 Menu_item menu_item)
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
         safe_apply(BUILDER_PRIVATIZE_MODULE, modulename);
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
      else if (strcmp(label, SUPPRESS_DEAD_CODE_TRANSFORM) == 0) {
         safe_apply(BUILDER_SUPPRESS_DEAD_CODE, modulename);
      }
      else if (strcmp(label, UNSPAGHETTIFY_TRANSFORM) == 0) {
         safe_apply(BUILDER_UNSPAGHETTIFY, modulename);
      }
      else if (strcmp(label, ATOMIZER_TRANSFORM) == 0) {
         safe_apply(BUILDER_ATOMIZER, modulename);
      }
      else if (strcmp(label, NEW_ATOMIZER_TRANSFORM) == 0) {
         safe_apply(BUILDER_NEW_ATOMIZER, modulename);
      }
      else if (strcmp(label, REDUCTIONS_TRANSFORM) == 0) {
         safe_apply(BUILDER_REDUCTIONS, modulename);
      }
      else {
         pips_error("transform_notify", "Bad choice");
      }
   }

   display_memory_usage();
}


void
create_transform_menu()
{
   edit_menu_item = 
      xv_create(NULL, MENUITEM, 
                MENU_STRING, "Edit",
                MENU_NOTIFY_PROC, edit_notify,
                MENU_RELEASE,
                NULL);
   
   transform_menu =
      xv_create(XV_NULL, MENU_COMMAND_MENU, 
                MENU_GEN_PIN_WINDOW, main_frame, "Transform Menu",
                MENU_TITLE_ITEM, "Apply a program transformation to a module ",
                MENU_ACTION_ITEM, ATOMIZER_TRANSFORM, transform_notify,
                MENU_ACTION_ITEM, SUPPRESS_DEAD_CODE_TRANSFORM, transform_notify,
                MENU_ACTION_ITEM, DISTRIBUTE_TRANSFORM, transform_notify,
                MENU_ACTION_ITEM, LOOP_INTERCHANGE_TRANSFORM, transform_notify,
                MENU_ACTION_ITEM, NEW_ATOMIZER_TRANSFORM, transform_notify,
                MENU_ACTION_ITEM, PARTIAL_EVAL_TRANSFORM, transform_notify,
                MENU_ACTION_ITEM, PRIVATIZE_TRANSFORM, transform_notify,
                MENU_ACTION_ITEM, REDUCTIONS_TRANSFORM, transform_notify,
                MENU_ACTION_ITEM, STRIP_MINE_TRANSFORM, transform_notify,
                MENU_ACTION_ITEM, UNROLL_TRANSFORM, transform_notify,
                MENU_ACTION_ITEM, UNSPAGHETTIFY_TRANSFORM, transform_notify,
                                 /* Just a separator: */
                MENU_ITEM, MENU_STRING, "--------", MENU_INACTIVE, TRUE,
                NULL,
                MENU_APPEND_ITEM, edit_menu_item,
                NULL);

   (void) xv_create(main_panel, PANEL_BUTTON,
                    PANEL_LABEL_STRING, "Transform/Edit",
                    PANEL_ITEM_MENU, transform_menu,
                    0);
}
