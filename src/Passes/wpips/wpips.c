#include <stdio.h>
extern int fprintf();
#include <malloc.h>

#include <sys/time.h>
#include <sys/resource.h>
#include <xview/xview.h>
#include <xview/panel.h>
#include <xview/text.h>
#include <xview/svrimage.h>
#include <types.h>

#include "genC.h"
#include "misc.h"

#include "ri.h"
#include "database.h"
#include "pipsdbm.h"

#include "wpips.h"
#include "xv_sizes.h"

#include "constants.h"

#include "top-level.h"


extern void(* pips_error_handler)();
extern void(* pips_warning_handler)();
extern void(* pips_log_handler)(char * fmt, va_list args);
extern void(* pips_update_props_handler)();


Frame main_frame, 
    schoose_frame, 
    mchoose_frame, 
    log_frame, 
    edit_frame[MAX_NUMBER_OF_WPIPS_WINDOWS], 
    help_frame, 
    query_frame,
    properties_frame;

Panel main_panel,
    status_panel,
    query_panel,
    mchoose_panel,
    schoose_panel,
    help_panel;



void create_menus()
{
    create_select_menu();
    create_props_menu_and_window();
    create_edit_menu();
/*    create_analyze_menu();*/
    create_transform_menu();
    create_help_menu();
    create_log_menu();
    create_quit_button();
}


static int first_mapping = TRUE;

void main_event_proc(window, event)
Xv_Window window;
Event *event;
{
    if (first_mapping == TRUE && event_id(event)==32526) {
		first_mapping = FALSE;

		/* we place all frames */
		place_frames();
    };
}


void create_main_window()
{
    /* Xv_window main_window; */

    main_panel = xv_create(main_frame, PANEL,
			   NULL);

    /* The following mask is necessary to avoid that key events occure twice.
     */
    /*    main_window = (Xv_Window) xv_find(main_frame, WINDOW, 0);
	  xv_set(main_window,
	  WIN_IGNORE_EVENT, WIN_UP_ASCII_EVENTS, 
	  NULL);
	  */
    /* commented out as we shifted to xview.3, 92.04 */
/*    xv_set(canvas_paint_window(main_panel),
	   WIN_IGNORE_EVENT, WIN_UP_ASCII_EVENTS,
	   WIN_CONSUME_EVENT, 32526, 
	   WIN_EVENT_PROC, main_event_proc,
	   NULL);
*/
}


short pips_bits[] = {
#include "pips.icon"
};

void create_icon()
{
    Server_image pips_image;
    Icon icon;
    Rect rect;

    pips_image = (Server_image)xv_create(NULL, SERVER_IMAGE, 
					 XV_WIDTH, 64,
					 XV_HEIGHT, 64,
					 SERVER_IMAGE_BITS, pips_bits, 
					 NULL);
    icon = (Icon)xv_create(NULL, ICON, 
			   ICON_IMAGE, pips_image,
			   NULL);
    rect.r_width= (int)xv_get(icon, XV_WIDTH);
    rect.r_height= (int)xv_get(icon, XV_HEIGHT);
    rect.r_left= 0;
    rect.r_top= 0;

    xv_set(main_frame, FRAME_ICON, icon, 
	   FRAME_CLOSED_RECT, &rect,
	   NULL);
}


void parse_arguments(argc, argv)
int argc;
char *argv[];
{
    string pname = NULL, mname = NULL, *files = NULL;
    int iarg = 1, nfiles;

    while (iarg < argc) {
	if (same_string_p(argv[iarg], "-workspace")) {
	    argv[iarg] = NULL;
	    pname = argv[++iarg];
	}
	else if (same_string_p(argv[iarg], "-module")) {
	    argv[iarg] = NULL;
	    mname = argv[++iarg];
	}
	else if (same_string_p(argv[iarg], "-files")) {
	    argv[iarg] = NULL;
	    files = &argv[++iarg];
	    nfiles = argc-iarg;
	}
	else {
	    if (argv[iarg][0] == '-') {
		fprintf(stderr, "Usage: %s ", argv[0]);
		fprintf(stderr, "[ X-Window options ] ");
		fprintf(stderr, "[ -workspace name [ -module name ] ");
		fprintf(stderr, "[ -files file1.f file2.f ... ] ]\n");
		exit(1);
	    }
	}

	iarg += 1;
    }

    if (pname != NULL) {
	if (files != NULL) {
	    db_create_program(pname);
	    create_program(&nfiles, files);
	}
	else {
	    open_program(pname);
	}

	if (mname != NULL) {
	    open_module(mname);
	}
    }
}



int main(argc,argv)
int argc;
char *argv[];
{
  pips_warning_handler = wpips_user_warning;
  pips_error_handler = wpips_user_error;
  pips_log_handler = wpips_user_log;
  pips_update_props_handler = update_props;

  /* Added for debug. RK, 8/06/93. */
  malloc_debug(1);

  initialize_newgen();

  debug_on("WPIPS_DEBUG_LEVEL");

  /* we parse command line arguments */
  /* XV_ERROR_PROC unset as we shifted to xview.3, Apr. 92 */
  xv_init(XV_INIT_ARGC_PTR_ARGV, &argc, argv, 
	  /* XV_ERROR_PROC, xview_error_recovery, */
	  0);

  /* we parse remaining command line arguments */
  parse_arguments(argc, argv);

  /* we create all frames */
  create_frames();

  create_main_window();

  create_help_window();

  create_schoose_window();

  create_mchoose_window();

  create_query_window();

  create_edit_window();

  create_log_window();

  create_menus();

  create_status_subwindow();

  create_icons();
  /*    create_icon();*/

  /* Call added. RK, 9/06/1993. */
  place_frames();

  xv_main_loop(main_frame);

  close_log_file();
  exit(0);
}
