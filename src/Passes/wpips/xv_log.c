#include <stdio.h>
#include <varargs.h>
#include <errno.h>

#include <sys/time.h>
#include <sys/resource.h>
#include <xview/xview.h>
#include <xview/panel.h>
#include <xview/text.h>
#include <xview/notice.h> 
#include <xview/xv_error.h>
#include <setjmp.h>

#include "genC.h"
#include "misc.h"
#include "database.h"
#include "pipsdbm.h"

#include "wpips.h"

static Textsw log_textsw;
static Menu_item open_front, clear, close;

#define LOG_FILE "LOGFILE"
/* Par de'faut, le fichier est ferme' : */
static FILE *log_file = NULL;


  void
close_log_file()
{
  if (log_file != NULL && get_bool_property("USER_LOG_P")==TRUE)
    if (fclose(log_file) != 0) {
      perror("close_log_file");
      abort();
    }
  log_file = NULL;
}


  void
open_log_file()
{
  char file_name[MAXPATHLEN];

  if (log_file != NULL)
    close_log_file();

  if (get_bool_property("USER_LOG_P")==TRUE) {
    (void) strcpy(file_name,
		  concatenate(database_directory(db_get_current_workspace()),
			      "/",
			      LOG_FILE,
			      NULL));
    if ((log_file = fopen(file_name, "w")) == NULL) {
      perror("open_log_file");
      abort();
    }
  }
}

  void
log_on_file(char chaine[])
{
  if (log_file != NULL && get_bool_property("USER_LOG_P")==TRUE) {
    if (fprintf(log_file, "%s", chaine) <= 0) {
      perror("log_on_file");
      abort();
    }
    else
      fflush(log_file);
  }
}


int go_on_p(s)
char *s;
{
    int	result;
    Event e;

    result = notice_prompt(main_frame, &e,
			   NOTICE_MESSAGE_STRINGS,
			   s,
			   0,
			   NOTICE_BUTTON_YES,	"Yes",
			   NOTICE_BUTTON_NO,	"Cancel",
			   0);

    return(result == NOTICE_YES);
}

/*VARARGS0*/
/*
void prompt_user(va_alist)
va_dcl
{
    Event e;
    static char message_buffer[SMALL_BUFFER_LENGTH];
    va_list args;
    char *fmt;

    va_start(args);

    fmt = va_arg(args, char *);

    (void) vsprintf(message_buffer, fmt, args);

    va_end(args);

    (void) notice_prompt(xv_find(main_frame, WINDOW, 0), 
			 &e,
			 NOTICE_MESSAGE_STRINGS,
			 message_buffer,
			 0,
			 NOTICE_BUTTON_YES,	"Press Here",
			 0);
}
*/

void prompt_user(message)
string message;
{
    Event e;
    static char message_buffer[SMALL_BUFFER_LENGTH];

    (void) vsprintf(message_buffer, message);

    (void) notice_prompt(xv_find(main_frame, WINDOW, 0), 
			 &e,
			 NOTICE_MESSAGE_STRINGS,
			 message_buffer,
			 0,
			 NOTICE_BUTTON_YES,	"Press Here",
			 0);
}


/* function suppressed on 92.04 as we shifted to xview.3
Xv_error_action xview_error_recovery(object, severity, avlist)
Xv_object object;
Xv_error_severity severity;
Attr_avlist avlist[ATTR_STANDARD_SIZE];
{
    char buf[32];

    fprintf(stderr, "wpips error on object called %s\n", 
	    object==NULL ? "undefined_object" : 
	    (char *)xv_get(object, XV_NAME));

	    / *(Xv_pkg *) xv_get(object, XV_TYPE)* /

    if (severity == XV_ERROR_RECOVERABLE) {
	fprintf(stderr, "Dump core? (y/n) "), fflush(stderr);
	gets(buf);
	if (buf[0] == 'y' || buf[0] == 'Y' || buf[0] == NULL)
	    abort();
	else return XV_ERROR_CONTINUE;
    }
    abort();
}
*/

/*VARARGS0*/
void wpips_user_error_message(error_buffer)
char error_buffer[];
{
    char * perror_buffer = &error_buffer[0];
    int l = (int) xv_get(log_textsw, TEXTSW_LENGTH);
    extern jmp_buf pips_top_level;

    /* terminate PIPS request */
    if(get_bool_property("ABORT_ON_USER_ERROR"))
		abort();
    else {
		prompt_user("Something went wrong. Check the log window");

		xv_set(log_textsw, TEXTSW_INSERTION_POINT, l, NULL);
		textsw_insert(log_textsw, error_buffer, strlen(perror_buffer));
		textsw_possibly_normalize(log_textsw, 
			(Textsw_index) xv_get(log_textsw, TEXTSW_INSERTION_POINT));
		show_message(error_buffer);
		XFlush((Display *) xv_get(main_frame, XV_DISPLAY));
		xv_set(clear, MENU_INACTIVE, FALSE, 0);

		longjmp(pips_top_level, 1);
    }

    (void) exit(1);
}

void wpips_user_warning_message(warning_buffer)
char warning_buffer[];
{
    int l = (int) xv_get(log_textsw, TEXTSW_LENGTH);

    log_on_file(warning_buffer);
    xv_set(log_textsw, TEXTSW_INSERTION_POINT, l, NULL);
    textsw_insert(log_textsw, warning_buffer, strlen(warning_buffer));
    textsw_possibly_normalize(log_textsw, 
		(Textsw_index) xv_get(log_textsw, TEXTSW_INSERTION_POINT));
    show_message(warning_buffer);
    XFlush((Display *) xv_get(main_frame, XV_DISPLAY));
    xv_set(clear, MENU_INACTIVE, FALSE, 0);
}


#define MAXARGS     100
/*VARARGS0*/
/*
void log_execl(va_alist)
va_dcl
{
    FILE *fd;
    char buffer[256];

    char command_buffer[SMALL_BUFFER_LENGTH];
    va_list args;
    char *s;

    va_start(args);

    strcpy(command_buffer, "");
    while ((s = va_arg(args, char *)) != (char *) NULL) {
	strcat(command_buffer, s);
	strcat(command_buffer, " ");
    }
    va_end(args);

    strcat(command_buffer, " 2>&1");
    if ((fd = popen(command_buffer, "r")) == NULL) {
		fprintf(stderr, "could not popen a new process\n");
		exit(1);
    }

    while (fgets(buffer, 256, fd) != NULL) {
	textsw_insert(log_textsw, buffer, strlen(buffer));
	textsw_possibly_normalize(log_textsw, 
		(Textsw_index) xv_get(log_textsw, TEXTSW_INSERTION_POINT));
	show_message(buffer);
	XFlush((Display *) xv_get(main_frame, XV_DISPLAY));
    }
    pclose(fd);

    xv_set(clear, MENU_INACTIVE, FALSE, 0);
}
*/

void wpips_user_log(fmt, args)
string fmt;
va_list args;
{
    int l = (int) xv_get(log_textsw, TEXTSW_LENGTH);
    static char log_buffer[SMALL_BUFFER_LENGTH];

    if(get_bool_property("USER_LOG_P")==FALSE)
	return;

    (void) vsprintf(log_buffer, fmt, args);

    log_on_file(log_buffer);
    xv_set(log_textsw, TEXTSW_INSERTION_POINT, l, NULL);
    textsw_insert(log_textsw, log_buffer, strlen(log_buffer));
    textsw_possibly_normalize(log_textsw, 
		(Textsw_index) xv_get(log_textsw, TEXTSW_INSERTION_POINT));
    show_message(log_buffer);
    XFlush((Display *) xv_get(main_frame, XV_DISPLAY));
    XFlush((Display *) xv_get(log_frame, XV_DISPLAY));
    xv_set(clear, MENU_INACTIVE, FALSE, 0);
}



void open_log_subwindow(menu, menu_item)
Menu menu;
Menu_item menu_item;
{
    xv_set(open_front, MENU_STRING, "Front", 0);
    xv_set(close, MENU_INACTIVE, FALSE, 0);
    unhide_window(log_frame);
}



void clear_log_subwindow(menu, menu_item)
Menu menu;
Menu_item menu_item;
{
    int l = (int) xv_get(log_textsw, TEXTSW_LENGTH);
    textsw_delete(log_textsw, 0, l);
    xv_set(clear, MENU_INACTIVE, TRUE, 0);
}



void close_log_subwindow(menu, menu_item)
Menu menu;
Menu_item menu_item;
{
    xv_set(open_front, MENU_STRING, "Open", 0); /*MENU_INACTIVE, FALSE, 0);*/
    xv_set(close, MENU_INACTIVE, TRUE, 0);
    hide_window(log_frame);
}



void create_log_menu()
{
    Menu menu;

    open_front = xv_create(NULL, MENUITEM, 
		     MENU_STRING, "Open",
		     MENU_NOTIFY_PROC, open_log_subwindow,
		     MENU_RELEASE,
		     NULL);

    clear = xv_create(NULL, MENUITEM, 
		      MENU_STRING, "Clear",
		      MENU_NOTIFY_PROC, clear_log_subwindow,
		      MENU_INACTIVE, TRUE,
		      MENU_RELEASE,
		      NULL);

    close = xv_create(NULL, MENUITEM, 
		      MENU_STRING, "Close",
		      MENU_NOTIFY_PROC, close_log_subwindow,
		      MENU_INACTIVE, TRUE,
		      MENU_RELEASE,
		      NULL);

    menu = xv_create(XV_NULL, MENU_COMMAND_MENU, 
		     MENU_APPEND_ITEM, open_front,
		     MENU_APPEND_ITEM, clear,
		     MENU_APPEND_ITEM, close,
		     NULL);

    (void) xv_create(main_panel, PANEL_BUTTON,
		     PANEL_LABEL_STRING, "Log  ",
		     PANEL_ITEM_MENU, menu,
		     0);
}



void create_log_window()
{
    /* Xv_Window window; */


    log_textsw = (Xv_Window) xv_create(log_frame, TEXTSW, 0);
/* recuperation d'event ne fonctionne pas -> installer TEXTSW_NOTIFY_PROC, */
/* autre suggestion: mettre un masque X */

/*    window = (Xv_Window) xv_find(log_frame, WINDOW, 0);

    xv_set(window, 
	   WIN_CONSUME_X_EVENT_MASK, EnterWindowMask, 
	   WIN_EVENT_PROC, default_win_interpose, 
	   NULL);
*/
}
