#include <stdio.h>
#include <fcntl.h>
#include <string.h>
#include <stdarg.h>
#include <setjmp.h>
#include <stdlib.h>

#include "genC.h"
#include "ri.h"
#include "database.h"
#include "graph.h"
#include "makefile.h"

#include "misc.h"
#include "ri-util.h"
#include "pipsdbm.h"

#include "constants.h"
#include "resources.h"
#include "pipsmake.h"

#include "top-level.h"

#include "tpips.h"

#define MAX_ARGS 128

#define TPIPS_PROMPT "pips> "

typedef struct {
    char *cde;
    void (*fct)();
} binding;

void update_props()
{
}

void clear_props()
{
}

void prompt_user(char * fmt, ...)
{
    va_list args;
    va_start(args, fmt);

    (void) vfprintf(stderr, fmt, args);

    va_end(args);

    fprintf(stderr, "Press <Return> to continue ");
    while (getchar() != '\n') ;
}


/* show_message() is equivalent to user_log() in tpips.
 * Therefore calls should be replaced. The problem comes from common.c, in
 * which show_message has its sens for wpips.
 * Functions in common.c should be writen in a `top level library'
 * 27.03.91 BB
 */

/*VARARGS0*/
void show_message(char * fmt, ...)
{
    va_list args;
    va_start(args, fmt);

    (void) vfprintf(stdout, fmt, args);

    va_end(args);
}


#define MAXARGS     100
#define LGBUF 256
char *ask()
{
    static char buffer[LGBUF+1];
    register int i;

    while (1) {
	int c;

	fprintf(stdout, TPIPS_PROMPT);

	for (i = 0; i < LGBUF && (c = getc(stdin)) != EOF; i++) {
	    buffer[i] = c;
	    if (c == '\n')
		break;
	}

	if (c == EOF)
	    return(NULL);

	if (i == LGBUF) {
	    fprintf(stdout, "Command line too long - try again\n");
	}
	else {
	    buffer[i] = '\0';
	    return(buffer);
	}
    }
}

void display_help(argc, argv)
int argc;
char *argv[];
{
    char *help_list[ARGS_LENGTH];
    int help_list_length = 0;

    int i;

    if (argc == 1) {
	get_help_topics(&help_list_length, help_list);
	show_message("Type 'display-help topic' with 'topic' in:\n");
    }
    else {
	get_help_topic(argv[1], &help_list_length, help_list);
    }

    for (i = 0; i < help_list_length; i++) {
	fprintf(stderr, "\t%s\n", help_list[i]);
    }
}



void display_status(argc, argv)
int argc;
char *argv[];
{
    char *s;
    static char *none = "none";

    fprintf(stderr, "Current directory: %s\n", get_cwd());

    if ((s = db_get_current_program_name()) == NULL)
	s = none;
    fprintf(stderr, "Current program  : %s\n", s);

    if ((s = db_get_current_module_name()) == NULL)
	s = none;
    fprintf(stderr, "Current module   : %s\n", s);
}



void program_create(argc, argv)
int argc;
char *argv[];
{
    int nargc;

    if (argc <= 2) {
	show_message("Correct syntax is:   %s\n", 
			"program-create name file.f ...");
	return;
    }

    nargc = argc-2;

    db_create_program(argv[1]);

    create_program(&nargc, argv+2);

    /*** parse_makefile() to make sure of saving a defined makefile ***/
    parse_makefile();
}



void program_open(argc, argv)
int argc;
char *argv[];
{
    if (argc != 2) {
	show_message("Correct syntax is:   %s\n",
			"program-open name");
	return;
    }

    open_program(argv[1]);
}



void program_close(argc, argv)
int argc;
char *argv[];
{
    if (argc != 1) {
	show_message("Correct syntax is:   %s\n", 
		     "program-close");
	return;
    }

    close_program();
}



void quit(argc, argv)
int argc;
char *argv[];
{
    close_program();
    exit(0);
}



void module_open(argc, argv)
int argc;
char *argv[];
{
    if (argc == 1) {
	char *module_list[ARGS_LENGTH];
	int  module_list_length = 0, i;

	show_message("Type 'module-open name' with 'name' in:\n");

	db_get_module_list(&module_list_length, module_list);
	for (i = 0; i < module_list_length; i++) {
	    fprintf(stderr, "\t%s\n", module_list[i]);
	}
	args_free(&module_list_length, module_list);

	return;
    }

    open_module(argv[1]);
}



void module_view(argc, argv)
int argc;
char *argv[];
{
    char *modulename = db_get_current_module_name();

    if(modulename != NULL) {
    char *filename =
	    db_get_file_resource(DBR_SOURCE_FILE, modulename, TRUE);

	if(filename != NULL)
	    system(concatenate("cat ", filename, NULL));
    }
}



void module_edit(argc, argv)
int argc;
char *argv[];
{
    char *modulename = db_get_current_module_name();
    char *filename = 
	db_get_file_resource(DBR_SOURCE_FILE, modulename, TRUE);

    show_message("Starting emacs ...");
    system(concatenate("emacs ", filename, NULL));
    show_message("\n");
}



void view(argc, argv)
int argc;
char *argv[];
{
    char *print_type;
    char * filename;

    if (strcmp(argv[0], "view") == 0) {
	print_type = DBR_PRINTED_FILE;
    }
    else if (strcmp(argv[0], "view-parallel") == 0) {
	print_type = DBR_PARALLELPRINTED_FILE;
    }
    else {
	pips_error("view", "bad function name : %s\n", argv[0]);
    }

    if((filename=build_view_file(print_type)) != NULL)
	system(concatenate("cat ", filename, NULL));
}



static binding binds[] = {
    {"display-status",      display_status},
    {"display-help",        display_help},
    {"program-create",      program_create},
    {"program-open",        program_open},
    {"program-close",       program_close},
    {"create-program",      program_create},
    {"open-program",        program_open},
    {"close-program",       program_close},
    {"quit",                quit},
    {"module-open",         module_open},
    {"module-view",         module_view},
    {"module-edit",         module_edit},
    {"open-module",         module_open},
    {"view-module",         module_view},
    {"edit-module",         module_edit},
    {"view",                view},
    {"view-parallel",       view},
    {"distribute",          distribute},
    {NULL,	   NULL}
};



main(argc, argv)
int argc;
char *argv[];
{
    extern jmp_buf pips_top_level;

    debug_on("PIPS_DEBUG_LEVEL");

    initialize_newgen();

    (void) setjmp(pips_top_level);

    while (1) {
	binding *pb;
	char *argv[MAX_ARGS];
	int argc, iarg;

	static char system_command[LGBUF+1];
	char *cmd;

	if ((cmd = ask()) == NULL) {
	    fprintf(stdout, "\n");
	    exit(0);
	}

	strcpy(system_command, cmd);

	for (iarg = 0; iarg < MAX_ARGS; iarg++)
	    argv[iarg] = NULL;

	argv[0] = strtok(cmd, " \t");

	if (argv[0] == NULL)
	    continue;

	argc = 1;
	while ((argv[argc] = strtok(NULL, " \t")) != NULL) {
	    if (++argc >= MAX_ARGS) {
		break;
	    }
	}

	if (argc >= MAX_ARGS) {
	    fprintf(stderr, "Too many arguments - Try again\n");
	    continue;
	}

	for (pb = binds; pb->cde != NULL; pb++) {
	    if (strcmp(pb->cde, argv[0]) == 0) {
		(pb->fct)(argc, argv);

		break;
	    }
	}

	if (pb->cde == NULL) {
	    if (strcmp(argv[0], "!") == 0) {
		*(strchr(system_command, '!')) = ' ';
		system(system_command);
	    }
	    else {
		fprintf(stderr, "Unknown command: %s\n", argv[0]);
	    }
	}
    }
}
