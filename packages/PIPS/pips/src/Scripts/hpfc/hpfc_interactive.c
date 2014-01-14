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
/*
 * interactive interface to hpfc,
 * based on the GNU READLINE library for interaction,
 * and the associated HISTORY library for history.
 * taken from bash. it's definitely great.
 */

#include <stdio.h>
#include <stdlib.h>
#include <strings.h>
#include <unistd.h>

#include <readline/readline.h>
#include <readline/history.h>

#define HPFC_PROMPT "hpfc> " 		/* prompt for readline  */
#define HPFC_SHELL "hpfc"   		/* forked shell script  */
#define HPFC_HISTENV "HPFC_HISTORY"	/* history file env variable */
#define HPFC_HISTORY_LENGTH 100		/* max length of history file */
#define HIST ".hpfc.history" 		/* default history file */

#define SHELL_ESCAPE "\\" 		/* ! used for history reference */
#define CHANGE_DIR   "cd "
#define QUIT         "quit"

/*  returns the full hpfc history file name, i.e.
 *  - $HPFC_HISTORY (if any)
 *  - $HOME/"HIST"
 */
static char * default_hist_file_name(void)
{
    char *home, *hist = getenv(HPFC_HISTENV), *name;

    if (hist) return hist;

    /* else builds the default name. memory leak.
     */
    home = getenv("HOME");
    name = (char*) malloc(sizeof(char)*(strlen(home)+strlen(HIST)+2));
    sprintf(name, "%s/%s", home, HIST);
    return name;
}

static char * initialize_hpfc_history(void)
{
    HIST_ENTRY * last_entry;
    char *file_name = default_hist_file_name();
    
    /*  initialize history: 
     *  read the history file, then point to the last entry.
     */
    using_history();
    read_history(file_name);
    history_set_pos(history_length);
    last_entry = previous_history();

    /* last points to the last history line of any.
     * used to avoid to put twice the same line.
     */
    return last_entry ? last_entry->line : NULL ;
}


/* Handlers
 */
void cdir_handler(const char * line)
{
    if (chdir(line+strlen(CHANGE_DIR)))
	fprintf(stderr, "error while changing directory\n");
}

void shell_handler(const char * line)
{
    system(line+strlen(SHELL_ESCAPE));
}

void quit_handler(const char __attribute__ ((unused)) * line)
{
    char *file_name = default_hist_file_name();

    /* close history: truncate list and write history file
     */
    stifle_history(HPFC_HISTORY_LENGTH);
    write_history(file_name);
    
    exit(0);
}

void default_handler(const char * line)
{
    char *shll = 
	(char*) malloc(sizeof(char)*(strlen(HPFC_SHELL)+strlen(line)+2));
    (void)sprintf(shll, "%s %s", HPFC_SHELL, line);
    system(shll);
    free(shll);
}

struct t_handler 
{
    char * name;
    void (*function)(const char *);
} ;

static struct t_handler handlers[] =
{
  { QUIT,		quit_handler },
  { CHANGE_DIR, 	cdir_handler },
  { SHELL_ESCAPE, 	shell_handler },
  { (char *) NULL, 	default_handler}
};

/* the lexer is quite simple:-)
 */
#define PREFIX_EQUAL_P(str, prf) (strncmp(str, prf, strlen(prf))==0)

static void (*find_handler(const char * line))(const char *)
{
    struct t_handler * x = handlers;
    while ((x->name) && !PREFIX_EQUAL_P(line, x->name)) x++;
    return x->function;
}

/* MAIN: interactive loop and history management.
 */
int main(void)
{
    char *last, *line;

    last = initialize_hpfc_history();

    /*  interactive loop
     */
    while ((line = readline(HPFC_PROMPT)))
    {
	/*   calls the appropriate handler.
	 */
	(find_handler(line))(line);

	/*   add to history if not the same as the last one.
	 */
	if (line && *line && ((last && strcmp(last, line)!=0) || (!last)))
	    add_history(line), last = line; 
	else
	    free(line);
    }

    fprintf(stdout, "\n"); /* for Ctrl-D terminations */
    quit_handler(line);
    return 0; 
}

/*   that is all
 */
