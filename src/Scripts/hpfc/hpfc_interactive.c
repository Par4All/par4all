/* $RCSfile: hpfc_interactive.c,v $ (version $Revision$)
 * $Date: 1995/08/08 10:54:37 $, 
 *
 * interactive interface to hpfc,
 * based on the GNU READLINE library for interaction,
 * and the associated HISTORY library for history.
 * taken from bash.
 */

#include <stdio.h>
#include <stdlib.h>
#include <strings.h>

extern int system();
extern int fprintf();
extern int chdir();

#include "readline/readline.h"
#include "readline/history.h"

#define HPFC_PROMPT "hpfc> " 		/* prompt for readline  */
#define HPFC_PREFIX "hpfc"   		/* forked shell script  */
#define HPFC_HISTENV "HPFC_HISTORY"	/* history file env variable */
#define HPFC_HISTORY_LENGTH 100		/* max length of history file */
#define HIST ".hpfc.history" 		/* default history file */

#define SHELL_ESCAPE "\\" 		/* ! used for history reference */
#define CHANGE_DIR   "cd "
#define QUIT         "quit"

/* the lexer is quite simple:-)
 */
#define PREFIX_EQUAL_P(str, prf) (strncmp(str, prf, strlen(prf))==0)

/*  returns the full hpfc history file name, i.e.
 *  - $HPFC_HISTORY (if any)
 *  - $HOME/"HIST"
 */
static char *default_hist_file_name()
{
    char *home, *hist = getenv(HPFC_HISTENV);

    if (hist) return hist;

    /* else builds the default name.
     */
    home = getenv("HOME");
    return sprintf((char*) malloc(sizeof(char)*(strlen(home)+strlen(HIST)+2)),
		   "%s/%s", home, HIST);
}

/* main: interactive loop and history management.
 */
int main()
{
    HIST_ENTRY * last_entry;
    char *last, *line, *file_name = default_hist_file_name();
    
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
    last = last_entry ? last_entry->line : NULL ;
    
    /*  interactive loop
     */
    while ((line = readline(HPFC_PROMPT)))
    {
	if (PREFIX_EQUAL_P(line, QUIT)) 
	    break;
	else if (PREFIX_EQUAL_P(line, CHANGE_DIR))
	{
	    if (chdir(line+strlen(CHANGE_DIR)))
		fprintf(stderr, "error while changing directory\n");
	}
	else if (PREFIX_EQUAL_P(line, SHELL_ESCAPE))
	{
	    /*   the shell escape is directly executed. That easy!
	     */
	    system(line+strlen(SHELL_ESCAPE));
	}
	else
	{
	    /*   calls a script:-)
	     *   All this stuff could be taken care of in C, but shell
	     *   scripts are much easier to develop:-) 
	     */
	    char *shll = (char*)
		malloc(sizeof(char)*(strlen(HPFC_PREFIX)+strlen(line)+2));

	    system(sprintf(shll, "%s %s", HPFC_PREFIX, line));

	    free(shll);
	}

	/*   add to history if not the same as the last one
	 */
	if (line && *line && ((last && strcmp(last, line)!=0) || (!last)))
	    add_history(line), last = line; 
	else
	    line = (char*) ((line) ? (free(line), NULL) : NULL);
    }

    if (!line) fprintf(stdout, "\n"); /* for Ctrl-D terminations */

    /*   close history: truncate list and write history file
     */
    stifle_history(HPFC_HISTORY_LENGTH);
    write_history(file_name);

    return 0;
}

/*   that is all
 */
