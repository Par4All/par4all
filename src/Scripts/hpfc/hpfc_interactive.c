/* $RCSfile: hpfc_interactive.c,v $ (version $Revision$)
 * $Date: 1995/04/25 10:33:22 $, 
 */

#include <stdio.h>
#include <strings.h>

extern int system();
extern int fprintf();
extern char *getenv();
extern int malloc();
extern void free();

#include "readline/readline.h"
#include "readline/history.h"

#define HPFC_PROMPT "hpfc> "
#define BUFFER_SIZE 1024
#define DEFAULT_FILE_NAME ".hpfc.history"

#define QUIT "qui"

static char * default_full_file_name()
{
    char *home = getenv("HOME");
    int    len = strlen(home) + strlen(DEFAULT_FILE_NAME) + 2;
    char *name = (char*) malloc(sizeof(char)*len);

    return(sprintf(name, "%s/%s", home, DEFAULT_FILE_NAME));
}

int main()
{
    char 
	*line,
	buffer[BUFFER_SIZE],
	*file_name = default_full_file_name();
    
    /*  initialize history
     */
    using_history();
    read_history(file_name);
    
    /*  interactive loop
     */
    while ((line=readline(HPFC_PROMPT)))
    {
	if ((strlen(line)+strlen(HPFC_PROMPT))>=BUFFER_SIZE) /* woh! */
	{
	    fprintf(stderr, "line too long\n");
	    continue;
	}	    

	if (strncmp(line, QUIT, 3)==0) /* quit! */
	    break;

	/*   calls a script:-)
	 *   All this stuff could be taken care of in the C, but shell
	 *   scripts are much easier to develop. 
	 */
	system(sprintf(buffer, "hpfc %s", line));

	/*   maybe a memory leak? I guess not.
	 */
	add_history(line); 
    }

    /*   close history
     */
    write_history(file_name);
    history_truncate_file(file_name, 100);

    fprintf(stdout, "\n"); /* usefull for Ctrl-D terminations */
    free(file_name);
    return(0);
}

/*   that is all
 */
