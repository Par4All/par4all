/* $RCSfile: hpfc_interactive.c,v $ (version $Revision$)
 * $Date: 1995/04/24 16:31:46 $, 
 */

#include <stdio.h>
#include <strings.h>

extern int system();
extern int fprintf();
extern char *getenv();

#include "readline/readline.h"
#include "readline/history.h"

#define HPFC_PROMPT "hpfc> "
#define BUFFER_SIZE 1024
#define DEFAULT_FILE_NAME ".hpfc.history"

#define QUIT "qui"

int main()
{
    char 
	*line,
	buffer[BUFFER_SIZE],
	*home = getenv("HOME"),
	file_name[BUFFER_SIZE]="";

    /*  default history file is ~/.hpfc.history
     */
    if ((strlen(home)+strlen(DEFAULT_FILE_NAME))<BUFFER_SIZE)
	sprintf(file_name, "%s/%s", home, DEFAULT_FILE_NAME);
    
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
    return(0);
}

/*   that is all
 */
