/* $RCSfile: hpfc_interactive.c,v $ (version $Revision$)
 * $Date: 1995/04/24 13:13:35 $, 
 */

#include <stdio.h>
#include <strings.h>

extern int system();
extern int fprintf();

#include "readline/readline.h"
#include "readline/history.h"

#define HPFC_PROMPT "hpfc> "
#define BUFFER_SIZE 1024
#define FILE_NAME NULL

#define QUIT "qui"

int main()
{
    char *line, buffer[BUFFER_SIZE];
    
    using_history();
    read_history(FILE_NAME);
    
    while ((line=readline(HPFC_PROMPT)))
    {
	if (strncmp(line, QUIT, 3)==0) 
	    break;

	system(sprintf(buffer, "hpfc %s", line));
	add_history(line);
    }

    write_history(FILE_NAME);

    fprintf(stdout, "\n");
    return(0);
}
