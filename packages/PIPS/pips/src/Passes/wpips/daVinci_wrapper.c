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
/* All the stuff to use the graph viewer daVinci from PIPS.

   Ronan.Keryell@cri.ensmp.fr
*/

#include <stdlib.h>
#include <unistd.h>
#include <stdarg.h>
#include <signal.h>
#include <sys/wait.h>
#include <errno.h>

#include "genC.h"

#include "constants.h"
#include "misc.h"

/* To monitor if we have already running daVinci: */
pid_t daVinci_pid = 0;

/* The daVinci context control: */
int daVinci_current_context = 0;
int daVinci_next_context_to_create = 0;

/* To process some eventual handler somewhere else: */
void ( * old_SIGCHLD_handler)() = NULL;

/* The pipes to communicate with daVinci: */
FILE * write_to_daVinci_stream;
int read_from_daVinci_fd;

static string
read_answer_from_daVinci()
{
    static char * command = NULL;
    static int command_size = 0;
    int position = 0;
    bool backslash_pending_p = FALSE;
    
    debug_on("DAVINCI_DEBUG_LEVEL");
    
    if (command_size == 0) {
	/* Allocate some place the first time: */
	command_size = 10;
	command = malloc(10);
    }
    
    /* A "\n" is used by daVinci to end answers: */
    for(;;) {
	char a_character;
	/* Not optimized: read characters one by one... */
	int length = read(read_from_daVinci_fd, &a_character, 1);
	if (length == 0)
	    continue;
	
	if (length == 1) {
	    pips_debug(8, "Read character \"%c\" (%d)\n",
		       a_character, a_character);
	    if (a_character == '\n')
		/* End of answer found: */
		break;

	    /* Deal with some '\' forms in strings: */
	    if (backslash_pending_p) {
		backslash_pending_p = FALSE;
		if (a_character == 'n')
		    a_character = '\n';
		/* Else, '\\' -> '\', '\"' -> '"' without doing
                   anything. */
	    }
	    if (position == command_size - 2) {
		/* No more place in the command buffer: */
		command_size *= 2;
		command = realloc(command, command_size);
	    }
	    command[position++] = a_character;
	}
    }
    /* To mark the end to ease later parsing: */
    command[position++] = '\001';
    command[position] = '\0';
    
    debug_off();
    return command;
}


static void
parse_daVinci_answer()
{
    char * command = read_answer_from_daVinci();
    char * buffer = malloc(strlen(command));
    
    for(;;) {
	if (strcmp(command, "ok\001")) {
	    pips_debug(8, "\"ok\" parsed\n");
	    break;
	}
	
	/* I love scanf-based parser... :-) */
	if (sscanf(command, "context(\"Context_%d\")\001",
		   &daVinci_current_context) == 1) {
	    pips_debug(8, "Current context set to %d\n",
		       daVinci_current_context);
	    break;
	}
	
	if (sscanf(command, "communication_error(\"%s\")\001",
		   buffer) == 1) {
	    user_warning("daVinci said...",
			 "communication_error(\"%s\")", buffer);
	    break;
	}
    }
}


/* Send a command to daVinci with an à la printf syntax: */
static void
send_command_to_daVinci(string command_format, ...)
{
    va_list some_arguments;

    va_start(some_arguments, command_format);
    vfprintf(write_to_daVinci_stream, command_format, some_arguments);
    va_end(some_arguments);
    
    /* A "\n" is used by daVinci to end commands: */
    fprintf(write_to_daVinci_stream, "\n");
    fflush(write_to_daVinci_stream);
}


static void
monitor_daVinci(int sig,
		int code,
		struct sigcontext * scp,
		char * addr)
{
    pid_t pid;
    int statusp;
    
    debug_on("DAVINCI_DEBUG_LEVEL");
    pid = waitpid(daVinci_pid, &statusp, WNOHANG);
    
    pips_debug(5, "waitpid -> %d\n", pid);

    if (pid == -1) {
	/* Not about daVinci, pass to the old handler: */
	debug_off();
	( * old_SIGCHLD_handler)(sig, code, scp, addr);
    }

    /* If the calling process is stopped, pid = 0: */
    if (pid != 0 && (statusp & 255) != 127) {
	/* Well, daVinci is no longer here... */
	pips_debug(5, "daVinci pid %d exited with status %x\n",
		   daVinci_pid, statusp);
	daVinci_pid = 0;
	/* Restore the old handler: */
	(void) signal(SIGCHLD, old_SIGCHLD_handler);
	old_SIGCHLD_handler = NULL;
    }
    else {
	/* Else, the process may have stopped: nothing to do. */
    }
    debug_off();
}


void
start_daVinci_if_not_running()
{
    int pips_output_fd[2], pips_input_fd[2];
    
    debug_on("DAVINCI_DEBUG_LEVEL");
    if (daVinci_pid != 0) {
	/* It is not necessary to start another daVinci: */
	debug_off();
	return;
    }
    
    /* Create the 2 pipes to communicate with daVinci.

       Hum, need some portability adjustment for SVR4... */
    if (pipe(pips_output_fd) != 0 || pipe(pips_input_fd) != 0) {
	perror("fork");
	pips_error("start_daVinci_if_not_running",
		   "Cannot create the 2 pipes.\n");
    }
    
    daVinci_pid = fork();
    pips_debug(5, "daVinci_pid = %d\n", daVinci_pid);

    if (daVinci_pid == -1) {
	perror("fork");
	pips_error("start_daVinci_if_not_running",
		   "Cannot fork a process.\n");
    }
    
    if (daVinci_pid == 0) {
	/* This is the child: */
	/* Connect the PIPS pipes to stdin and stdout of daVinci: */
	if (dup2(pips_output_fd[0], fileno(stdin)) == -1
	    || dup2(pips_input_fd[1], fileno(stdout)) == -1) {
	    perror("dup2");
	    pips_error("start_daVinci_if_not_running",
		       "Cannot dup file descriptors.\n");
	}
	
	execlp("daVinci", "daVinci", "-pipe", NULL);
	perror("execl of daVinci");
	pips_error("start_daVinci_if_not_running",
		   "Cannot start the daVinci process.\n");
	/* Hum, the PIPS parent will go on... */
    }
    else {
	/* The parent: */
	write_to_daVinci_stream = fdopen(pips_output_fd[1], "w");
	if (write_to_daVinci_stream == NULL)
	    pips_error("start_daVinci_if_not_running",
		       "Cannot fdopen pips_output_fd[1].\n");
	
	read_from_daVinci_fd = pips_input_fd[0];

	/* Install the handler to monitor daVinci: */
	old_SIGCHLD_handler = signal(SIGCHLD, monitor_daVinci);

	/* Wait for the OK stuff. */
	parse_daVinci_answer();
    }
    debug_off();
}


void
create_daVinci_new_context()
{
    start_daVinci_if_not_running();
    (void) printf("%s\n", read_answer_from_daVinci());
    send_command_to_daVinci("multi(open_context(\"Context_%d\"))",
			    daVinci_next_context_to_create++);
}

void
send_graph_to_daVinci(string graph_file_name)
{
    create_daVinci_new_context();
    (void) printf("%s\n", read_answer_from_daVinci());
    (void) printf("%s\n", read_answer_from_daVinci());

    /* Send the graph: */
    send_command_to_daVinci("menu(file(open_graph(\"%s\")))",
			    graph_file_name);
}


main ()
{
    /*
    start_daVinci_if_not_running();
    send_command_to_daVinci("multi(open_context(\"Context_1\"))");
    */
    send_graph_to_daVinci("ESSAI.pref-daVinci%");
    send_graph_to_daVinci("ESSAI.pref-daVinci");
    for (;;)
	(void) printf("%s\n", read_answer_from_daVinci());
    
    system("sleep 100");
    exit(0);
}
