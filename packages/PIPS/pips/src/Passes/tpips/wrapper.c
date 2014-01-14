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
 * UNIX wrapper around tpips for jpips, for signal management.
 * Under special comments on the pipe, signals are sent to tpips.
 * I could not use popen() because I found no way to access the pid
 * of the child process. Basically this is a re-implementation of popen().
 *
 * stdin TPIPS -w [stdout -> stdin] TPIPS stdout
 *
 * The full picture with jpips:
 *
 *                                         -> kill/signal ->
 *       -> OutputStream/stdin -> TPIPS -w -> out/stdin   ->
 * JPIPS                                                     TPIPS
 *       <- InputStream/stdout <----------------------------
 *
 *
 * What we might have some day:
 *
 * tty <-> EMACS <-> JPIPS > TPIPS -W s> TPIPS <-> XTERM 
 *                         <---------------+
 *
 * @author Fabien Coelho (with some advises from Ronan Keryell)
 */

/* UNIX headers.
 */
#include <unistd.h>
#include <sys/types.h>
#include <signal.h>
#include <errno.h>

/* Standard C headers.
 */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define SIZE		1024				/* buffer size */

#define SIGNAL_TAG	"# jpips send signal to tpips "

/* available tags
 */
#define CHECKPOINT	"CHECKPOINT"
#define INTERRUPT	"INTERRUPT"
#define KILLNOW		"EXIT"
#define ABORTIT		"ABORT"

#define WRAPPER		"[tpips_wrapper] "

#define starts_with(s1, s2) (strncmp(s1, s2, strlen(s2))==0) /* a la java */

/* global because needed in tpw_fatal_error.
 */
static pid_t tpips_pid = 0;

/* Enable log with -DLOG for debug.
 */
static void tpw_log(char * 
#if !defined(LOG)
		    __attribute__ ((unused)) 
#endif
		    message)
{
#if defined(LOG)
    fprintf(stderr, WRAPPER "%s\n", message);
    fflush(stderr);
#endif
}

static void tpw_fatal_error(char * message)
{
    fprintf(stderr, WRAPPER "fatal error: %s\n", message);
    fflush(stderr);
    kill(tpips_pid, SIGKILL);
    exit(1);
}

/* checks returned value for most system commands (-1 and errno set).
 */
static void tpw_check_perror(int en, char * message)
{
    if (en==-1)
    {
	perror(message);
	tpw_fatal_error(message);
    }
}

#define tpw_check(what, comment) \
	tpw_log(comment), tpw_check_perror(what, comment)
#define KILL(pid,sig) if (pid!=0) { tpw_check(kill(pid,sig),"kill()"); }

typedef void (*sig_handler_t)(int);

/* signal handler for tpips_wrapper.
 * basically forwards signals to tpips.
 * tries to stop tpips before exiting.
 * stops if tpips is stopped.
 */
static void tpw_sig_handler(int sn /* signal number */)
{
    /* these signals are traced.
     */
    fprintf(stderr, WRAPPER "signal %d caught!\n", sn);
    fflush(stderr);

    switch (sn)
    {
    case SIGHUP:
    case SIGINT:
    case SIGTERM:
    case SIGUSR1:
    case SIGUSR2:
	/* forward signals to tpips.
	 */
	KILL(tpips_pid, sn); 
	break;
    case SIGQUIT:
    case SIGABRT:
    case SIGCHLD: /* tpips stopped. */
    case SIGPIPE: /* idem? */
	fprintf(stderr, WRAPPER "killing tpips...\n");
	fflush(stderr);
	KILL(tpips_pid, SIGKILL);
	exit(2);
	break;
    default:
	fprintf(stderr, WRAPPER "unexpected signal (%d)\n", sn);
	tpw_fatal_error("unexpected signal");
    }

    /* reset signal handler.
     */
    (void) signal(sn, tpw_sig_handler);
}

/* special handlers for tpips wrapper.
 */
static void tpw_set_sig_handlers(void)
{
    (void) signal(SIGHUP,  tpw_sig_handler);
    (void) signal(SIGINT,  tpw_sig_handler);
    (void) signal(SIGTERM, tpw_sig_handler);

    (void) signal(SIGUSR1, tpw_sig_handler);
    (void) signal(SIGUSR2, tpw_sig_handler);

    (void) signal(SIGQUIT, tpw_sig_handler);
    (void) signal(SIGABRT, tpw_sig_handler);
    (void) signal(SIGCHLD, tpw_sig_handler);
    (void) signal(SIGPIPE, tpw_sig_handler);
}

static FILE * in_from_wrapper = NULL;
static FILE * out_to_tpips = NULL;

/* fork, with stdout-stdin link kept
 * @return the created pid.
 * should be in_from_jpips instead of stdin?
 * what about out_to_jpips and the wrapper???
 * fdopen() might be of some help.
 */
static pid_t tpw_fork_inout(void)
{
   int filedes[2];
   pid_t process;
   
   /* create pipe. 
    */
   tpw_check(pipe(filedes), "pipe()");

   /* fork 
    */
   process = fork();
   tpw_check(process, "fork()");

   if (process==0)
   {
       /* CHILD
	*/
       in_from_wrapper = fdopen(filedes[0], "r");
       if (!in_from_wrapper) tpw_check(-1, "fdopen()");
       
       tpw_check(dup2(filedes[0], 0), "dup2()"); /* stdin = in_from_wrapper; */
   }
   else
   {
       /* PARENT 
	*/
       out_to_tpips = fdopen(filedes[1], "w");
       if (!out_to_tpips) tpw_check(-1, "fdopen()");
   }

   if (process) 
   {
       /* the output might be interleaved with tpips output...
	* maybe I should lock stderr.
	*/
       fprintf(stderr, WRAPPER "started, tpips pid is %d\n", (int) process);
       fflush(stderr);
   }

   tpw_log("tpw_fork_inout() done");

   return process;
}

/* @return a line from "in", or NULL at EOF.
 * @caution a pointer to a static buffer is returned!
 */
static char * tpw_read_line(FILE * in)
{
    static char buffer[SIZE]; 
    int c=0, i=0;

    while (c != '\n' && i<SIZE-1 && (c = getc(in))!=EOF)
	buffer[i++] = (char) c;

    if (i==0) return NULL;

    buffer[i++] = '\0';
    return buffer;
}

/* send line to tpips.
 */
static void tpw_send_string_to_tpips(char * line)
{
    if (!out_to_tpips) 
	tpw_fatal_error("no out_to_tpips");

    fputs(line, out_to_tpips);
    fflush(out_to_tpips);
    
    /* could send a signal to tpips to warn about the incoming command.
     * it would be read from in_from_wrapper only.
     */
}

/* fork a tpips to goes on, while the current process acts as a wrapper,
 * which forwards orders, and perform some special signal handling.
 */
void tpips_wrapper(void)
{
    char * line;

    tpips_pid = tpw_fork_inout();

    /* the child is a new tpips, it keeps on executing...
     * the parent just acts as a wrapper, it never returns from here.
     */
    if (tpips_pid==0) return; /* CHILD is tpips and returns. */

    /* code for the WRAPPER starts here. the PARENT.
     */

    /* special handlers.
     */
    tpw_set_sig_handlers();

    while ((line = tpw_read_line(stdin)))
    {
	/* forward to tpips.
	 * how to ensure that tpips is alive?
	 * SIGCHLD and SIGPIPE are caught.
	 */
	tpw_send_string_to_tpips(line);

	/* handle signals.
	 */
	if (starts_with(line, SIGNAL_TAG))
	{
	    line += strlen(SIGNAL_TAG);

	    if (starts_with(line, CHECKPOINT)) {
		KILL(tpips_pid, SIGUSR1);
	    } else if (starts_with(line, INTERRUPT)) {
		KILL(tpips_pid, SIGINT);
	    } else if (starts_with(line, ABORTIT)) {
		KILL(tpips_pid, SIGABRT);
	    } else if (starts_with(line, KILLNOW)) {
		KILL(tpips_pid, SIGUSR2);
	    } else {
		tpw_fatal_error(line);
	    }
	}
    }

    /* stop tpips on EOF.
     */
    tpw_log("exiting...");
    fputs("exit\n", stdout);
    fclose(stdout);
    KILL(tpips_pid, SIGKILL);

    exit(0); /* wrapper mission done. */
}
