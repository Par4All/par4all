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
 * signal management for pips.
 * moved from misc to top level so as to interact with pipsmake/pipsdbm...
 */

#include <stdlib.h>
#include <stdio.h>
#include <signal.h>
#include <unistd.h>

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "misc.h"
#include "pipsdbm.h"
#include "pipsmake.h"
#include "top-level.h"

static void pips_signal_handler(int num)
{
    user_log("interruption signal %d caught!\n", num);

    switch (num) 
    {
    case SIGINT:
    case SIGHUP:
    case SIGTERM:
	user_log("interrupting pipsmake as soon as possible...\n");
	interrupt_pipsmake_asap();
	break;
    case SIGUSR1:
	user_log("interruption for checkpointing...\n");
	/* cold blooded. 
	 * might be quite dangerous for the life of the process.
	 * should not enter twice in there... 
	 * might be convinient anyway.
	 */
	checkpoint_workspace();
	break;
    case SIGUSR2:
	user_log("interruption for exiting...\n");
	exit(2);
	break;
    default:
	fprintf(stderr, "[pips_signal_handler] unexpected signal %d\n", num);
	abort();
    }

    /* must reset handler once raised.
     */
    (void) signal(num, pips_signal_handler);
}

void initialize_signal_catcher(void)
{
    // misc signals
    (void) signal(SIGINT,  pips_signal_handler);
    (void) signal(SIGHUP,  pips_signal_handler);
    (void) signal(SIGTERM, pips_signal_handler);

    (void) signal(SIGUSR1, pips_signal_handler);
    (void) signal(SIGUSR2, pips_signal_handler);

    // timeout handling
    set_pips_timeout_from_env();
}

/* keep track of current timeout
 */
static unsigned int pips_timeout_delay;

/* exit on timeout
 */
static void pips_timeout_handler(int __attribute__ ((__unused__)) sig)
{
  fprintf(stderr,
          "\nERROR: pips timeout of %d seconds reached\n", pips_timeout_delay);
  // use 203 exit status for timeout, should be easy to detect
  exit(203);
}

/* set pips timeout on delay
 */
void set_pips_timeout(unsigned int delay)
{
  // pips_user_warning("setting pips timeout to %d\n", delay);
  if (delay>0) {
    pips_timeout_delay = delay;
    struct sigaction act;
    act.sa_handler = pips_timeout_handler;
    sigemptyset(&act.sa_mask);
    act.sa_flags = 0;
    sigaction(SIGALRM, &act, NULL);
    alarm(delay);
 }
}

/* set pips timeout using PIPS_TIMEOUT environment variable
 */
void set_pips_timeout_from_env(void)
{
  string sdelay = getenv("PIPS_TIMEOUT");
  if (sdelay) {
    unsigned int delay = atoi(sdelay);
    set_pips_timeout(delay);
  }
}

/* delete current pips timeout settings
 */
void reset_pips_timeout(void)
{
  // pips_user_warning("resetting pips timeout\n");
  // cleanup alarm
  pips_timeout_delay = 0;
  struct sigaction act;
  act.sa_handler = SIG_DFL;
  sigemptyset(&act.sa_mask);
  act.sa_flags = 0;
  sigaction(SIGALRM, &act, NULL);
  alarm(0); 
}
