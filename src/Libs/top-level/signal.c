/*
 * $Id$
 *
 * signal management for pips.
 * moved from misc to top level so as to interact with pipsmake/pipsdbm...
 *
 * $Log: signal.c,v $
 * Revision 1.1  1998/05/25 17:09:59  coelho
 * Initial revision
 *
 */

#include <stdio.h>
#include <signal.h>

extern void checkpoint_workspace(void); /* in pipsmake */
extern void interrupt_pipsmake_asap(void); /* in pipsdbm */

static void pips_signal_handler(int num)
{
    fprintf(stderr, "signal %d occurred!\n", num);
    fflush(stderr);

    switch (num) 
    {
    case SIGINT:
    case SIGHUP:
    case SIGTERM:
    case SIGUSR1:
	interrupt_pipsmake_asap();
	break;
    case SIGUSR2:
	checkpoint_workspace();
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
    (void) signal(SIGINT,  pips_signal_handler);
    (void) signal(SIGHUP,  pips_signal_handler);
    (void) signal(SIGTERM, pips_signal_handler);

    (void) signal(SIGUSR1, pips_signal_handler);
    (void) signal(SIGUSR2, pips_signal_handler);
}
