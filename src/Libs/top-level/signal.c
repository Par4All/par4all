/*
 * $Id$
 *
 * signal management for pips.
 * moved from misc to top level so as to interact with pipsmake/pipsdbm...
 *
 * $Log: signal.c,v $
 * Revision 1.3  1998/05/26 16:51:56  coelho
 * missing break added.
 *
 * Revision 1.2  1998/05/25 17:24:34  coelho
 * interruption with different behaviors...
 *
 * Revision 1.1  1998/05/25 17:09:59  coelho
 * Initial revision
 *
 */

#include <stdio.h>
#include <signal.h>

extern void checkpoint_workspace(void); /* in pipsmake */
extern void interrupt_pipsmake_asap(void); /* in pipsdbm */
extern void user_log(char *, ...); /* in misc */

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
    (void) signal(SIGINT,  pips_signal_handler);
    (void) signal(SIGHUP,  pips_signal_handler);
    (void) signal(SIGTERM, pips_signal_handler);

    (void) signal(SIGUSR1, pips_signal_handler);
    (void) signal(SIGUSR2, pips_signal_handler);
}
