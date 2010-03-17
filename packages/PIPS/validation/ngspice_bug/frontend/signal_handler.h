/*************
 * Header file for signal_handler.c
 * 1999 E. Rouat
 * $Id: signal_handler.h,v 1.2 2005/05/30 17:21:11 sjborley Exp $
 ************/

#ifndef SIGNAL_HANDLER_H_INCLUDED
#define SIGNAL_HANDLER_H_INCLUDED

RETSIGTYPE ft_sigintr(void);
RETSIGTYPE sigfloat(int sig, int code);
RETSIGTYPE sigstop(void);
RETSIGTYPE sigcont(void);
RETSIGTYPE sigill(void);
RETSIGTYPE sigbus(void);
RETSIGTYPE sigsegv(void);
RETSIGTYPE sig_sys(void);

extern JMP_BUF jbuf;

#endif
