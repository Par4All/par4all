/********************************************************/
/* This file contains initializations for Cipol-Light. */
/*******************************************************/

/*** Include files ***/

#include <stdio.h>
#include <stddef.h>
#include <setjmp.h>

#ifdef DBMALLOC4
#       include "malloc4.h"
#endif
#ifdef DBMALLOC9
#       include "malloc9.h"
#endif

/*** Global values ***/

jmp_buf retFromError;		/* Allows to resume from a fatal error */
unsigned char errorFlag;	/* We must know when a fatal error was */
				/* raised                              */
unsigned int errorLine;		/* To memorize syntax error line */

/*** Initialization function ***/

void recover();

void Cipol_Init()
{
errorFlag=NULL;		/* No fatal error yet */
errorLine=1;
			/* Errors may occurs in Cipol-Light */
errModuleSet("CIPOL-LIGHT");
errHandlerSet(recover);	/* Set te handler for fatal errors */
}

/*** Set the error flag ***/

void raiseError()
{
errorFlag=1;
printf("error\n");
}

/*** Reset the error flag ***/

void resetError()
{
errorFlag=NULL;
}

/*** Handler for fatal errors ***/

void recover()
{
raiseError();			/* An error occurs */
longjmp(retFromError,1);	/* Just jump to the recover point */
}
