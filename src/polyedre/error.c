/*******************************************************************/
/**** This file contains functions for the generation of errors ****/
/*******************************************************************/

/*** Include files ***/
#include <stdio.h>
#include <stddef.h>
#include "errortxt.h"
#include "erroridt.h"

/*** Statics (and globals) variables ***/
#define MOD_LEN 20
static char module[MOD_LEN];
static void (*handler)()=NULL;

/*** Functions ***/

/** Updating of "module" static variable **/
void errModuleSet(name)
char *name;
{
 if(strlen(name)>=MOD_LEN) name[MOD_LEN-1]=0;
 strcpy(module,name);
}

/** This function allow to specify a handler routine for the terminaison
    of TERROR type errors **/
void errHandlerSet(funct)
void (*funct)();
{
handler=funct;
}

/** Generation of an error **/
void errError(id,type,funct,param)
int id;			/* Identifier of error */
int type;		/* Type of the error */
char *funct;		/* Name of the function which is generating the error */
char *param;		/* Some informations about the error */
{
 char buffer[100];

 sprintf(buffer,errorText[id],param);
 fprintf(stderr,"%s.%s : %s : %s.\n\r",module,funct,errorType[type],buffer);
 if(type==TERROR){
   if(handler!=NULL) handler(id);
   exit(-1);
}  }
