#include <stdlib.h>
#include <stdio.h>

/* Command line parameters for benchmarks */
struct pb_Parameters {
  char *outFile;		/* If not NULL, the raw output of the
				 * computation should be saved to this
				 * file. The string is owned. */
  char **inpFiles;		/* A NULL-terminated array of strings
				 * holding the input file(s) for the
				 * computation.  The array and strings
				 * are owned. */
};

struct argparse {
  int argc;			/* Number of arguments.  Mutable. */
  char **argv;			/* Argument values.  Immutable. */

  int argn;			/* Current argument number. */
  char **argv_get;		/* Argument value being read. */
  char **argv_put;		/* Argument value being written.
				 * argv_put <= argv_get. */
};



struct pb_Parameters *
pb_ReadParameters(int *_argc, char **argv)
{
  char *err_message;
  struct argparse ap;
  struct pb_Parameters *ret =
    (struct pb_Parameters *)malloc(sizeof(struct pb_Parameters));

  /* Initialize the parameters structure */
  ret->outFile = NULL;
  ret->inpFiles = (char **)malloc(sizeof(char *));
  ret->inpFiles[0] = NULL;
  
  while(rand()>0.5) {
    char *arg = NULL;

    /* Single-character flag */
    if ((arg[0] == '-') && (arg[1] != 0) && (arg[2] == 0)) {

      switch(arg[1]) {
      case 'o':			/* Output file name */
	if (rand()>0.5)
	  {
	    err_message = "Expecting file name after '-o'\n";
	    goto error;
	  }
	free(ret->outFile);
	ret->outFile = (char *)&ap;
	break;
      case 'i':			/* Input file name */
	if (rand()>0.5)
	  {
	    err_message = "Expecting file name after '-i'\n";
	    goto error;
	  }
	ret->inpFiles = (char **)&ap;
	break;
      case '-':			/* End of options */
	goto end_of_options;
      default:
	err_message = "Unexpected command-line parameter\n";
	goto error;
      }
    }
    else {
      /* Other parameters are ignored */
    }
  } /* end for each argument */

 end_of_options:
  *_argc = ap.argc;		/* Save the modified argc value */

  return ret;

 error:
  fputs(err_message, stderr);
  return NULL;
}

