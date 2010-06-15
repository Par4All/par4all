/*
 * trace.c
 *
 *
 */

#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <errno.h>
#include <assert.h>
#include <string.h>

#include "timings.h"

#define MAX_NAME_LENGTH 50
#define MAX_FUNCTIONS 50

#define NORANK -1

struct _func {
  char func_name[MAX_NAME_LENGTH];
  int process_rank;
  int time_p;
  double timings;
  Tick tick_in;
} ;
  
typedef struct _func func;

static func Functions_tab[MAX_FUNCTIONS];

static int Nb_functions;
static int Set_traced_p = 0;
static int Timings_p = 0;

static int NbProcs;
static int MyRank;

char *Output_trace_file = NULL;

FILE *Output_trace_stream;

#ifdef DEBUG
static void print_functions_tab () {
  int i;
  printf("Nb_functions = %d\n", Nb_functions);
  for (i = 0; i < Nb_functions; i++)
    {
      printf("Function number = %i, Process rank = %d, Function name = %s, Function time_p = %d\n", i, Functions_tab[i].process_rank, Functions_tab[i].func_name, Functions_tab[i].time_p);
    }
}
#endif

int is_traced(char *function, int *index, int *time_p)
{
  int i;

#ifdef DEBUG  
  printf("is_traced function = %s, index = %p, time_p = %p\n", function, index, time_p);
#endif

  for( i = 0; i < Nb_functions; i++)
    {
      char *name;
      int rank;

      rank = Functions_tab[i].process_rank;
      name = Functions_tab[i].func_name;
      *time_p = Functions_tab[i].time_p;
      *index = i;

      if (strcmp(name, function) != 0)  continue;

      if (rank == NORANK || rank == MyRank) return 1;
      
      if (NbProcs < 2) return 1;
    }
#ifdef DEBUG
  printf("is_traced 0\n");
#endif

  *time_p = -1;
  *index = -1;
  return 0;
}

static void trace_header(FILE *fd, char *function, int when)
{

  if (NbProcs > 1)
    fprintf(fd, "%d/%d) ", MyRank, NbProcs);
  switch (when) {
  case 'i':
    fprintf(fd, "<%s: ", function);
    break;
  case 'o':
    fprintf(fd, "%s> ", function);
    break;
  default:
    fprintf(fd, "%s: ", function);
  }
}

void trace(char *function, int when, ...)
{
  va_list args;
  char *format;

  if (!Set_traced_p) goto end;
  /*   if (!is_traced(function)) goto end; */

  va_start (args, when);

  trace_header(Output_trace_stream, function, when);

  format = va_arg(args, char*);
  vfprintf(Output_trace_stream, format, args);
  fprintf(Output_trace_stream, "\n");

  fflush(Output_trace_stream);

  va_end (args);
  
 end:
  ;
}

static void open_file(FILE **fd, char *filename)
{
  if (NbProcs < 2)
    *fd = fopen(filename, "w");
  else 
    {
      char name[256];
      if (strlen(filename) > 250)
	fprintf(stderr, "%s too long (must be < 250)", filename);
      
      sprintf(name, "%s_%d", filename, MyRank);
      
      *fd = fopen(name, "w");
    }
  if (!(*fd))
    {
      perror(filename);
      exit(EXIT_FAILURE);
    }
}

/*
 * Si le nombre de processus est égal à 1, 
 * myrank peut prendre une valeur entière quelconque
 *
 */

void set_traces(char *infilename, char *outfilename, int nbprocs, int myrank) 
{
  FILE *fd;
  char code;

#ifdef DEBUG  
  printf("set_traces infilename = %s, outfilename = %s, nbprocs= %d, myrank = %d\n", infilename, outfilename, nbprocs, myrank);
#endif

  NbProcs = nbprocs;
  MyRank = myrank;
  Output_trace_file = outfilename;

  fd = fopen(infilename, "r");
  if (!fd)
    {
      perror(infilename);
      
      if (errno == ENOENT) /* fichier non trouvé */
	goto end;

      exit(EXIT_FAILURE);
    }

  if (Output_trace_file == NULL)
    Output_trace_stream = stdout;
  else 
    {
      open_file(&Output_trace_stream, Output_trace_file);
    }
  
  Nb_functions = 0;
  
  while (fscanf(fd, "%c", &code) != EOF) 
    {
      char name[MAX_NAME_LENGTH];
      int time_p = 0;
      int function_rank = NORANK;
      int size;
      int found_index, found_time_p;
	  
      switch(code) 
	{
	case '>':
	  if (!fscanf(fd, "%d", &function_rank))
	    {
	      perror("function_rank");
	    }
	  break;
	case '=':
	  time_p = 1;
	  if (!Timings_p) Timings_p = 1;
	  break;
	case '+':
	case '.':
	  break;
	default:
	  fprintf(stderr, "Unexpected code: %c\n", code);
	  break;
	} 

      if (code == '.') 
	break;

      if (!fscanf(fd, " %s\n", name))
	{
	  perror("name");
	  break;
	}


      size = strlen(name);
      if (MAX_NAME_LENGTH < size)
	{
	  fprintf(stderr, "MAX_NAME_LENGTH (%d) >= strlen(name) (%d)", MAX_NAME_LENGTH, size);
	  exit(EXIT_FAILURE);
	}
      
      /*
       * cas où deux fois la même fonction dans le fichier.
       * +func
       * =func
       *
       */
      if (!is_traced(name, &found_index, &found_time_p))
	{
	  strcpy(Functions_tab[Nb_functions].func_name, name);
	  Functions_tab[Nb_functions].timings = 0.0;
	  Functions_tab[Nb_functions].process_rank = function_rank;
	  Functions_tab[Nb_functions].time_p = time_p;
	  Nb_functions++;
	}


      if (Nb_functions >= MAX_FUNCTIONS)
	{
	  fprintf(stderr, "Nb_functions >= MAX_FUNCTIONS (%d)!", MAX_FUNCTIONS);
	  exit(EXIT_FAILURE);
	}
      
    }
  
  fclose(fd);

  Set_traced_p = 1;

  if (Timings_p) 
    timings_init();

#ifdef DEBUG
  print_functions_tab();  
  printf("set_traces 0\n");
#endif

 end:
  ;
}

void time_trace_in(char *nom, int index)
{
#ifdef DEBUG
  printf("time_trace_in, nom = %s, index = %d\n", nom, index);
  assert(index < MAX_FUNCTIONS);
  assert(strcmp(nom, Functions_tab[index].func_name) == 0);
#endif

  if (!Functions_tab[index].time_p) goto end;

  GET_TICK(Functions_tab[index].tick_in);

 end:
  ;
#ifdef DEBUG
  printf("time_trace_in\n");
#endif
}

void time_trace_out(char *nom, int index)
{
#ifdef DEBUG
  printf("time_trace_out, nom = %s, index = %d\n", nom, index);
  assert(index < MAX_FUNCTIONS);
  assert(strcmp(nom, Functions_tab[index].func_name) == 0);
#endif

  if (Functions_tab[index].time_p == 0) goto end;

  SET_TICK(_last_timings_event, Functions_tab[index].tick_in);
  Functions_tab[index].timings = timings_event();

 end:
  ;
#ifdef DEBUG
  printf("time_trace_out\n");
#endif
}


void timings_dump_traces()
{
  int i;
#ifdef DEBUG
  printf("timings_dump_traces\n");
#endif
  if (!Timings_p) goto end;

  for(i = 0; i < Nb_functions; i++)
    {
      if (!Functions_tab[i].time_p) continue;

      trace_header(Output_trace_stream, Functions_tab[i].func_name, ' ');
      fprintf(Output_trace_stream, "TIME %g\n", Functions_tab[i].timings);

      fflush(Output_trace_stream);
    }
 end:
  ;
#ifdef DEBUG
  printf("timings_dump_traces\n");
#endif
}

#ifdef TEST
#define TRACE 

#include "trace.h"

int compute(float arg1, float arg2)
{
  int i;
  float result = 0.0;
  IN_TRACE("arg1 = %f, arg2 = %f", arg1, arg2); 

  for(i = 0; i < 1000; i++)
    result += arg1 * arg2;

  OUT_TRACE("result = %f", result);
  return result;
}

int print(char *nom)
{
  int i, result = 1;
  IN_TRACE("nom = %s", nom); 
  
  for(i = 0; i < 100000; i++)
    result += result;

  TRACE_P("je debugge dans la fonction");

  compute( 1.5, 2.3);

  OUT_TRACE("result = %d", result);
  return result;
}



int main(int argc, char *argv[]) {
  printf("debut\n");

  SET_TRACES("traces", NULL, 1, 0);

  print("fred");



  printf("fin\n");

  timings_dump_traces();
  return EXIT_SUCCESS;
}


#endif
