#include <stdio.h>
#include <stdlib.h>
#include <string.h>



int arg_p(int argc, char **argv, char *opt, int *val)
{
  int i;
  int result = 0; 

  if (argc == 1)
    goto  end;

  for (i = 1; i < argc - 1; i++)
    if (!strcmp(argv[i], opt))
      {
	*val = atoi(argv[i + 1]);
	result =  1;
	break;
      }

 end:
  return result;
}

int arg_s(int argc, char **argv, char *opt, char **string)
{
  int result = 0;
  int i;

  if (argc == 1)
    goto  end;
  
  for (i = 1; i < argc - 1; i++)
    if (!strcmp(argv[i], opt))
      {
	*string = argv[i + 1];
	result = 1;
	break;
      }

 end:
  return result;
}

#ifdef TEST
/* ./testutil -p 4 -s lucien */

int main(int argc, char **argv)
{
  int monentier;
  char *machaine="\n";

  if(arg_p(argc, argv, "-p", &monentier))
    printf("option -p trouvee\n");
  
  if (arg_s(argc, argv, "-s", &machaine))
    printf("option -s trouvee\n");
  
  printf("monentier = %d, machaine = %s\n", monentier, machaine);

  if(!arg_p(argc, argv, "-p2", &monentier))
      printf("option -p2 non trouvee et c'est normal\n");

  if(!arg_s(argc, argv, "-s2", &machaine))
    printf("option -s2 pas trouvee et c'est normal\n");

  return EXIT_SUCCESS;
}


#endif
