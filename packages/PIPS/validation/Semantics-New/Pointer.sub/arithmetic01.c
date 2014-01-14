#include <stdio.h>
#include <stdlib.h>

int main()
{
  long int a[10], *p, i, *q;
  for (i=0; i<10; i++)
    a[i] = i*i;
    
  p=&a[0];
  printf("%i\n", p);
  printf("%i\n", *p);
  p++;
  printf("%i\n", p);
  printf("%i\n", *p);
  p+=1;
  printf("%i\n", p);
  printf("%i\n", *p);
  q=p+1;
  printf("p=%i\n", p);
  printf("*p=%i\n", *p);
  printf("q=%i\n", q);
  printf("*q=%i\n", *q);
  i=q-p;
  printf("q-p=%i\n", i);
//   
//   
//   p = malloc(10*sizeof(*p));
//   q=p;
//   printf("%i\n", p);
//   printf("%i\n", *p);
//   p++;
//   printf("%i\n", p);
//   printf("%i\n", *p);
//   p+=1;
//   printf("%i\n", p);
//   printf("%i\n", *p);
//   p=p+1;
//   printf("%i\n", p);
//   printf("%i\n", *p);
//   
//   free(q);
  
  return 0;
}
