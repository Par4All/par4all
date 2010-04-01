#include<stdlib.h>
#include<stdio.h>
int main(){
  typedef struct{
  int *p;
  }my_str;
  
  my_str* s, *s1;
  int i =0;
  s =(my_str*) malloc(sizeof(my_str));
  s1 =(my_str) malloc(sizeof(my_str));
  return 0;

}
