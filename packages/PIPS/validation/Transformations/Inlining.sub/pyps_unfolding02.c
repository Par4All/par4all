#include <stdio.h>

void callee1() {
  printf("In callee 1 !\n");
  
}


void callee2() {
  printf("In callee 2, I will call callee1 !\n");
  callee1();
}

int unfold() {
  printf("I have to be unfolded !\n");
  callee1();
  callee2();
}

int non_unfold() {
  printf("I have to be unfolded !\n");
  callee1();
  callee2();
}

