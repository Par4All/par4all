// Check fix point for do while

// Check the resulting points-to information after the loop: a cannot
// point to nowhere

// Check sorting before print-out...

// #include <stdio.h>

int main() {
  int *a, *b, c, d, i;
   c = 0;
   d = 1;
   i = 2;
   b = &c;
 
   do {
     a = b;
     b = &d;
     i = i+1;
   } while(i<5); 

  return 0;
}
