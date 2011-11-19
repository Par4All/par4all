int main() {
  int t1 = 1;
     register int *p1 asm ("r0") = &t1;
     register int *p2 asm ("r1") = &t1;
     register int *result asm ("r0");
     asm ("sysint" : "=r" (result) : "0" (p1), "r" (p2));
     return 0;
}
