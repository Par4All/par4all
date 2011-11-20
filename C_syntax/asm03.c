/* gcc extension not supported by -std=c99, gcc -c does not work either on ibarron:
 *
 *francois@ibarron:~/MYPIPS/validation/C_syntax$ gcc -c asm03.c
 * asm03.c: In function ‘main’:
 * asm03.c:5:20: error: invalid register name for ‘p1’
 * asm03.c:6:20: error: invalid register name for ‘p2’
 * asm03.c:7:20: error: invalid register name for ‘result’
 */

int main() {
  int t1 = 1;
     register int *p1 asm ("r0") = &t1;
     register int *p2 asm ("r1") = &t1;
     register int *result asm ("r0");
     asm ("sysint" : "=r" (result) : "0" (p1), "r" (p2));
     return 0;
}
