// fonctionne par exemple avec pips r18437

#include <stdlib.h>

#define USE_ASSERT

#ifdef USE_ASSERT
#define assert(c) if (!(c)) error()
#else
#define assert(c) {}
#endif

#define check(c) if (!(c)) error()

/*#define NO_LOOP1*/
/*#define NO_LOOP2*/
/*#define NO_LOOP3*/

void error() {
	exit(1);
}

int alea() {
	return rand() % 2;
}

#define G_WC1 (a < b || b == 0)
#define G_WC2 (b < a || a == 0)
#define U_TW1 (a = b + 1)
#define U_TW2 (b = a + 1)
#define U_WC1 check(G_WC1)
#define U_WC2 check(G_WC2)
#define U_CT1 (a = 0)
#define U_CT2 (b = 0)

void run() {
	int a = 0, b = 0;
	
	U_TW1;
	
	while (1) {

#ifndef NO_LOOP1
		while (alea()) {
			if (1) {
				U_WC1;
				U_CT1;
				U_TW1;
			}
		}
#endif

#ifndef NO_LOOP2
		while (alea()) {
			if (1) {
				if (alea()) {
					U_WC1;
					U_TW2;
				}
				else {
					U_TW2;
					check(!G_WC2);
					U_WC1;
				}
				check(!G_WC2);
				U_CT1;
				U_WC2;
				U_CT2;
				U_TW1;
			}
		}
#endif
		
#ifndef NO_LOOP3
		while (alea()) {
			if (1) {
				if (alea()) {
					U_WC1;
					U_TW2;
				}
				else {
					U_TW2;
					check(!G_WC2);
					U_WC1;
				}
				check(!G_WC2);
				U_CT1;
				if (alea()) {
					U_WC2;
					U_TW1;
				}
				else {
					U_TW1;
					check(!G_WC1);
					U_WC2;
				}
				U_CT2;
			}
		}
#endif

	}
}

int main(void) {
  run();
  return 0;
}

