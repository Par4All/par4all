#include <stdlib.h>
#include <stdio.h>

#define DO_CHECKING 1
#define GOOD (i <= n0)

int flip(void) {
	return rand() % 2;
}
#define OR(t1, t2) {if (flip()) {t1} else {t2}}
#define LOOP(j) {while (flip()) {j}}

void deadlock() {
	printf("deadlock\n");
	while (1);
}
#define ASSUME(c) {if (!(c)) deadlock();}

#if DO_CHECKING == 0
#define CHECK(c)
#define CHECK_NOT(c)
#else
void checking_error(void) {
	fprintf(stderr, "checking error");
	exit(2);
}
#define CHECK(c) {if (!(c)) checking_error();}
#define CHECK_NOT(c) {if (c) checking_error();}
#endif

#define COMMAND_NOCHECK(g, a) {ASSUME(g); a;}
#ifdef GOOD
#define COMMAND(g, a) {COMMAND_NOCHECK(g, a); CHECK(GOOD);}
#else
#ifdef BAD
#define COMMAND(g, a) {COMMAND_NOCHECK(g, a); CHECK_NOT(BAD);}
#endif
#endif

#define G1 (n > 0 && m > 0 && s == 1)
#define A1 {n--; m--; i++; s = 2;}
#define C1 COMMAND(G1, A1)

#define G2 (n > 0 && s == 2)
#define A2 {n--; m++; i++;}
#define C2 COMMAND(G2, A2)

#define G3 (s == 2)
#define A3 {s = 1;}
#define C3 COMMAND(G3, A3)

#define INI {n0 = rand(); m = rand(); n = n0; ASSUME(n > 0 && m > 0); s = 1; i = 0;}

// transition system

void ts_singlestate(void) {
	int n0, n, m, s, i;
	INI;
	LOOP(OR(C1, OR(C2, C3)));
}

int main(void) {
	ts_singlestate();
	return 0;
}

