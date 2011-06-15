#include <stdlib.h>
#include <stdio.h>

#define DO_CHECKING 1
#define GOOD (j1 <= n && j2 <= n)

int flip(void) {
	return rand() % 2;
}
#define OR(t1, t2) {if (flip()) {t1} else {t2}}
#define LOOP(t) {while (flip()) {t}}

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

#define G1 (i > 0 && i < m)
#define A1 {i--; j1++;}
#define C1 COMMAND(G1, A1)

#define G2 (i > 0 && i >= m)
#define A2 {i -= m; j2++;}
#define C2 COMMAND(G2, A2)

#define INI {m = rand(); n = rand(); ASSUME(0 < m && m < n); i = n; j1 = j2 = 0;}

// transition system

void ts_singlestate(void) {
	int i, m, n, j1, j2;
	INI;
	LOOP(OR(C1, C2));
}

int main(void) {
	ts_singlestate();
	return 0;
}

