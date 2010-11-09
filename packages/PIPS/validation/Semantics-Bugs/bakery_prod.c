/* Premier encodage de l'algorithme d'exclusion mutuelle dit "bakery"
 *
 * Version tire de Pugh 1995
 */

#include <stdlib.h>

// #include <assert.h>
// ne fonctionne pas :
// #define assert(c) ((c)?(void) 0:error())
// marche avec :
#define assert(c) if (!(c)) error()

void error() {
	exit(1);
}

#define NO_LOOP

int alea(void) {
	return rand() % 2;
}

#define G_WC1 (a < b || b == 0)
#define G_WC2 (b < a || a == 0)
#define U_TW1 (a = b + 1)
#define U_TW2 (b = a + 1)
#define U_CT1 (a = 0)
#define U_CT2 (b = 0)

void run() {
	int a = 0, b = 0;
	float z;

TT:
	assert(a == 0 && b == 0);
	U_TW1; // la symetrie permet de ne traiter que la moitie des cas
	goto WT;

WT:
	assert(a > 0 && b == 0);
	if (alea()) {
		assert(G_WC1);
		goto CT;
	}
	else {
		U_TW2;
		z = 0.;
		goto WW1;
	}

CT:
	assert(a > 0 && b == 0);
	if (alea()) {
#ifdef NO_LOOP
		return;
#else
		U_CT1;
		goto TT;
#endif
	}
	else {
		U_TW2;
		goto CW;
	}

WW1:
	z = 0.;
	assert(a > 0 && a < b);
	assert(!G_WC2);
	assert(G_WC1);
	goto CW;

CW:
	return;
	/*assert(a > 0 && a < b);*/
	/*assert(!G_WC2);*/
	/*U_CT1;*/
	/*goto TW;*/

/*TW:*/
	/*assert(a == 0 && b > 0);*/
	/*if (alea()) {*/
		/*U_TW1;*/
		/*goto WW2;*/
	/*}*/
	/*else {*/
		/*assert(G_WC2);*/
		/*goto TC;*/
	/*}*/

/*WW2:*/
	/*assert(a > b && b > 0);*/
	/*assert(!G_WC1);*/
	/*assert(G_WC2);*/
	/*goto WC;*/

/*TC:*/
	/*assert(a == 0 && b > 0);*/
	/*if (alea()) {*/
		/*U_TW1;*/
		/*goto WC;*/
	/*}*/
	/*else {*/
/*#ifdef NO_LOOP*/
		/*return;*/
/*#else*/
		/*U_CT2;*/
		/*goto TT;*/
/*#endif*/
	/*}*/

/*WC:*/
	/*assert(a > b && b > 0);*/
	/*assert(!G_WC1);*/
/*#ifdef NO_LOOP*/
	/*return;*/
/*#else*/
	/*U_CT2;*/
	/*goto WT;*/
/*#endif*/
}

int main(void) {
  run();
  return 0;
}

