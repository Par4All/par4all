/* package arithmetique */

/*LINTLIBRARY*/

#include <stdio.h>

#include "assert.h"
#include "arithmetique.h"

/* int divide(int a, int b): calcul du divide de a par b;
 * le reste (qui n'est pas retourne) est toujours positif; il
 * est fourni par la fonction modulo()
 *
 * Il y a quatre configuration de signe a traiter:
 *  1. a>0 && b>0: a / b
 *  2. a<0 && b>0: (a-b+1) / b
 *  3. a>0 && b<0: cf. 1. apres changement de signe de b, puis changement 
 *     de signe du resultat
 *  4. a<0 && b<0: cf. 2. apres changement de signe de b, puis changement
 *     de signe du resultat
 *  5. a==0: 0
 */
int divide_fast(a,b)
int a,b;
{
    /* definition d'une look-up table pour les valeurs de a appartenant
       a [-DIVIDE_MAX_A..DIVIDE_MAX_A] et pour les valeurs de b
       appartenant a [1..DIVIDE_MAX_B] (en fait [-DIVIDE_MAX_B..DIVIDE_MAX_B]
       a cause du changement de signe)
       
       Serait-il utile d'ajouter une test b==1 pour supprimer une colonne?

       Serait-il utile de tester b > a pour renvoyer 0 ou -1 tout de suite?
       */
#define DIVIDE_MAX_A 7
#define DIVIDE_MAX_B 8
    static int divide_look_up[2*DIVIDE_MAX_A+1][DIVIDE_MAX_B]= {
	/* b ==         1   2   3   4   5   6   7   8 */
	{/* a == - 7 */ -7, -4, -3, -2, -2, -2, -1, -1},
	{/* a == - 6 */ -6, -3, -2, -2, -2, -1, -1, -1},
	{/* a == - 5 */ -5, -3, -2, -2, -1, -1, -1, -1},
        {/* a == - 4 */ -4, -2, -2, -2, -1, -1, -1, -1},
        {/* a == - 3 */ -3, -2, -1, -1, -1, -1, -1, -1},
        {/* a == - 2 */ -2, -1, -1, -1, -1, -1, -1, -1},
        {/* a == - 1 */ -1, -1, -1, -1, -1, -1, -1, -1},
        {/* a ==   0 */  0,  0,  0,  0,  0,  0,  0,  0},
        {/* a ==   1 */  1,  0,  0,  0,  0,  0,  0,  0},
        {/* a ==   2 */  2,  1,  0,  0,  0,  0,  0,  0},
        {/* a ==   3 */  3,  1,  1,  0,  0,  0,  0,  0},
        {/* a ==   4 */  4,  2,  1,  1,  0,  0,  0,  0},
        {/* a ==   5 */  5,  2,  1,  1,  1,  0,  0,  0},
	{/* a ==   6 */  6,  3,  2,  1,  1,  1,  0,  0},
	{/* a ==   7 */  7,  3,  2,  1,  1,  1,  1,  0}
    };
    /* translation de a pour acces a la look-up table par indice positif:
       la == a + DIVIDE_MAX_A >= 0 */
    int la;
    /* valeur du quotient C */
    int quotient;

    assert(b!=0);

    /* serait-il utile d'optimiser la division de a=0 par b? Ou bien
       cette routine n'est-elle jamais appelee avec a=0 par le package vecteur?
       */

    if(b>0) {
	if((la=a+DIVIDE_MAX_A) >= 0 && a <= DIVIDE_MAX_A && 
	   b <= DIVIDE_MAX_B) {
	    /* acceleration par une look-up table */
	    quotient = divide_look_up[la][b-1];
	}
	else {
	    /* calcul effectif du quotient: attention, portabilite douteuse */
	    if(a > 0) 
		quotient = a / b;
	    else
		quotient = (a-b+1) / b;
	}
    }
    else {
	/* b est negatif, on prend l'oppose et on corrige le resultat */
	b = -b;
	if((la=a+DIVIDE_MAX_A) >= 0 && a <= DIVIDE_MAX_A && 
	   b <= DIVIDE_MAX_B) {
	    /* acceleration par une look-up table */
	    quotient = -divide_look_up[la][b-1];
	}
	else {
	    /* calcul effectif du divide: attention, portabilite douteuse */
	    if(a > 0) 
		quotient = - (a / b);
	    else
		quotient = - ((a-b+1) / b);
	}
    }
    return(quotient);
}

int divide_slow(a,b)
int a;
int b;
{ return(DIVIDE(a,b));}
