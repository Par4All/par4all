/* package plint: programmation lineaire en nombres entiers
 *
 * Corinne Ancourt
 *
 * utilise les packages:
 *  arithmetique.h
 *  vecteur.h
 *  contrainte.h
 *  sc.h
 *  ray_dte.h
 *  sommet.h
 *  sg.h
 *  polyedre.h (indirectement, pour les fonctions de conversion de sc en
 *              liste de sommets et reciproquement)
 *  matrice.h 
 */

/*
 * Representation d'une solution d'un systeme lineaire
 *
 * Pourrais-tu etre plus precise, Corinne? FI
 */

/* constante associee a une solution (Comment est-elle associee? FI)    */
#define SOLUTION 0

typedef struct Ssolution{
    /* variable du systeme */
    Variable var;
    /* valeur de la variable */
    Value val;
    /* denominateur de la valeur de la variable */
    Value denominateur;
    /* pointeur vers la variable suivante */
    struct Ssolution *succ;
} *Psolution,Ssolution;

