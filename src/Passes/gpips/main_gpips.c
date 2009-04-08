/*
 * $Id: main_gpips.c 12279 2005-12-23 14:29:06Z coelho $
 *
 * This file contains the main for gpips.
 * Please, do not change anything! do any change to wpips_main().
 *
 * FC.
 */
/*
 * forked to main_gpips.c
 * Edited by Johan GALL
 *
 */

extern char * pips_thanks(char *, char *);
extern int gpips_main(int, char**);

int main(int argc, char ** argv)
{
    pips_thanks("gpips", argv[0]);
    return gpips_main(argc, argv);
}
