/* 
 * $Id$
 *
 * This file contains the main for wpips.
 * Please, do not change anything! do any change to wpips_main().
 *
 * FC.
 */

extern char * pips_thanks(char *, char *);
extern int wpips_main(int, char**);

int main(int argc, char ** argv)
{
    pips_thanks("wpips", argv[0]);
    return wpips_main(argc, argv);
}
