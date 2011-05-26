/* The extern declaration is invalidated by the direct declaration or
 *  vice-versa
 *
 * In the internal representation the initial value is not part of the
 * declaration statement but of the symbol table.
 *
 * In the same way, the extern declaration is not part of the
 * declaration statement but a field of code.
 *
 * So the very same information is available to reconstruct both
 * declaration statements for a in the compilation unit.
 *
 * FI: I do not see a way out, unless a specific declaration statement
 * is introduced (currently, we use calls to continue as declaration
 * statements).
 */

/* a cannot be initialized twice in the prettyprinted code... so the
   initialization is lost... or the prettyprinted code is illegal */
int a = 1;

/* a is not static, but it is not fully external as it has been
   declared and allocated and potentially initialized in this
   module. */
extern int a;
