#ifndef _PROPAGATE
#define _PRPPAGATE
/* propagate.c */
extern void CheckStride(loop Loop);
extern list CompRegionsExactUnion(list l1, list l2, bool (*union_combinable_p)(effect, effect));
extern list CompRegionsMayUnion(list l1, list l2, bool (*union_combinable_p)(effect, effect));
extern bool comp_regions(char *module_name);
extern list comp_regions_of_statement(statement s);
extern list comp_regions_of_instruction(instruction i, transformer t_inst, transformer context, list *plpropreg);
extern list comp_regions_of_block(list linst);
extern list comp_regions_of_test(test t, transformer context, list *plpropreg);
extern list comp_regions_of_loop(loop l, transformer loop_trans, transformer context, list *plpropreg);
extern list comp_regions_of_call(call c, transformer context, list *plpropreg);
extern list comp_regions_of_unstructured(unstructured u, transformer t_unst);
extern list comp_regions_of_range(range r, transformer context);
extern list comp_regions_of_syntax(syntax s, transformer context);
extern list comp_regions_of_expressions(list exprs, transformer context);
extern list comp_regions_of_expression(expression e, transformer context);
extern list comp_regions_of_read(reference ref, transformer context);
extern list comp_regions_of_write(reference ref, transformer context);

#endif


