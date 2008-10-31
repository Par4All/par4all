
/* simulation of the type region */

#define region effect
#define REGION EFFECT  /* for casts */
#define REGION_ EFFECT_  /* for some left casts */
#define REGION_TYPE EFFECT_TYPE
#define REGION_NEWGEN_DOMAIN EFFECT_NEWGEN_DOMAIN
#define gen_REGION_cons gen_effect_cons
#define gen_region_cons gen_effect_cons

#define region_undefined effect_undefined
#define region_undefined_p(reg)  effect_undefined_p((reg))
#define make_region(reference,action,approximation,system) \
    make_effect(make_cell(is_cell_preference, make_preference(reference)),\
    (action),(approximation), \
    make_descriptor(is_descriptor_convex,system))
#define region_reference(reg) \
    preference_reference(cell_preference(effect_cell(reg)))
#define region_any_reference(reg) \
    cell_preference_p(effect_cell(reg))? preference_reference(cell_preference(effect_cell(reg))):cell_reference(effect_cell(reg))
#define region_action(reg) effect_action(reg)
#define region_approximation(reg) effect_approximation(reg)
#define region_context(reg) effect_context(reg)
#define region_cell(reg) effect_cell(reg)
#define copy_region(reg) region_dup((reg))
/* FI: much too dangerous! */
/* #define free_region(reg) region_free((reg)) */
#define free_region(reg) free_effect((reg))

#define region_entity(reg) reference_variable(region_any_reference(reg))
#define region_action_tag(reg) action_tag(effect_action(reg))
#define region_approximation_tag(reg) \
    approximation_tag(effect_approximation(reg))
#define region_system_(reg) \
    descriptor_convex_(effect_descriptor(reg))
#define region_system(reg) \
    descriptor_convex(effect_descriptor(reg))

/* useful region macros */

#define region_read_p(reg) (action_tag(effect_action(reg))==is_action_read)
#define region_write_p(reg) (action_tag(effect_action(reg))==is_action_write)
#define region_may_p(reg) \
        (approximation_tag(effect_approximation(reg)) == is_approximation_may)
#define region_must_p(reg) \
        (approximation_tag(effect_approximation(reg)) == is_approximation_must)
#define region_exact_p(reg) \
        (approximation_tag(effect_approximation(reg)) == is_approximation_must)
#define region_empty_p(reg) sc_empty_p(region_system(reg))
#define region_rn_p(reg) sc_rn_p(region_system(reg))
#define region_scalar_p(reg) entity_scalar_p(region_entity(reg))
#define effect_region_p(e) (descriptor_convex_p(effect_descriptor(e)))

/* consistency checking */
#define debug_regions_consistency(l_reg)\
ifdebug(1){regions_consistent_p(l_reg);}
#define debug_region_consistency(reg)\
ifdebug(1){region_consistent_p(reg);}

/* convenient debug messages */
#define debug_print_region(level, message, region) \
  ifdebug(level) { pips_debug(level, "%s\n", message); print_region(region);}

/* other things */

#define PHI_PREFIX "PHI"
#define PSI_PREFIX "PSI"
#define BETA_PREFIX "BETA"
#define PROPER TRUE
#define SUMMARY FALSE

#define REGIONS_MODULE_NAME "REGIONS-PACKAGE"


/* TRUE if e is a phi variable
 * PHI entities have a name like: REGIONS:PHI#, where # is a number.
 * takes care if TCST and undefined entities, just in case.
 * FC, 09/12/94
 */
#define variable_phi_p(e) \
  ((e)!=(entity)NULL && (e)!=entity_undefined && \
    strncmp(entity_name(e), REGIONS_MODULE_NAME, 10)==0 && \
    strstr(entity_name(e), PHI_PREFIX) != NULL)

#define variable_psi_p(e) \
  ((e)!=(entity)NULL && (e)!=entity_undefined && \
    strncmp(entity_name(e), REGIONS_MODULE_NAME, 10)==0 && \
    strstr(entity_name(e), PSI_PREFIX) != NULL)

#define variable_beta_p(e)\
  ((e)!=(entity)NULL && (e)!=entity_undefined && \
    strncmp(entity_name(e), REGIONS_MODULE_NAME, 10)==0 && \
    strstr(entity_name(e), BETA_PREFIX) != NULL)

#define R_RW 0
#define R_IN 1
#define R_OUT 2

#define SEQUENTIAL_REGION_SUFFIX ".reg"
#define SEQUENTIAL_PROPER_REGION_SUFFIX ".preg"
#define USER_REGION_SUFFIX ".ureg"
#define SEQUENTIAL_IN_REGION_SUFFIX ".inreg"
#define USER_IN_REGION_SUFFIX ".uinreg"
#define SEQUENTIAL_OUT_REGION_SUFFIX ".outreg"
#define USER_OUT_REGION_SUFFIX ".uoutreg"
