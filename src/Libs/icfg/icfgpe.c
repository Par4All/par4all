
text get_text_proper_effects_flt(string module_name, entity e_flt)
{
  text t;
  set_methodes_for_rw_effects_prettyprint(module_name);
  t = get_any_effect_type_text(module_name, DBR_PROPER_EFFECTS, string_undefined, TRUE);
  reset_methodes_for_effects_prettyprint(module_name);
  return t;
}
