/* Bug found in EffectsWithPointsTo/array03.c
 *
 * Excerpt
 *
 * Another bug found in effects_with_points_to_computation(): a wrong
 * 0 subscript is added to all references. For instance b[0][3] is
 * transformed into b[0][0][3]. See:
 *
 * simple_effect_to_constant_path_effects_with_points_to()
 *
 * eval_simple_cell_with_points_to()
 *
 * generic_eval_cell_with_points_to()
 *  current_max_path_length is computed by reference_to_points_to_matching_list 
 *  -> reference_to_points_to_matching_list()
 *      -> simple_cell_reference_preceding_p(),
 *      -> cell_reference_conversion_func = simple_reference_to_simple_reference_conversion()
 *
 *  -> transform_sink_cells_from_matching_list(.., current_max_path_length)
 *      How do you compute "nb_common_indices"? a.k.a. "current_max_path_length"
 *      -> simple_cell_reference_with_address_of_cell_reference_translation()
 */

#define N 5
#define M 3

float d[N][M];

int array15(float (*b)[M])
{
  float c;
  (*b)[3] = 2.0;
  c = (*b)[3];
  b[1][3] = 2.0;
  c = b[1][3];
  
  ((*b)[3])++;
  (*b)[3] += 5.0;
  (b[1][3])++;
  b[1][3] += 5.0;

  return (1);
}

int main() 
{
  float a[N][M], ret;
  
  ret = array15(a);
  
  return 1;
}
