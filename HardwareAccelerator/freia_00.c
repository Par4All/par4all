typedef struct __freia_data_2d;
typedef struct __freia_data_2d * freia_data_2d;
typedef int freia_error;

freia_error freia_api_add(freia_data_2d o, freia_data_2d i0, freia_data_2d i1);
freia_error freia_api_sub(freia_data_2d o, freia_data_2d i0, freia_data_2d i1);

freia_error freia_stuff(freia_data_2d o0, freia_data_2d o1,
			freia_data_2d i0, freia_data_2d i1)
{
  freia_api_add(o0, i0, i1);
  freia_api_sub(o1, o0, i1);
  return 0;
}
