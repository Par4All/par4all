typedef enum { TOTO } smoothing_fun_t;


typedef struct {
	smoothing_fun_t fun;
} hs_config_t;



int main()
{
  hs_config_t config = { 0 };
  int n = config.fun;
  config.fun = TOTO;
  return(config.fun);
}

