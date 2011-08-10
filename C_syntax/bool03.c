/* The C code is wrong on purpose: checking error recovery */

struct token_pos
{
    char *beg;
    char *end;
};

typedef struct token_pos token_pos_t;

static void
tokens_grow(token_pos_t **token_ptr, int *token_lim_ptr, bool tokens_on_heap)
{
}
