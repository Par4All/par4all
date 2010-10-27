
enum region_to_dma_switch { dma_load, dma_store, dma_allocate, dma_deallocate };
/* Add NewGen-like methods: */
#define dma_load_p(e) ((e) == dma_load )
#define dma_store_p(e) ((e) == dma_store )
#define dma_allocate_p(e) ((e) == dma_allocate )
#define dma_deallocate_p(e) ((e) == dma_deallocate )

#define OUTLINE_PRAGMA "outline this"
#define OUTLINE_IGNORE "outline_ignore"



