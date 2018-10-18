/* TPL_HEADER_START */
/* The following lines are automatically generated. DO NOT EDIT. */
/* TPL_HEADER_END */
void shmem_TYPENAME_sum_to_all(TYPE * dest, const TYPE * source, int nreduce, int PE_start,
                               int logPE_stride, int PE_size, TYPE * pWrk, long *pSync);
void shmem_TYPENAME_prod_to_all(TYPE * dest, const TYPE * source, int nreduce, int PE_start,
                                int logPE_stride, int PE_size, TYPE * pWrk, long *pSync);
