#include <unistd.h>
#include <string.h>
#include <assert.h>

#include "shmem.h"

typedef struct ptr_sizes_s
{
	void * ptr;
	size_t size;
	struct ptr_sizes_s * next;
} ptr_size;

ptr_size * shmallocd_ptrs_sizes;

/* dispense mem from sheap in getpagesize() chunks */
void * bmem_alloc (size_t size)
{
	void * ptr = (void *)shmem_sheap_current_ptr;
	ptr_size * curr = (ptr_size *)malloc (sizeof(ptr_size));
	/* FIXME we might need a macro, as this particular signature
	   may not be available in all unix-like systems
	 */	
	long pg_sz = sysconf(_SC_PAGESIZE);
	if ((size % pg_sz) == 0)	
		curr->size = size;
	else { 
		size_t align_bump = (size%pg_sz ? 1 : 0);
    		size_t align_size = (size/pg_sz + align_bump) * pg_sz;
		curr->size = align_size; 
	}

	if (curr->size >= shmem_sheap_size) {
		printf ("[E] Insufficient memory in pool\n");
		MPI_Abort (MPI_COMM_WORLD, curr->size);
	}
	
	curr->ptr = shmem_sheap_current_ptr;

	/* Current head */
	if (shmallocd_ptrs_sizes == NULL) {
		shmallocd_ptrs_sizes = curr;
		shmallocd_ptrs_sizes->next = NULL;
	}
	else {
		curr->next = shmallocd_ptrs_sizes;
		shmallocd_ptrs_sizes = curr;
	}

	return ptr;
}

void bmem_free (void * ptr)
{
	ptr_size * curr, * prev;
	curr = shmallocd_ptrs_sizes;
	prev = NULL;

	if (ptr == NULL) {
		printf ("[W] Invalid pointer to free\n");
		return;
	}
	
	/* search for ptr in the linked-list */
	while((curr->ptr != ptr) && (curr->next != NULL))
	{
		prev = curr;
		curr = curr->next;
	}
	/* ptr should be found */	
	if (curr->ptr == ptr) {
		if (prev == NULL) /* node is the first node which is at head */
			shmallocd_ptrs_sizes = curr->next;
		else	/* node is somewhere between head and tail */
			prev->next = curr->next;		
		
		free (curr);
	}

	return;
}

void * bmem_realloc (void * ptr, size_t size)
{
	/* Find passed pointer info */
	ptr_size * curr;
	curr = shmallocd_ptrs_sizes;
	
	while (curr) {
		if (curr->ptr == ptr)
			break;
		curr = curr->next;
	}

	if (curr == NULL) {
		printf ("[E] Invalid pointer to resize\n");
		MPI_Abort (MPI_COMM_WORLD, curr->size);
	}

	/* First negate the size of the to-be-reallocated pointer */
	shmem_sheap_current_ptr -= curr->size;
	/* New offset */
	shmem_sheap_current_ptr += size;
	
	if ((unsigned long)shmem_sheap_current_ptr > (unsigned long)shmem_sheap_size) {
		printf ("[E] Address not within symm heap range\n");
		MPI_Abort (MPI_COMM_WORLD, curr->size);
	}

	void * new_ptr = bmem_alloc (size);
	memcpy (new_ptr, ptr, size);
	bmem_free (ptr); /* free old pointer */

	return new_ptr;	
}
/*
   The logic is taken from stackoverflow post:
http://stackoverflow.com/questions/227897/solve-the-memory-alignment-in-c-interview-question-that-stumped-me
 */
void * bmem_align (size_t alignment, size_t size)
{
	/* Notes: Sayan: This will flip the bits */
	uintptr_t mask = ~(uintptr_t)(alignment - 1);
	void * ptr = shmem_sheap_current_ptr;
	shmem_sheap_current_ptr += size;
	
	if ((unsigned long)shmem_sheap_current_ptr > (unsigned long)shmem_sheap_size) {
		printf ("[E] Address not within symm heap range\n");
		MPI_Abort (MPI_COMM_WORLD, size);
	}
	
	/* Notes: Sayan: Add alignment to the first pointer, suppose it
	returns a bad alignment, then fix it by and-ing with mask, eg: 1+0 = 0 */
	void * mem = (void *)(((uintptr_t)ptr + alignment - 1) & mask);
		
	/* book-keeping */
	ptr_size * curr = (ptr_size *)malloc (sizeof(ptr_size));
	curr->size = size;
	curr->ptr = ptr;

	/* Current head */
	if (shmallocd_ptrs_sizes == NULL) {
		shmallocd_ptrs_sizes = curr;
		shmallocd_ptrs_sizes->next = NULL;
	}
	else {
		curr->next = shmallocd_ptrs_sizes;
		shmallocd_ptrs_sizes = curr;
	}

	return mem;
}
