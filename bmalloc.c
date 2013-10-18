#include <unistd.h>
#include <string.h>
#include <assert.h>

#include "shmem-internals.h"

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
	//void * ptr = (void *)((unsigned long)shmem_sheap_base_ptrs[_my_pe()] + (unsigned long)shmem_sheap_current_ptr);
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
                __shmem_abort(curr->size, "[E] Insufficient memory in pool");
	}
	
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
	
	shmem_sheap_current_ptr += size;

	shmem_barrier_all();
	
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

	shmem_barrier_all();
	
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

	/* Behaves like shmalloc if ptr is NULL */
	if (curr == NULL) {
		ptr = bmem_alloc (size);
		return ptr;
	}

	curr->size = size;
	/* First negate the size of the to-be-reallocated pointer */
	shmem_sheap_current_ptr -= size;

	void * new_ptr = bmem_alloc (size);
	memcpy (new_ptr, ptr, size);
	bmem_free (ptr); /* free old pointer */

	shmem_barrier_all();
	
	return new_ptr;	
}

/*
   The logic is taken from stackoverflow post:
http://stackoverflow.com/questions/227897/solve-the-memory-alignment-in-c-interview-question-that-stumped-me
 */
void * bmem_align (size_t alignment, size_t size)
{
	/* OpenSHMEM 1.0 spec says nothing about this case */
	if (alignment > size) {
		return NULL;
	}
	/* Allocate enough memory */	
	shmem_sheap_current_ptr += (size + alignment - 1);
	
	/* Notes: Sayan: This will flip the bits */
	uintptr_t mask = ~(uintptr_t)(alignment - 1);
	void * ptr = shmem_sheap_current_ptr;
	/*
	   The parameter size must be less than or equal to the amount of symmetric heap space
	   available for the calling PE; otherwise shmemalign returns NULL. - OpenSHMEM 1.0 spec
	 */
	if ((unsigned long)shmem_sheap_current_ptr > (unsigned long)shmem_sheap_size) return NULL;

	/* Notes: Sayan: Add alignment to the first pointer, suppose it (size+alignment-1)
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

	/*
	shmemalign() calls shmem_barrier_all() before returning to ensure that all the PEs par-
	ticipate - (same for shmalloc, shrealloc and shfree) : OpenSHMEM spec 1.0
	*/
	shmem_barrier_all();

	return mem;
}
