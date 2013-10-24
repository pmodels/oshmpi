#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <shmem.h>

#define DEFAULT_SIZE		(1<<20)
#define DEFAULT_ITERS		10000
#define SIZEOF_LONG		sizeof(long)

int _world_rank, _world_size;

/* Aggregate b/w and latency */
/* long* msg_size bytes is sent from lo_rank to lo_rank+1, lo_rank+2 .... hi_rank 
   all other processes sends data back to lo_rank
 */
long pSync0[_SHMEM_BARRIER_SYNC_SIZE],
     pSync1[_SHMEM_BARRIER_SYNC_SIZE],
     pSync2[_SHMEM_BARRIER_SYNC_SIZE];
double pWrk0[_SHMEM_REDUCE_MIN_WRKDATA_SIZE], 
       pWrk1[_SHMEM_REDUCE_MIN_WRKDATA_SIZE],
       pWrk2[_SHMEM_REDUCE_MIN_WRKDATA_SIZE];    

void ping_pong_lbw(int lo_rank, 
		int hi_rank, 
		int logPE_stride,
		unsigned int msg_size /* actual msg size is 2^msg_size */)
{
	double time_start, time_end;
	if ((hi_rank - lo_rank + 1) > _world_size)
		return;

	unsigned int nelems = (msg_size);
	if (_world_rank == 0)
		printf("Message size: %lu\n", nelems*sizeof(long));

	long * sendbuf = shmalloc(nelems*SIZEOF_LONG);
	long * recvbuf = shmalloc(nelems*SIZEOF_LONG);

	for (int j = 0; j < DEFAULT_ITERS; j++) {
		/* Initialize arrays */
		for (int i = 0; i < nelems; i++) {
			sendbuf[i] = i;
			recvbuf[i] = -99;
		}
		shmem_barrier(lo_rank, logPE_stride, (hi_rank - lo_rank + 1), pSync1);
		time_start = shmem_wtime();

		/* From PE lo_rank+1 till hi_rank with increments of 2^logPE_stride = 1<<logPE_stride */
		if (_world_rank == lo_rank) {
			for (int i = lo_rank+1; i < hi_rank+1; i+=(1 << logPE_stride))
				shmem_long_put(recvbuf, sendbuf, nelems, i);
		}
		else {
			for (int i = 0; i < nelems; i++)
				shmem_wait(&recvbuf[i], -99);
#if SHMEM_DEBUG > 2		
			printf("[%d]: Waiting for puts to complete from rank = %d\n", _world_rank, lo_rank);	
#endif	
		}

		for (int i = 0; i < nelems; i++) 
			sendbuf[i] = -99;

		shmem_barrier(lo_rank, logPE_stride, (hi_rank - lo_rank + 1), pSync0);

		/* From rest of the PEs to PE lo_rank */
		if (_world_rank != lo_rank) {
			shmem_long_put(recvbuf, sendbuf, nelems, lo_rank);
		}
		else { 
			for (int i = 0; i < nelems; i++)
				shmem_wait(&recvbuf[i], i);
#if SHMEM_DEBUG > 2		
			printf("[%d]: Waiting for puts to complete from rank = %d\n", lo_rank, _world_rank);	
#endif	
		}
		shmem_barrier(lo_rank, logPE_stride, (hi_rank - lo_rank + 1), pSync1);
		time_end = shmem_wtime();
	}
	/* Compute average by reducing this total_clock_time */
	double clock_time_PE = time_end - time_start;
	double total_clock_time, max_clock_time, min_clock_time;
	shmem_double_sum_to_all(&total_clock_time, &clock_time_PE, 1, 
			0, 0, (hi_rank-lo_rank+1), pWrk0, pSync0);
	shmem_double_max_to_all(&max_clock_time, &clock_time_PE, 1, 
			0, 0, (hi_rank-lo_rank+1), pWrk1, pSync1);
	shmem_double_min_to_all(&min_clock_time, &clock_time_PE, 1, 
			0, 0, (hi_rank-lo_rank+1), pWrk2, pSync2);    
	//MPI_Reduce(&clock_time_PE, &total_clock_time, 1, MPI_DOUBLE, MPI_SUM, 0, SHMEM_COMM_WORLD);
	if (_world_rank == 0) {
		printf("Avg. Latency : %f, Avg. Bandwidth: %f MB/s for Message size: %lu bytes\n", (total_clock_time/_world_size), 
				(double)((nelems*SIZEOF_LONG)/(double)((total_clock_time/_world_size) * 1024 * 1024)), nelems*SIZEOF_LONG);
		printf("Max. Latency : %f, Min. Bandwidth: %f MB/s for Message size: %lu bytes\n", (max_clock_time), 
				(double)((nelems*SIZEOF_LONG)/(double)(max_clock_time * 1024 * 1024)), nelems*SIZEOF_LONG);
		printf("Min. Latency : %f, Max. Bandwidth: %f MB/s for Message size: %lu bytes\n", (min_clock_time), 
				(double)((nelems*SIZEOF_LONG)/(double)(min_clock_time * 1024 * 1024)), nelems*SIZEOF_LONG);
	}

	shfree(sendbuf);
	shfree(recvbuf);

	return;

}

/* long* msg_size bytes is PUT from left rank to right rank
 * in the fashion of a natural ring (ordered according to PE ranks)
 */
void natural_ring_lbw (unsigned int msg_size /* actual msg size is 2^msg_size */)
{
	unsigned int nelems = (msg_size);
	unsigned int left_rank, right_rank;
	double time_end, time_start;
	if (_world_rank == 0)
		printf("Message size: %lu\n", nelems*sizeof(long));

	long * sendbuf = shmalloc(nelems*SIZEOF_LONG);
	long * recvbuf = shmalloc(nelems*SIZEOF_LONG);

	for (int j = 0; j < DEFAULT_ITERS; j++) {
		/* Initialize arrays */
		for (int i = 0; i < nelems; i++) {
			sendbuf[i] = i;
			recvbuf[i] = -99;
		}
		shmem_barrier_all();

		left_rank = (_world_rank - 1 + _world_size)%_world_size;
		right_rank = (_world_rank + 1)%_world_size;
		time_start = shmem_wtime();

		if (_world_rank == left_rank) {		
			shmem_fence();
			shmem_long_put(recvbuf, sendbuf, nelems, right_rank);
		}
		/* Right rank waits till it receives the data */
		if (_world_rank == right_rank) {
			for (int i = 0; i < nelems; i++)
				shmem_wait(&recvbuf[i], -99);
		}
		/* Repeat above, now other way round */
		/* Re-Initialize arrays */
		for (int i = 0; i < nelems; i++) {
			sendbuf[i] = -99;
		}
		shmem_barrier_all();

		if (_world_rank == right_rank) {		
			shmem_fence();
			shmem_long_put(recvbuf, sendbuf, nelems, right_rank);
		}
		/* Right rank waits till it receives the data */
		if (_world_rank == left_rank) {
			for (int i = 0; i < nelems; i++)
				shmem_wait(&recvbuf[i], i);
		}


		time_end = shmem_wtime();
	}
	shmem_barrier_all();

	/* Compute average by reducing this total_clock_time */
	double clock_time_PE = time_end - time_start;
	double total_clock_time, max_clock_time, min_clock_time;
	shmem_double_sum_to_all(&total_clock_time, &clock_time_PE, 1, 
			0, 0, _world_size, pWrk0, pSync0);
	shmem_double_max_to_all(&max_clock_time, &clock_time_PE, 1, 
			0, 0, _world_size, pWrk1, pSync1);
	shmem_double_min_to_all(&min_clock_time, &clock_time_PE, 1, 
			0, 0, _world_size, pWrk2, pSync2);    
	//MPI_Reduce(&clock_time_PE, &total_clock_time, 1, MPI_DOUBLE, MPI_SUM, 0, SHMEM_COMM_WORLD);
	if (_world_rank == 0) {
		printf("Avg. Latency : %f, Avg. Bandwidth: %f MB/s for Message size: %lu bytes\n", (total_clock_time/_world_size), 
				(double)((nelems*SIZEOF_LONG)/(double)((total_clock_time/_world_size) * 1024 * 1024)), nelems*SIZEOF_LONG);
		printf("Max. Latency : %f, Min. Bandwidth: %f MB/s for Message size: %lu bytes\n", (max_clock_time), 
				(double)((nelems*SIZEOF_LONG)/(double)(max_clock_time * 1024 * 1024)), nelems*SIZEOF_LONG);
		printf("Min. Latency : %f, Max. Bandwidth: %f MB/s for Message size: %lu bytes\n", (min_clock_time), 
				(double)((nelems*SIZEOF_LONG)/(double)(min_clock_time * 1024 * 1024)), nelems*SIZEOF_LONG);
	}

	shfree(sendbuf);
	shfree(recvbuf);

	return;

}

/* Taken from: http://stackoverflow.com/questions/6127503/shuffle-array-in-c
 */
/* Arrange the N elements of ARRAY in random order.
   Only effective if N is much smaller than RAND_MAX;
   if this may not be the case, use a better random
   number generator. */
void shuffle(int *array, size_t n)
{
	if (n > 1) 
	{
		size_t i;
		for (i = 0; i < n - 1; i++) 
		{
			size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
			array[i] ^= array[j]; 
			array[j] ^= array [i]; 
			array [i] ^= array [j];
		}
	}
	return;
}
/* long* msg_size bytes is PUT from left rank to right rank
 * in the fashion of a ring (ranks ordered in a random fashion)
 */
void random_ring_lbw (unsigned int msg_size /* actual msg size is 2^msg_size */)
{
	unsigned int nelems = (msg_size);
	unsigned int left_rank, right_rank;
	double time_end, time_start;
	if (_world_rank == 0)
		printf("Message size: %lu\n", nelems*sizeof(long));

	for (int i = 0; i < _SHMEM_BCAST_SYNC_SIZE; i++)
		pSync0[i] = _SHMEM_SYNC_VALUE;

	/* Construct a random pattern and bcast it to all ranks */
	int * proclist = shmalloc(sizeof(int)*(_world_size+1));
	if (_world_rank == 0) {
		for (int i = 0; i < _world_size; i++)
			proclist[i] = i;
		proclist[_world_size] = 0;
		shuffle(proclist, (_world_size+1));
	}    

	shmem_broadcast32(proclist, proclist, (_world_size+1), 0, 0, 0, _world_size, pSync0);

	long * sendbuf = shmalloc(nelems*SIZEOF_LONG);
	long * recvbuf = shmalloc(nelems*SIZEOF_LONG);

	for (int j = 0; j < DEFAULT_ITERS; j++) {
		/* Initialize arrays */
		for (int i = 0; i < nelems; i++) {
			sendbuf[i] = i;
			recvbuf[i] = -99;
		}
		shmem_barrier_all();

		left_rank = proclist[(((_world_rank-1) < 0) ? _world_size-1 : (_world_rank-1))];
		right_rank = proclist[(((_world_rank+1) > _world_size-1) ? 0 : (_world_rank+1))];
		time_start = shmem_wtime();

		if (_world_rank == left_rank) {		
			shmem_fence();
			shmem_long_put(recvbuf, sendbuf, nelems, right_rank);
		}
		/* Right rank waits till it receives the data */
		if (_world_rank == right_rank) {
			for (int i = 0; i < nelems; i++)
				shmem_wait(&recvbuf[i], -99);
		}
		/* Repeat above, now other way round */
		/* Re-Initialize arrays */
		for (int i = 0; i < nelems; i++) {
			sendbuf[i] = -99;
		}
		shmem_barrier_all();

		if (_world_rank == right_rank) {		
			shmem_fence();
			shmem_long_put(recvbuf, sendbuf, nelems, right_rank);
		}
		/* Right rank waits till it receives the data */
		if (_world_rank == left_rank) {
			for (int i = 0; i < nelems; i++)
				shmem_wait(&recvbuf[i], i);
		}


		time_end = shmem_wtime();
	}
	shmem_barrier_all();

	/* Compute average by reducing this total_clock_time */
	double clock_time_PE = time_end - time_start;
	double total_clock_time, max_clock_time, min_clock_time;
	shmem_double_sum_to_all(&total_clock_time, &clock_time_PE, 1, 
			0, 0, _world_size, pWrk0, pSync0);
	shmem_double_max_to_all(&max_clock_time, &clock_time_PE, 1, 
			0, 0, _world_size, pWrk1, pSync1);
	shmem_double_min_to_all(&min_clock_time, &clock_time_PE, 1, 
			0, 0, _world_size, pWrk2, pSync2);    
	//MPI_Reduce(&clock_time_PE, &total_clock_time, 1, MPI_DOUBLE, MPI_SUM, 0, SHMEM_COMM_WORLD);
	if (_world_rank == 0) {
		printf("Avg. Latency : %f, Avg. Bandwidth: %f MB/s for Message size: %lu bytes\n", (total_clock_time/_world_size), 
				(double)((nelems*SIZEOF_LONG)/(double)((total_clock_time/_world_size) * 1024 * 1024)), nelems*SIZEOF_LONG);
		printf("Max. Latency : %f, Min. Bandwidth: %f MB/s for Message size: %lu bytes\n", (max_clock_time), 
				(double)((nelems*SIZEOF_LONG)/(double)(max_clock_time * 1024 * 1024)), nelems*SIZEOF_LONG);
		printf("Min. Latency : %f, Max. Bandwidth: %f MB/s for Message size: %lu bytes\n", (min_clock_time), 
				(double)((nelems*SIZEOF_LONG)/(double)(min_clock_time * 1024 * 1024)), nelems*SIZEOF_LONG);
	}

	shfree(sendbuf);
	shfree(recvbuf);
	shfree(proclist);
	return;

}
/* target_rank = world_size - world_rank - 1 */
void link_contended_lbw (unsigned int msg_size /* actual msg size is 2^msg_size */)
{
	unsigned int nelems = (msg_size);
	double time_end, time_start;
	unsigned int target_rank;
	if (_world_rank == 0)
		printf("Message size: %lu\n", nelems*sizeof(long));

	long * sendbuf = shmalloc(nelems*SIZEOF_LONG);
	long * recvbuf = shmalloc(nelems*SIZEOF_LONG);

	for (int j = 0; j < DEFAULT_ITERS; j++) {
		/* Initialize arrays */
		for (int i = 0; i < nelems; i++) {
			sendbuf[i] = i;
			recvbuf[i] = -99;
		}

		target_rank = (_world_size - _world_rank - 1);

		time_start = shmem_wtime();

		if (_world_rank != target_rank)
			shmem_long_put(recvbuf, sendbuf, nelems, target_rank);
		else {
			for (int i = 0; i < nelems; i++)
				shmem_wait(&recvbuf[i],-99);
		}

		shmem_quiet();

		/* Re-initialize */
		for (int i = 0; i < nelems; i++) {
			sendbuf[i] = -99;
		}

		if (_world_rank == target_rank)
			shmem_long_get(sendbuf, recvbuf, nelems, target_rank);
		else { /* Wait */
			for (int i = 0; i < nelems; i++)
				shmem_wait(&sendbuf[i],i);
		}

		shmem_barrier_all();

		time_end = shmem_wtime();
	}
	/* Compute average by reducing this total_clock_time */
	double clock_time_PE = time_end - time_start;
	double total_clock_time, max_clock_time, min_clock_time;
	shmem_double_sum_to_all(&total_clock_time, &clock_time_PE, 1, 
			0, 0, _world_size, pWrk0, pSync0);
	shmem_double_max_to_all(&max_clock_time, &clock_time_PE, 1, 
			0, 0, _world_size, pWrk1, pSync1);
	shmem_double_min_to_all(&min_clock_time, &clock_time_PE, 1, 
			0, 0, _world_size, pWrk2, pSync2);    
	//MPI_Reduce(&clock_time_PE, &total_clock_time, 1, MPI_DOUBLE, MPI_SUM, 0, SHMEM_COMM_WORLD);
	if (_world_rank == 0) {
		printf("Avg. Latency : %f, Avg. Bandwidth: %f MB/s for Message size: %lu bytes\n", (total_clock_time/_world_size), 
				(double)((nelems*SIZEOF_LONG)/(double)((total_clock_time/_world_size) * 1024 * 1024)), nelems*SIZEOF_LONG);
		printf("Max. Latency : %f, Min. Bandwidth: %f MB/s for Message size: %lu bytes\n", (max_clock_time), 
				(double)((nelems*SIZEOF_LONG)/(double)(max_clock_time * 1024 * 1024)), nelems*SIZEOF_LONG);
		printf("Min. Latency : %f, Max. Bandwidth: %f MB/s for Message size: %lu bytes\n", (min_clock_time), 
				(double)((nelems*SIZEOF_LONG)/(double)(min_clock_time * 1024 * 1024)), nelems*SIZEOF_LONG);
	}
	/* Verify */
	if (_world_rank == 0) {

	}

	shfree(sendbuf);
	shfree(recvbuf);

	return;

}

/* scatter/gather: send/receive data from PE_root to/from all other PEs */
void scatter_gather_lbw (int PE_root,
		int logPE_stride,
		int scatter_or_gather, /* 1 for scatter, 0 for gather */
		unsigned int msg_size /* actual msg size is 2^msg_size */)
{
	double time_end, time_start;
	long * sendbuf, * recvbuf;
	if (PE_root < 0 && PE_root >= _world_size)
		return;

	unsigned int nelems = (msg_size); 
	if (_world_rank == 0)
		printf("Message size: %lu\n", nelems*sizeof(long));

	if (scatter_or_gather) { /* Scatter */
		sendbuf = shmalloc(nelems*_world_size*SIZEOF_LONG);
		recvbuf = shmalloc(nelems*SIZEOF_LONG);
	}
	else { /* Gather */
		sendbuf = shmalloc(nelems*SIZEOF_LONG);
		recvbuf = shmalloc(nelems * _world_size * SIZEOF_LONG);
	}

	for (int j = 0; j < DEFAULT_ITERS; j++) {
		if (scatter_or_gather) { /* Scatter */
			/* Initialize arrays */
			for (int i = 0; i < nelems; i++) {
				recvbuf[i] = -99;
			}

			for (int i = 0; i < (nelems*_world_size); i++) {
				sendbuf[i] = i;
			}  

		}
		else { /* Gather */
			/* Initialize arrays */
			for (int i = 0; i < (nelems*_world_size); i++) {
				recvbuf[i] = -99;
			}

			for (int i = 0; i < nelems; i++) {
				sendbuf[i] = i;
			}  
		}

		shmem_barrier_all();

		time_start = shmem_wtime();

		if (_world_rank == PE_root && scatter_or_gather) {/* Scatter to rest of the PEs */
			for (int i = 0; i < _world_size; i+=(1 << logPE_stride)) {
				shmem_long_put(recvbuf, (sendbuf+i*nelems), nelems, i);
			}
		}
		if (_world_rank != PE_root && scatter_or_gather) {/* wait */
			for (int i = 0; i < nelems; i++)
				shmem_wait(&recvbuf[i], -99);
		}

		if (!scatter_or_gather) {/* Gather from rest of the PEs */
			for (int i = 0; i < _world_size; i+=(1 << logPE_stride)) {
				shmem_long_put((recvbuf+i*nelems), sendbuf, nelems, PE_root);
			}
		}
		if (_world_rank == PE_root && !scatter_or_gather) {/* wait */
			for (int i = 0; i < nelems*_world_size; i++)
				shmem_wait(&recvbuf[i], -99);
		}

		time_end = shmem_wtime();  
	}
	shmem_barrier_all();
	/* Compute average by reducing this total_clock_time */
	double clock_time_PE = time_end - time_start;
	double total_clock_time, max_clock_time, min_clock_time;
	shmem_double_sum_to_all(&total_clock_time, &clock_time_PE, 1, 
			0, 0, _world_size, pWrk0, pSync0);
	shmem_double_max_to_all(&max_clock_time, &clock_time_PE, 1, 
			0, 0, _world_size, pWrk1, pSync1);
	shmem_double_min_to_all(&min_clock_time, &clock_time_PE, 1, 
			0, 0, _world_size, pWrk2, pSync2);    
	//MPI_Reduce(&clock_time_PE, &total_clock_time, 1, MPI_DOUBLE, MPI_SUM, 0, SHMEM_COMM_WORLD);
	if (_world_rank == 0) {
		printf("Avg. Latency : %f, Avg. Bandwidth: %f MB/s for Message size: %lu bytes\n", (total_clock_time/_world_size), 
				(double)((nelems*SIZEOF_LONG)/(double)((total_clock_time/_world_size) * 1024 * 1024)), nelems*SIZEOF_LONG);
		printf("Max. Latency : %f, Min. Bandwidth: %f MB/s for Message size: %lu bytes\n", (max_clock_time), 
				(double)((nelems*SIZEOF_LONG)/(double)(max_clock_time * 1024 * 1024)), nelems*SIZEOF_LONG);
		printf("Min. Latency : %f, Max. Bandwidth: %f MB/s for Message size: %lu bytes\n", (min_clock_time), 
				(double)((nelems*SIZEOF_LONG)/(double)(min_clock_time * 1024 * 1024)), nelems*SIZEOF_LONG);
	}

	shfree(sendbuf);
	shfree(recvbuf);

	return;

}

/* all-to-all pattern */
void a2a_lbw (int logPE_stride,
		unsigned int msg_size /* actual msg size is 2^msg_size */)
{	
	double time_end, time_start;
	unsigned int nelems = (msg_size); 
	if (_world_rank == 0)
		printf("Message size: %lu\n", nelems*sizeof(long));

	long * sendbuf = shmalloc(nelems*_world_size*SIZEOF_LONG);
	long * recvbuf = shmalloc(nelems*_world_size*SIZEOF_LONG);

	for (int j = 0; j < DEFAULT_ITERS; j++) {
		/* Initialize arrays */
		for (int i = 0; i < (nelems*_world_size); i++) {
			recvbuf[i] = -99;
			sendbuf[i] = i;
		}

		shmem_barrier_all();

		time_start = shmem_wtime();

		for (int i = 0; i < _world_size; i+=(1 << logPE_stride)) {
			shmem_long_put((sendbuf+i*nelems), (recvbuf+i*nelems), nelems, i);
			shmem_fence();
		}

		shmem_barrier_all();

		time_end = shmem_wtime();  
	}

	/* Compute average by reducing this total_clock_time */
	double clock_time_PE = time_end - time_start;
	double total_clock_time, max_clock_time, min_clock_time;
	shmem_double_sum_to_all(&total_clock_time, &clock_time_PE, 1, 
			0, 0, _world_size, pWrk0, pSync0);
	shmem_double_max_to_all(&max_clock_time, &clock_time_PE, 1, 
			0, 0, _world_size, pWrk1, pSync1);
	shmem_double_min_to_all(&min_clock_time, &clock_time_PE, 1, 
			0, 0, _world_size, pWrk2, pSync2);    
	//MPI_Reduce(&clock_time_PE, &total_clock_time, 1, MPI_DOUBLE, MPI_SUM, 0, SHMEM_COMM_WORLD);
	if (_world_rank == 0) {
		printf("Avg. Latency : %f, Avg. Bandwidth: %f MB/s for Message size: %lu bytes\n", (total_clock_time/_world_size), 
				(double)((nelems*SIZEOF_LONG)/(double)((total_clock_time/_world_size) * 1024 * 1024)), nelems*SIZEOF_LONG);
		printf("Max. Latency : %f, Min. Bandwidth: %f MB/s for Message size: %lu bytes\n", (max_clock_time), 
				(double)((nelems*SIZEOF_LONG)/(double)(max_clock_time * 1024 * 1024)), nelems*SIZEOF_LONG);
		printf("Min. Latency : %f, Max. Bandwidth: %f MB/s for Message size: %lu bytes\n", (min_clock_time), 
				(double)((nelems*SIZEOF_LONG)/(double)(min_clock_time * 1024 * 1024)), nelems*SIZEOF_LONG);
	}

	shfree(sendbuf);
	shfree(recvbuf);

	return;

}

int main(int argc, char * argv[])
{
	/* DEFAULT */
	int which_test = 0;

	/*OpenSHMEM initilization*/
	start_pes (0);
	_world_size = _num_pes ();
	_world_rank = _my_pe ();
	/* wait for user to input runtime params */
	for(int j=0; j < _SHMEM_BARRIER_SYNC_SIZE; j++) {
		pSync0[j] = pSync1[j] = pSync2[j] = _SHMEM_SYNC_VALUE;
	} 

	/* argument processing */
	if (argc < 2) {
		if (_world_rank == 0)
			printf ("Expected: ./a.out <1-7> ..using defaults\n");
	}
	else
		which_test = atoi(argv[1]);

	switch(which_test) {
		case 1:
			if (_world_rank == 0)
				printf("Ping-Pong tests on %d PEs\n",_world_size);
			for (int i = 1; i < DEFAULT_SIZE; i*=2)
				ping_pong_lbw(0,_world_size-1, 0, i);
			break;
		case 2:
			if (_world_rank == 0)
				printf("Natural ring test on %d PEs\n",_world_size);
			for (int i = 1; i < DEFAULT_SIZE; i*=2)
				natural_ring_lbw (i);
			break;
		case 3:
			if (_world_rank == 0)
				printf("Random ring test on %d PEs\n",_world_size);
			for (int i = 1; i < DEFAULT_SIZE; i*=2)
				random_ring_lbw (i);
			break;
		case 4:
			if (_world_rank == 0)
				printf("Link contended test on %d PEs\n",_world_size);
			for (int i = 1; i < DEFAULT_SIZE; i*=2)
				link_contended_lbw (i);
			break;
		case 5:
			if (_world_rank == 0)
				printf("Scatter test on %d PEs\n",_world_size);
			for (int i = 1; i < DEFAULT_SIZE; i*=2)
				scatter_gather_lbw (0, 0, 1, i);
			break;
		case 6:
			if (_world_rank == 0)
				printf("Gather test on %d PEs\n",_world_rank);
			for (int i = 1; i < DEFAULT_SIZE; i*=2)
				scatter_gather_lbw (0, 0, 0, i);
			break;
		case 7:
			if (_world_rank == 0)
				printf("All-to-all test on %d PEs\n",_world_rank);
			for (int i = 1; i < DEFAULT_SIZE; i*=2)
				a2a_lbw (0, i);
			break;
		case 8:
			if (_world_rank == 0)
				printf("Ping-Pong tests on %d PEs\n",_world_size);
			for (int i = 1; i < DEFAULT_SIZE; i*=2)
				ping_pong_lbw(0,_world_size-1, 0, i);
			if (_world_rank == 0)
				printf("Natural ring test on %d PEs\n",_world_size);
			for (int i = 1; i < DEFAULT_SIZE; i*=2)
				natural_ring_lbw (i);
			if (_world_rank == 0)
				printf("Random ring test on %d PEs\n",_world_size);
			for (int i = 1; i < DEFAULT_SIZE; i*=2)
				random_ring_lbw (i);
			if (_world_rank == 0)
				printf("Link contended test on %d PEs\n",_world_size);
			for (int i = 1; i < DEFAULT_SIZE; i*=2)
				link_contended_lbw (i);
			if (_world_rank == 0)
				printf("Scatter test on %d PEs\n",_world_size);
			for (int i = 1; i < DEFAULT_SIZE; i*=2)
				scatter_gather_lbw (0, 0, 1, i);
			if (_world_rank == 0)
				printf("Gather test on %d PEs\n",_world_rank);
			for (int i = 1; i < DEFAULT_SIZE; i*=2)
				scatter_gather_lbw (0, 0, 0, i);
			if (_world_rank == 0)
				printf("All-to-all test on %d PEs\n",_world_rank);
			for (int i = 1; i < DEFAULT_SIZE; i*=2)
				a2a_lbw (0, i);
			break;
		default:
			printf("Enter a number between 1-8\n");
	}

	return 0;
}
