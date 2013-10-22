#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <shmem.h>
#include <shmem-internals.h>

#define DEFAULT_TEST 		1
#define DEFAULT_ITERS		10
#define DEFAULT_LOG2_SIZE	8

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
	if ((_world_rank < lo_rank) || (_world_rank > hi_rank))
		return;

	if ((lo_rank >= hi_rank) || (lo_rank < 0) || (hi_rank < 0) || (hi_rank - lo_rank + 1) > _world_size)
		__shmem_abort(101, "Check passed hi/lo ranks");

	unsigned int current_size = (msg_size<<1)*sizeof(long);

	if (current_size >= shmem_sheap_size)
		__shmem_abort((msg_size<<1), "Message size should be within the sheap size");

	long * sendbuf = shmalloc(current_size);
	long * recvbuf = shmalloc(current_size);

	/* Initialize arrays */
	for (int i = 0; i < (current_size/sizeof(long)); i++) {
		sendbuf[i] = _world_rank;
		recvbuf[i] = -99;
	}

	double time_start = shmem_wtime();
	/* From PE 0 till hi_rank with increments of 1 */
	if (_world_rank == lo_rank) {
		for (int i = lo_rank+1; i <= hi_rank; i+=(logPE_stride<<1)) {  
			shmem_long_put(recvbuf, sendbuf, current_size, i);
		}
	}

	shmem_barrier(lo_rank, logPE_stride, (hi_rank-lo_rank+1), pSync0);

	/* From rest of the PEs to PE 0*/
	if (_world_rank != 0) {
		shmem_long_put(sendbuf, recvbuf, current_size, 0);
	}

	shmem_barrier(lo_rank, logPE_stride, (hi_rank-lo_rank+1), pSync1);
	double time_end = shmem_wtime();

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
		printf("Avg. Latency : %f, Avg. Bandwidth: %f MB/s for Message size: %u bytes\n", (total_clock_time/_world_size), 
				(double)(current_size/(double)(total_clock_time * 1024 * 1024)), current_size);
		printf("Max. Latency : %f, Min. Bandwidth: %f MB/s for Message size: %u bytes\n", (max_clock_time), 
				(double)(current_size/(double)(max_clock_time * 1024 * 1024)), current_size);
		printf("Min. Latency : %f, Max. Bandwidth: %f MB/s for Message size: %u bytes\n", (min_clock_time), 
				(double)(current_size/(double)(min_clock_time * 1024 * 1024)), current_size);
	}
	/* Verify */
	if (_world_rank == 0) {

	}

	shfree(sendbuf);
	shfree(recvbuf);

	return;

}

/* long* msg_size bytes is PUT from lo_rank to lo_rank+1, lo_rank+1 to lo_rank+2 .... hi_rank to lo_rank
 * in the fashion of a natural ring
 */
void natural_ring_lbw (unsigned int msg_size /* actual msg size is 2^msg_size */)
{
	unsigned int current_size = (msg_size<<1)*sizeof(long);

	if (current_size >= shmem_sheap_size)
		__shmem_abort((msg_size<<1), "Message size should be within the sheap size");

	long * sendbuf = shmalloc(current_size);
	long * recvbuf = shmalloc(current_size);

	/* Initialize arrays */
	for (int i = 0; i < (current_size/sizeof(long)); i++) {
		sendbuf[i] = _world_rank;
		recvbuf[i] = -99;
	}

	unsigned int target_rank = (_world_rank + 1)%_world_size;

	double time_start = shmem_wtime();

	shmem_long_put(sendbuf, recvbuf, current_size, target_rank);

	shmem_fence();

	shmem_long_get(recvbuf, sendbuf, current_size, target_rank);

	shmem_fence();    

	double time_end = shmem_wtime();

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
		printf("Avg. Latency : %f, Avg. Bandwidth: %f MB/s for Message size: %u bytes\n", (total_clock_time/_world_size), 
				(double)(current_size/(double)(total_clock_time * 1024 * 1024)), current_size);
		printf("Max. Latency : %f, Min. Bandwidth: %f MB/s for Message size: %u bytes\n", (max_clock_time), 
				(double)(current_size/(double)(max_clock_time * 1024 * 1024)), current_size);
		printf("Min. Latency : %f, Max. Bandwidth: %f MB/s for Message size: %u bytes\n", (min_clock_time), 
				(double)(current_size/(double)(min_clock_time * 1024 * 1024)), current_size);
	}
	/* Verify */
	if (_world_rank == 0) {

	}

	shfree(sendbuf);
	shfree(recvbuf);

	return;

}

/* target_rank = world_size - world_rank */
void link_contended_lbw (unsigned int msg_size /* actual msg size is 2^msg_size */)
{
	unsigned int current_size = (msg_size<<1)*sizeof(long);

	if (current_size >= shmem_sheap_size)
		__shmem_abort((msg_size<<1), "Message size should be within the sheap size");

	long * sendbuf = shmalloc(current_size);
	long * recvbuf = shmalloc(current_size);

	/* Initialize arrays */
	for (int i = 0; i < (current_size/sizeof(long)); i++) {
		sendbuf[i] = _world_rank;
		recvbuf[i] = -99;
	}

	unsigned int target_rank = _world_size - _world_rank;

	double time_start = shmem_wtime();

	shmem_long_put(sendbuf, recvbuf, current_size, target_rank);

	shmem_fence();

	shmem_long_get(recvbuf, sendbuf, current_size, target_rank);

	shmem_fence();

	double time_end = shmem_wtime();

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
		printf("Avg. Latency : %f, Avg. Bandwidth: %f MB/s for Message size: %u bytes\n", (total_clock_time/_world_size), 
				(double)(current_size/(double)(total_clock_time * 1024 * 1024)), current_size);
		printf("Max. Latency : %f, Min. Bandwidth: %f MB/s for Message size: %u bytes\n", (max_clock_time), 
				(double)(current_size/(double)(max_clock_time * 1024 * 1024)), current_size);
		printf("Min. Latency : %f, Max. Bandwidth: %f MB/s for Message size: %u bytes\n", (min_clock_time), 
				(double)(current_size/(double)(min_clock_time * 1024 * 1024)), current_size);
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
	long * sendbuf, * recvbuf;
	if (PE_root < 0 && PE_root >= _world_size)
		return;

	unsigned int current_size = (msg_size<<1)*sizeof(long); 

	if (current_size >= shmem_sheap_size)
		__shmem_abort((msg_size<<1), "Message size should be within the sheap size");

	if (scatter_or_gather) {
		sendbuf = shmalloc(current_size*_world_size);
		recvbuf = shmalloc(current_size);

		/* Initialize arrays */
		for (int i = 0; i < (current_size/sizeof(long)); i++) {
			recvbuf[i] = -99;
		}

		for (int i = 0; i < (current_size/sizeof(long))*_world_size; i++) {
			sendbuf[i] = _world_rank;
		}  

	}
	else {
		sendbuf = shmalloc(current_size);
		recvbuf = shmalloc(current_size * _world_size);

		/* Initialize arrays */
		for (int i = 0; i < (current_size/sizeof(long))*_world_size; i++) {
			recvbuf[i] = -99;
		}

		for (int i = 0; i < (current_size/sizeof(long)); i++) {
			sendbuf[i] = _world_rank;
		}  

	}

	shmem_barrier_all();

	double time_start = shmem_wtime();

	if (_world_rank == PE_root && scatter_or_gather) {/* Scatter to rest of the PEs */
		for (int i = 0; i < _world_size; i+=(logPE_stride << 1)) {
			if (i != PE_root)
				shmem_long_put((sendbuf+i*current_size), recvbuf, current_size, i);
		}
	}

	if (_world_rank == PE_root && !scatter_or_gather) {/* Gather from rest of the PEs */
		for (int i = 0; i < _world_size; i++) {
			if (i != PE_root)
				shmem_long_get((recvbuf+i*current_size), sendbuf, current_size, i);
		}
	}

	shmem_barrier_all();

	double time_end = shmem_wtime();  

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
		printf("Avg. Latency : %f, Avg. Bandwidth: %f MB/s for Message size: %u bytes\n", (total_clock_time/_world_size), 
				(double)(current_size/(double)(total_clock_time * 1024 * 1024)), current_size);
		printf("Max. Latency : %f, Min. Bandwidth: %f MB/s for Message size: %u bytes\n", (max_clock_time), 
				(double)(current_size/(double)(max_clock_time * 1024 * 1024)), current_size);
		printf("Min. Latency : %f, Max. Bandwidth: %f MB/s for Message size: %u bytes\n", (min_clock_time), 
				(double)(current_size/(double)(min_clock_time * 1024 * 1024)), current_size);
	}
	/* Verify */
	if (_world_rank == 0) {

	}

	shfree(sendbuf);
	shfree(recvbuf);

	return;

}

/* all-to-all pattern */
void a2a_lbw (int logPE_stride,
		unsigned int msg_size /* actual msg size is 2^msg_size */)
{
	if (logPE_stride < 0)
		logPE_stride = 0;

	unsigned int current_size = (msg_size<<1)*sizeof(long); 

	if (current_size >= shmem_sheap_size)
		__shmem_abort((msg_size<<1), "Message size should be within the sheap size");

	long * sendbuf = shmalloc(current_size*_world_size);
	long * recvbuf = shmalloc(current_size*_world_size);

	/* Initialize arrays */
	for (int i = 0; i < (current_size/sizeof(long)*_world_size); i++) {
		recvbuf[i] = -99;
		sendbuf[i] = _world_rank;
	}

	shmem_barrier_all();

	double time_start = shmem_wtime();

	for (int i = 0; i < _world_size; i+=(logPE_stride << 1)) {
		shmem_long_put((sendbuf+i*current_size), (recvbuf+i*current_size), current_size, i);
	}

	shmem_barrier_all();

	double time_end = shmem_wtime();  

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
		printf("Avg. Latency : %f, Avg. Bandwidth: %f MB/s for Message size: %u bytes\n", (total_clock_time/_world_size), 
				(double)(current_size/(double)(total_clock_time * 1024 * 1024)), current_size);
		printf("Max. Latency : %f, Min. Bandwidth: %f MB/s for Message size: %u bytes\n", (max_clock_time), 
				(double)(current_size/(double)(max_clock_time * 1024 * 1024)), current_size);
		printf("Min. Latency : %f, Max. Bandwidth: %f MB/s for Message size: %u bytes\n", (min_clock_time), 
				(double)(current_size/(double)(min_clock_time * 1024 * 1024)), current_size);
	}
	/* Verify */
	if (_world_rank == 0) {

	}

	shfree(sendbuf);
	shfree(recvbuf);

	return;

}

int main(int argc, char * argv[])
{
	/* DEFAULT */
	unsigned int log2_size = DEFAULT_LOG2_SIZE;
	int which_test = DEFAULT_TEST;
	int iterations = DEFAULT_ITERS;

	/*OpenSHMEM initilization*/
	start_pes (0);
	_world_size = _num_pes ();
	_world_rank = _my_pe ();
	/* wait for user to input runtime params */
	for(int j=0; j < _SHMEM_BARRIER_SYNC_SIZE; j++) {
		pSync0[j] = pSync1[j] = pSync2[j] = _SHMEM_SYNC_VALUE;
	} 

	/* argument processing */
	if (argc < 2)
		printf ("expected: ./a.out <log-base-2-start-size> <1-7> <iterations>..using defaults\n");
	else {
		log2_size = atoi(argv[1]);
		which_test = atoi(argv[2]);
		iterations = atoi(argv[3]);
	}
#if SHMEM_DEBUG > 1	
	printf("log2_start_size = %u, which_test = %d and iterations = %d\n", log2_size, which_test, iterations);
#endif	
	switch(which_test) {
		case 1:
			if (_world_rank == 0)
				printf("Ping-Pong tests\n");
			for (int i = log2_size; i < iterations; i*=2)
				ping_pong_lbw(0,_world_size-1, 0, i);
			break;
		case 2:
			if (_world_rank == 0)
				printf("Natural ring test\n");
			for (int i = log2_size; i < iterations; i*=2)
				natural_ring_lbw (i);
			break;
		case 3:
			if (_world_rank == 0)
				printf("Link contended test\n");
			for (int i = log2_size; i < iterations; i*=2)
				link_contended_lbw (i);
			break;
		case 4:
			if (_world_rank == 0)
				printf("Scatter test\n");
			for (int i = log2_size; i < iterations; i*=2)
				scatter_gather_lbw (0, 0, 1, i);
			break;
		case 5:
			if (_world_rank == 0)
				printf("Gather test\n");
			for (int i = log2_size; i < iterations; i*=2)
				scatter_gather_lbw (0, 0, 0, i);
			break;
		case 6:
			if (_world_rank == 0)
				printf("All-to-all test\n");
			for (int i = log2_size; i < iterations; i*=2)
				a2a_lbw (0, i);
			break;
		case 7:
			if (_world_rank == 0)
				printf("Ping-Pong tests\n");
			for (int i = log2_size; i < iterations; i*=2)
				ping_pong_lbw(0,_world_size-1, 0, i);
			if (_world_rank == 0)
				printf("Natural ring test\n");
			for (int i = log2_size; i < iterations; i*=2)
				natural_ring_lbw (i);
			if (_world_rank == 0)
				printf("Link contended test\n");
			for (int i = log2_size; i < iterations; i*=2)
				link_contended_lbw (i);
			if (_world_rank == 0)
				printf("Scatter test\n");
			for (int i = log2_size; i < iterations; i*=2)
				scatter_gather_lbw (0, 0, 1, i);
			if (_world_rank == 0)
				printf("Gather test\n");
			for (int i = log2_size; i < iterations; i*=2)
				scatter_gather_lbw (0, 0, 0, i);
			if (_world_rank == 0)
				printf("All-to-all test\n");
			for (int i = log2_size; i < iterations; i*=2)
				a2a_lbw (0, i);
			break;
		default:
			printf("Enter a number between 1-7\n");
	}

	return 0;
}
