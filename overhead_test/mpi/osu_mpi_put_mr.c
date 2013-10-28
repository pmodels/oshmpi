/*
 * Copyright (C) 2002-2013 the Network-Based Computing Laboratory
 * (NBCL), The Ohio State University. 
 *
 * Contact: Dr. D. K. Panda (panda@cse.ohio-state.edu)
 *
 * For detailed copyright and licensing information, please refer to the
 * copyright file COPYRIGHT in the top level OMB directory.
 */
/*
 * Notes: Sayan: This program was modified to convert this to 
 * MPI-3 one-sided message rate test
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <mpi.h>

#include "osu_common.h"

#define MPI_THREAD_STRING(level)  \
	( level==MPI_THREAD_MULTIPLE ? "THREAD_MULTIPLE" : \
	  ( level==MPI_THREAD_SERIALIZED ? "THREAD_SERIALIZED" : \
	    ( level==MPI_THREAD_FUNNELED ? "THREAD_FUNNELED" : \
	      ( "THREAD_SINGLE" ) ) ) )

#define MPI_INIT_NOT_FINAL(flag)        MPI_Initialized(flag) && !MPI_Finalized(flag) 

#define ITERS_SMALL     (500)          
#define ITERS_LARGE     (50)
#define LARGE_THRESHOLD (8192)
#define MAX_MSG_SZ (1<<22)

#ifndef FIELD_WIDTH
#   define FIELD_WIDTH 20
#endif

#ifndef FLOAT_PRECISION
#   define FLOAT_PRECISION 2
#endif

#define HEADER  "MPI-3 One-sided Put Message rate (adapted from OSU OpenSHMEM benchmark)\n"

struct pe_vars {
	int me;
	int npes;
	int pairs;
	int nxtpe;
	MPI_Win win;
};

void init_mpi (struct pe_vars * v)
{
	int mpi_provided;

	MPI_Init_thread( NULL, NULL, MPI_THREAD_SERIALIZED, &mpi_provided );
	MPI_Query_thread(&mpi_provided);

	if (strcmp((const char *)MPI_THREAD_STRING(mpi_provided),"WTF") == 0)
		MPI_Abort (MPI_COMM_WORLD, 5);

	MPI_Comm_rank( MPI_COMM_WORLD, &(v->me) );
	MPI_Comm_size( MPI_COMM_WORLD, &(v->npes) );


	v->pairs = v->npes / 2;
	v->nxtpe = ( v->me < v->pairs ) ? ( v->me + v->pairs ) : ( v->me - v->pairs );

	return;
}

void check_usage (int argc, char * argv [], int npes, int me)
{
	if (argc > 2) {
		fprintf(stderr, "No need to pass anything...");
		exit(EXIT_FAILURE);
	}

	if (2 > npes) {
		if (0 == me) {
			fprintf(stderr, "This test requires at least two processes\n");
		}

		exit(EXIT_FAILURE);
	}

	return;
}

void print_header (int me)
{
	if(me == 0) {
		fprintf(stdout, HEADER);
		fprintf(stdout, "%-*s%*s\n", 10, "# Size", FIELD_WIDTH, "Messages/s");
		fflush(stdout);
	}
}

char * allocate_memory (int me, MPI_Win * win)
{
	char * msg_buffer; 
	char * win_base ; /* base */
	
	MPI_Info info;
	MPI_Info_create(&info);
	MPI_Info_set(info, "same_size", "true");
	
	MPI_Alloc_mem((MAX_MSG_SZ * ITERS_LARGE), info, &msg_buffer);
	MPI_Win_allocate((MAX_MSG_SZ * ITERS_LARGE) * sizeof(char), sizeof(char), MPI_INFO_NULL, MPI_COMM_WORLD, &win_base, win);
	MPI_Win_lock_all (MPI_MODE_NOCHECK, *win);

        MPI_Info_free(&info);

	if (NULL == msg_buffer && MPI_BOTTOM == win_base) {
		fprintf(stderr, "Failed to allocate window (pe: %d)\n", me);
		exit(EXIT_FAILURE);
	}

	return msg_buffer;
}

double message_rate (char * buffer, int size, int iterations, int me, int pairs, int nxtpe, MPI_Win win)
{
	int64_t begin, end; 
	int i, offset;

	/*
	 * Touch memory
	 */
	memset(buffer, size, MAX_MSG_SZ * ITERS_LARGE);

	MPI_Barrier(MPI_COMM_WORLD);
	
	if (me < pairs) {
		begin = TIME();

		for (i = 0, offset = 0; i < iterations; i++, offset++) {
			MPI_Put ((buffer + offset*size), size, MPI_CHAR, nxtpe, offset*size, size, MPI_CHAR, win);
			//MPI_Win_flush_local (nxtpe, win);
		}
		//MPI_Win_flush_all(win);
		MPI_Win_flush(nxtpe, win);
		end = TIME();

		return ((double)iterations * 1e6) / ((double)end - (double)begin);
	}
	return 0;
}

void print_message_rate (int size, double rate, int me)
{
	if (me == 0) { 
		fprintf(stdout, "%-*d%*.*f\n", 10, size, FIELD_WIDTH, FLOAT_PRECISION,
				rate);
		fflush(stdout);
	}
}

void benchmark (char * msg_buffer, int me, int pairs, int nxtpe, MPI_Win win)
{
	static double mr, mr_sum;
	int iters;
	
	if (msg_buffer == NULL) {
		printf("Input buffer is NULL, no reason to proceed\n");
		exit(-1);
	}
	/*
	 * Warmup
	 */
	if (me < pairs) {
		for (int i = 0; i < ITERS_LARGE; i += 1) {
			MPI_Put ((msg_buffer + i*MAX_MSG_SZ), MAX_MSG_SZ, MPI_CHAR, nxtpe, i*MAX_MSG_SZ, MAX_MSG_SZ, MPI_CHAR, win);
			MPI_Win_flush_local (nxtpe, win);
		}
	}

	MPI_Win_flush_all(win);
	MPI_Barrier(MPI_COMM_WORLD);
	/*
	 * Benchmark
	 */
	for (long size = 1; size <= MAX_MSG_SZ; size <<= 1) {
        	iters = size < LARGE_THRESHOLD ? ITERS_SMALL : ITERS_LARGE;
		mr = message_rate(msg_buffer, size, iters, me, pairs, nxtpe, win);
		MPI_Reduce(&mr, &mr_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		print_message_rate(size, mr_sum, me);
	}
}

int main (int argc, char *argv[])
{
        struct pe_vars v;
	char * msg_buffer;

	/*
	 * Initialize
	 */
	init_mpi(&v);
	check_usage(argc, argv, v.npes, v.me);
	print_header(v.me);

	if (v.me == 0) printf("Total processes = %d\n",v.npes);
	/*
	 * Allocate Memory
	 */
	msg_buffer = allocate_memory(v.me, &(v.win) );
	memset(msg_buffer, 0, MAX_MSG_SZ * ITERS_LARGE);
	/*
	 * Time Put Message Rate
	 */
	benchmark(msg_buffer, v.me, v.pairs, v.nxtpe, v.win);
	/*
	 * Finalize
	 */
	MPI_Win_unlock_all(v.win);
	MPI_Win_free(&v.win); 	
	MPI_Free_mem(msg_buffer);

	MPI_Finalize();

	return EXIT_SUCCESS;
}
