#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>
#include <time.h>
#include <assert.h>

#include <shmem.h>

#define ALIGN 16

/* Global variables */
typedef enum xfer_t_e
{
  GET = 1,
  PUT = 2
}
xfer_t;

int _world_rank, _world_size;
/* global counter, perform fadd against this var */
int gcounter = 0;

long pSync0[_SHMEM_BARRIER_SYNC_SIZE], pSync1[_SHMEM_BARRIER_SYNC_SIZE];
double pWrk0[_SHMEM_REDUCE_MIN_WRKDATA_SIZE],
  pWrk1[_SHMEM_REDUCE_MIN_WRKDATA_SIZE];

double total_clock_time, max_clock_time, min_clock_time, clock_time_PE;

/* Utilities */
/*
    double
shmem_wtime (void)
{
  double wtime;
  struct timeval tv;
  gettimeofday (&tv, NULL);
  wtime = tv.tv_sec;
  wtime += (double) tv.tv_usec / 1000000.0;

  return wtime;
}
*/
void
_transfer (void *A, int ilo, int jlo, int ihigh, int jhigh,
	   void *my_A, int chunksize, int ga_dim, int type_size,
	   int rank, xfer_t xtype)
{
  int icur, count, jlast;
  int disp;

  icur = ilo;
  jlast = (_world_rank + 1) * chunksize - 1;
  if (jlast > jhigh)
    jlast = jhigh;
  count = jlast - jlo + 1;

  while (icur <= jhigh * ga_dim)
    {
#if DEBUG>4
      printf ("jlo = %d, jhigh = %d\n", jlo, jhigh);
      printf ("count = %d\n", count);
#endif
      disp = icur * ga_dim * count;
#if DEBUG>2
      printf ("disp = %d\n", disp);
#endif
      switch (xtype)
	{
	case PUT:
	  shmem_putmem ((A + disp), my_A, count * type_size, rank);
	case GET:
	  shmem_getmem (my_A, (A + disp), count * type_size, rank);
	}
      /* resize */
      my_A = my_A + count * type_size;
      icur += ga_dim;
    }
  return;
}

// ga_XYdim must be perfectly 
// divisible by npes
void
matvec (unsigned int ga_XYdim, int root)
{
  /* npes needs to be perfect multiple of ga_size */
  assert (ga_XYdim % _world_size == 0);

  int count_p = 0, next_p = 0;
  int hi[2], lo[2];

  int global_Asize = ga_XYdim * ga_XYdim * sizeof (double);
  int global_bsize = ga_XYdim * sizeof (double);
  int local_dim_1d = ga_XYdim / _world_size;
  long local_Asize = local_dim_1d * local_dim_1d * sizeof (double);
  long local_bsize = local_dim_1d * sizeof (double);

  double *my_A = (double *) shmemalign (ALIGN, local_Asize);
  double *my_b = (double *) shmemalign (ALIGN, local_bsize);
  double *my_Ab = (double *) shmemalign (ALIGN, local_bsize);

  double *A = (double *) shmemalign (ALIGN, global_Asize);
  double *b = (double *) shmemalign (ALIGN, global_bsize);
  double *result = (double *) shmemalign (ALIGN, global_bsize);
  /* Initialize GA for rank 0 (root) only, others 
   * will get/put 
   */
  if (_world_rank == root)
    {
      /* fill with random data */
      //srand (time (NULL));
      for (int i = 0; i < ga_XYdim; i++)
	for (int j = 0; j < ga_XYdim; j++)
	  A[i * ga_XYdim + j] = j + 5;
      //my_A[i * local_dim_1d + j] = rand () * 0.001;
      for (int j = 0; j < ga_XYdim; j++)
	{
	  b[j] = j + 2;
	  result[j] = -1.0;
	}
      if (ga_XYdim <= 10)
	{
	  printf ("Matrix A:\n");
	  for (int i = 0; i < ga_XYdim; i++)
	    {
	      for (int j = 0; j < ga_XYdim; j++)
		{
		  printf ("%f    ", A[i * ga_XYdim + j]);
		}
	      printf ("\n");
	    }
	}
    }

  shmem_barrier_all ();

  /* Get chunks from global array [which is in root] 
   * to local buffers...each chunk is 
   * local_dim_1d x local_dim_1d
   */
  next_p = shmem_int_fadd (&gcounter, 1, root);
  for (int i = 0; i < ga_XYdim; i += local_dim_1d)
    {
      if (next_p == count_p)
	{
	  // b vector
	  shmem_getmem (my_b, b + i, local_dim_1d * sizeof (double), root);
	  for (int j = 0; j < ga_XYdim; j += local_dim_1d)
	    {
	      /* Indices of patch A */
	      lo[0] = i;
	      lo[1] = j;
	      hi[0] = lo[0] + local_dim_1d;
	      hi[1] = lo[1] + local_dim_1d;

	      hi[0] = hi[0] - 1;
	      hi[1] = hi[1] - 1;
#if DEBUG>1
	      printf ("%d: GET: lo[0,1] = %d,%d and hi[0,1] = %d,%d\n",
		      _world_rank, lo[0], lo[1], hi[0], hi[1]);
#endif
	      // noncontiguous gets from global array A
	      // populate row by row of my_A
	      _transfer (A, lo[0], lo[1], hi[0], hi[1], my_A, local_dim_1d,
			 ga_XYdim, sizeof (double), root, GET);
	      // probably we need an equivalent MPI_Win_flush here...
	      /* local matrix vector computation */
	      for (int m = 0; m < local_dim_1d; m++)
		{
		  my_Ab[m] = 0.0;
		  for (int n = 0; n < local_dim_1d; n++)
		    {
		      my_Ab[m] += my_A[m * local_dim_1d + n] * my_b[n];
		    }
		}
	    }
	  // reduction across rows
	  shmem_double_sum_to_all (result, my_Ab, local_dim_1d, 0,
				   0, _world_size, pWrk1, pSync1);
	  next_p = shmem_int_fadd (&gcounter, 1, root);
	}
      count_p++;
    }

  shmem_barrier_all ();


#ifdef TEST
  if (ga_XYdim <= 10)
    {
      for (int l = 0; l < _world_size; l++)
	{
	  shmem_barrier_all ();
	  if (l == _world_rank)
	    {
	      printf ("[%d]Snapshot of Matrix A:\n", _world_rank);
	      for (int i = 0; i < local_dim_1d; i++)
		{
		  for (int j = 0; j < local_dim_1d; j++)
		    {
		      printf ("%f    ", my_A[i * local_dim_1d + j]);
		    }
		  printf ("\n");
		}
	      printf ("[%d]Vector b:\n", _world_rank);
	      for (int j = 0; j < local_dim_1d; j++)
		{
		  printf ("%f\n", my_b[j]);
		}
	      printf ("[%d]Result:\n", _world_rank);
	      for (int j = 0; j < local_dim_1d; j++)
		{
		  printf ("%f\n", result[j]);
		}
	    }
	}
    }
#endif
  shmem_barrier_all ();
  shfree (my_A);
  shfree (my_b);
  shfree (my_Ab);
  shfree (A);
  shfree (b);
  shfree (result);
  return;
}

int
main (int argc, char *argv[])
{
  double time_start, time_end;
  int N, root;
  if (argc != 3)
    {
      printf ("./matvec.exe <dimension(square)> <root-rank>");
      exit (-1);
    }
  else
    {
      N = atoi (argv[1]);
      root = atoi (argv[2]);
    }
  /*OpenSHMEM initilization */
  start_pes (0);
  _world_size = _num_pes ();
  _world_rank = _my_pe ();
  /* wait for user to input runtime params */
  for (int j = 0; j < _SHMEM_BARRIER_SYNC_SIZE; j++)
    {
      pSync0[j] = pSync1[j] = _SHMEM_SYNC_VALUE;
    }

  time_start = shmem_wtime ();
  matvec (N, root);
  time_end = shmem_wtime ();
  clock_time_PE = time_end - time_start;
  shmem_double_sum_to_all (&total_clock_time, &clock_time_PE, 1,
			   0, 0, _world_size, pWrk0, pSync0);
  if (_world_rank == 0)
    {
      printf ("Avg. time taken for Matrix-vector multiplication : %f \n",
	      (total_clock_time / _world_size));
    }

  return 0;
}
