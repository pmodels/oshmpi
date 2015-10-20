#include <stdio.h>
#include <stdlib.h>

#include <mpi.h>

int main(int argc, char* argv[])
{
    MPI_Init(&argc,&argv);

    MPI_Aint bytes = (argc>1) ? atol(argv[1]) : 128*1024*1024;
    printf("bytes = %zu\n", bytes);

    MPI_Comm comm_shared = MPI_COMM_NULL;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0 /* key */, MPI_INFO_NULL, &comm_shared);

    MPI_Info info_win = MPI_INFO_NULL;
    MPI_Info_create(&info_win);
    MPI_Info_set(info_win, "alloc_shared_noncontig", "true");

    MPI_Win win_shared = MPI_WIN_NULL;
    void * base_ptr = NULL;
    int rc = MPI_Win_allocate_shared(bytes, 1 /* disp_unit */, info_win, comm_shared, &base_ptr, &win_shared);

    memset(base_ptr,255,bytes);

    MPI_Info_free(&info_win);

    MPI_Comm_free(&comm_shared);
    MPI_Finalize();
    return 0;
}
