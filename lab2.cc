#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>

#define MPI_ULL MPI_UNSIGNED_LONG_LONG
#define MPI_CW MPI_COMM_WORLD
#define MPI_SI MPI_STATUS_IGNORE

using ULL = unsigned long long;

void serialCompute(const ULL &r, const ULL &k)
{
	ULL pixels = 0;
    ULL r_squared = r * r;

    for (ULL x = 0; x < r; x++) {
        pixels += ceil(sqrtl(r_squared - x * x));
    }

    printf("%llu\n", (4 * pixels) % k);

    MPI_Finalize();

    exit(0);
}

void parallelCompute(const ULL &r, const ULL &k, int t)
{
    int task_rank;
    long double fp;
    ULL rsq = r * r;
    ULL f, psum = 0, sum;

    MPI_Comm_rank(MPI_CW, &task_rank);

    for (ULL x = task_rank; x < r; x += t) {
        fp = sqrtl(rsq - x * x);
        f = (ULL)fp;
        psum += (f < fp) ? f + 1 : f;
    }
    if (t > 9) psum %= k;

    MPI_Reduce(&psum, &sum, 1, MPI_ULL, MPI_SUM, 0, MPI_CW);

    if (task_rank == 0) {
        printf("%llu\n", (4 * sum) % k);
    }

    MPI_Finalize();

    exit(0);
}

int main(int argc, char** argv)
{
	if (argc != 3) {
		fprintf(stderr, "Must provide exactly 2 arguments!\n");
		return 1;
	}

    int num_tasks;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_CW, &num_tasks);

    if (num_tasks == 1)
        serialCompute(atoll(argv[1]), atoll(argv[2]));
    else
        parallelCompute(atoll(argv[1]), atoll(argv[2]), num_tasks);

    return 0;
}
