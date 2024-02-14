// -*- c -*-

#ifndef FLOAT
#define FLOAT double // Note: must match .c file
#endif

__kernel void mandel_iters(int xsteps,
                           int max_iters,
                           __global FLOAT const *real,
                           __global FLOAT const *imag,
                           __global int *iters)
{
        unsigned int rank = get_global_id(0);

        int j = rank / xsteps;
        int i = rank % xsteps;

        FLOAT x0 = real[i];
        FLOAT y0 = imag[j];

        FLOAT x = x0;
        FLOAT y = y0;

        FLOAT x2 = x * x;
        FLOAT y2 = y * y;

        int   n = 0;
        while ((x2 + y2 < 4.0)
               && (n < max_iters))
        {
                y = 2 * x * y + y0;
                x = x2 - y2 + x0;

                n++;

                x2 = x * x;
                y2 = y * y;
        }

        iters[rank] = n;
}
