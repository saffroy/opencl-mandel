// -*- c -*-

#include "mandel.h"

__kernel void mandel_iters(int max_iters,
                           int max_loops,
                           __global struct point_state *statevec)
{
        __global struct point_state *state = statevec + get_global_id(0);

        if (state->escaped
            || state->iters >= max_iters)
                return;

        FLOAT x0 = state->x0;
        FLOAT y0 = state->y0;
        FLOAT x = state->x;
        FLOAT y = state->y;
        int iters = state->iters;

        FLOAT x2 = x * x;
        FLOAT y2 = y * y;

        int n = 0;

        while ((x2 + y2 < 4.0)
               && (n < max_loops)
               && (iters < max_iters))
        {
                y = 2 * x * y + y0;
                x = x2 - y2 + x0;

                n++;
                iters++;

                x2 = x * x;
                y2 = y * y;
        }

        state->x = x;
        state->y = y;
        state->iters = iters;
        state->escaped = (x2 + y2 >= 4.0);
}
