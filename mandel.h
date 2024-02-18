#ifndef _MANDEL_H
#define _MANDEL_H

// shared defs between .c and .cl files

#ifndef FLOAT
#define FLOAT double
#endif

struct point_state {
        FLOAT x0;
        FLOAT y0;
        FLOAT x;
        FLOAT y;
        int iters;
        char escaped;
};

#endif // _MANDEL_H
