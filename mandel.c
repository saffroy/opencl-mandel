#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include <CL/cl.h>

#include "mandel.h"

#define XSTEPS 1024
#define YSTEPS 1024

#define XMIN -2.0
#define YMIN -1.5
#define XRANGE 3.0
#define YRANGE 3.0
#define DX (XRANGE / (FLOAT)XSTEPS)
#define DY (YRANGE / (FLOAT)YSTEPS)

#define MAXLOOP 256*10
#define MAXITER 256*100

#define ASSERT_CL_SUCCESS(_rc)                                          \
        do {                                                            \
                if (_rc != CL_SUCCESS) {                                \
                        fprintf(stderr, "opencl error: %d\n", _rc);     \
                }                                                       \
                assert(_rc == CL_SUCCESS);                              \
        } while (0)

static cl_device_id
find_device() {
        cl_platform_id platform;
        cl_uint num_platforms;
        cl_int rc;

        rc = clGetPlatformIDs(1, &platform, &num_platforms);
        ASSERT_CL_SUCCESS(rc);
        assert(num_platforms > 0);

        cl_device_id device;
        cl_uint num_devices;
        rc = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU,
                            1, &device, &num_devices);
        ASSERT_CL_SUCCESS(rc);
        assert(num_devices > 0);

        return device;
}

static char*
load_prog() {
        FILE *f = fopen("mandel.cl", "r");
        assert(f);

        fseek(f, 0, SEEK_END);
        int length = ftell(f);
        assert(length > 0);
        fseek(f, 0, SEEK_SET);

        char *prog = malloc(length+1);
        assert(prog);
        int rc = fread(prog, 1, length, f);
        assert(rc == length);
        prog[length] = 0; // NUL char

        fclose(f);
        return prog;
}

static cl_program
build_prog(cl_context ctx, cl_device_id device) {
        char *prog = load_prog();
        size_t prog_len = strlen(prog);

        cl_int rc;
        cl_program program =
                clCreateProgramWithSource(ctx, 1, &prog, &prog_len, &rc);
        ASSERT_CL_SUCCESS(rc);
        free(prog);

        rc = clBuildProgram(program, 0, NULL, "-I.", NULL, NULL);
        if (rc != CL_SUCCESS) {
                fprintf(stderr, "clBuildProgram error: %d\n", rc);

                size_t build_log_len;
                rc = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                                           0, NULL, &build_log_len);
                ASSERT_CL_SUCCESS(rc);

                char *buff_erro = malloc(build_log_len);
                assert(buff_erro);

                rc = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                                           build_log_len, buff_erro, NULL);
                ASSERT_CL_SUCCESS(rc);

                fprintf(stderr,"Build log: \n%s\n", buff_erro);
                exit(1);
        }

        return program;
}

int main() {
#define SIZE_STATE (sizeof(*statevec) * XSTEPS * YSTEPS)
        struct point_state *statevec = malloc(SIZE_STATE);
        assert(statevec);

        for (int j = 0; j < YSTEPS; j++) {
                for (int i = 0; i < XSTEPS; i++) {
                        struct point_state *state = statevec + j*XSTEPS + i;
                        state->x0 = XMIN + (FLOAT)i * DX;
                        state->y0 = YMIN + (FLOAT)j * DY;
                        state->x = state->x0;
                        state->y = state->y0;
                        state->iters = 0;
                        state->escaped = 0;
                }
        }

        cl_device_id device = find_device();
        cl_int rc;
        cl_context ctx = clCreateContext(NULL, 1, &device, NULL, NULL, &rc);

        cl_command_queue cq = clCreateCommandQueue(ctx, device, 0, &rc);
        ASSERT_CL_SUCCESS(rc);

        // device memory buffers
        cl_mem state_d = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
                                        SIZE_STATE, NULL, &rc);
        ASSERT_CL_SUCCESS(rc);

        rc = clEnqueueWriteBuffer(cq, state_d, CL_TRUE, 0,
                                  SIZE_STATE, statevec, 0, NULL, NULL);
        ASSERT_CL_SUCCESS(rc);
        
        // kernel
        cl_program prog = build_prog(ctx, device);
        cl_kernel kernel = clCreateKernel(prog, "mandel_iters", &rc);
        ASSERT_CL_SUCCESS(rc);

        int arg_maxiters = MAXITER;
        int arg_maxloops = MAXLOOP;

        rc = clSetKernelArg(kernel, 0, sizeof(arg_maxiters), &arg_maxiters);
        ASSERT_CL_SUCCESS(rc);
        rc = clSetKernelArg(kernel, 1, sizeof(arg_maxloops), &arg_maxloops);
        ASSERT_CL_SUCCESS(rc);
        rc = clSetKernelArg(kernel, 2, sizeof(cl_mem), &state_d); 
        ASSERT_CL_SUCCESS(rc);

        // enqueue jobs
        size_t local_size = 32; // seems to give best performance, no clue why
        size_t global_size = XSTEPS * YSTEPS;
        for (int i = 0; i < MAXITER; i+= MAXLOOP)
        {
                rc = clEnqueueNDRangeKernel(cq, kernel, 1, NULL,
                                            &global_size, &local_size, 0, NULL, NULL);
                ASSERT_CL_SUCCESS(rc);
        }

        // wait for completion
        rc = clFinish(cq);
        ASSERT_CL_SUCCESS(rc);

        // read back results
        rc = clEnqueueReadBuffer(cq, state_d, CL_TRUE, 0,
                                 SIZE_STATE, statevec, 0, NULL, NULL);
        ASSERT_CL_SUCCESS(rc);

        // release resources
        rc = clReleaseKernel(kernel);
        ASSERT_CL_SUCCESS(rc);
        rc = clReleaseProgram(prog);
        ASSERT_CL_SUCCESS(rc);
        rc = clReleaseMemObject(state_d);
        ASSERT_CL_SUCCESS(rc);
        rc = clReleaseCommandQueue(cq);
        ASSERT_CL_SUCCESS(rc);
        rc = clReleaseContext(ctx);
        ASSERT_CL_SUCCESS(rc);

        // write image
        FILE *out = fopen("mandel.ppm", "w");
        assert(out);
        fprintf(out, "P3\n%d %d\n255\n", XSTEPS, YSTEPS);
        for (int j = 0; j < YSTEPS; j++) {
                for (int i = 0; i < XSTEPS; i++) {
                        struct point_state *state = statevec + j*XSTEPS + i;
                        int c = (state->iters == MAXITER
                                 ? 0 : state->iters % 256);
                        fprintf(out, "%d %d %d\n", c, c, c);
                }
        }
        fclose(out);

        free(statevec);

        return 0;
}
