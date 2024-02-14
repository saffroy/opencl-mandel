#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include <CL/cl.h>

#ifndef FLOAT
#define FLOAT double // Note: must match .cl file
#endif

#define XSTEPS 800
#define YSTEPS 600

#define XMIN -2.5
#define YMIN -1.5
#define XRANGE 4.0
#define YRANGE 3.0
#define DX (XRANGE / (FLOAT)XSTEPS)
#define DY (YRANGE / (FLOAT)YSTEPS)

#define MAXITER 256

static cl_device_id
find_device() {
        cl_platform_id platform;
        cl_uint num_platforms;
        cl_int rc;

        rc = clGetPlatformIDs(1, &platform, &num_platforms);
        assert(rc == CL_SUCCESS);
        assert(num_platforms > 0);

        cl_device_id device;
        cl_uint num_devices;
        rc = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU,
                            1, &device, &num_devices);
        assert(rc == CL_SUCCESS);
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
        assert(rc == CL_SUCCESS);
        free(prog);

        rc = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
        assert(rc == CL_SUCCESS);

        //XXX TODO: log build errors

        return program;
}

int main() {
#define SIZE_REALS (sizeof(*reals) * XSTEPS)
#define SIZE_IMAGS (sizeof(*imags) * YSTEPS)
#define SIZE_ITERS (sizeof(*iters) * XSTEPS * YSTEPS)

        FLOAT *reals = calloc(SIZE_REALS, 1);
        FLOAT *imags = calloc(SIZE_IMAGS, 1);
        int *iters = calloc(SIZE_ITERS, 1);

        assert(reals && imags && iters);

        for (int i = 0; i < XSTEPS; i++)
                reals[i] = XMIN + (FLOAT)i * DX;
        for (int i = 0; i < YSTEPS; i++)
                imags[i] = YMIN + (FLOAT)i * DY;

        cl_device_id device = find_device();
        cl_int rc;
        cl_context ctx = clCreateContext(NULL, 1, &device, NULL, NULL, &rc);

        cl_command_queue cq = clCreateCommandQueue(ctx, device, 0, &rc);
        assert(rc == CL_SUCCESS);

        // device memory buffers
        cl_mem reals_d = clCreateBuffer(ctx, CL_MEM_READ_ONLY,
                                        SIZE_REALS, NULL, NULL);
        cl_mem imags_d = clCreateBuffer(ctx, CL_MEM_READ_ONLY,
                                        SIZE_IMAGS, NULL, NULL);
        cl_mem iters_d = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
                                        SIZE_ITERS, NULL, NULL);

        rc = clEnqueueWriteBuffer(cq, reals_d, CL_TRUE, 0,
                                  SIZE_REALS, reals, 0, NULL, NULL);
        assert(rc == CL_SUCCESS);
        rc = clEnqueueWriteBuffer(cq, imags_d, CL_TRUE, 0,
                                  SIZE_IMAGS, imags, 0, NULL, NULL);
        assert(rc == CL_SUCCESS);
        
        // kernel
        cl_program prog = build_prog(ctx, device);
        cl_kernel kernel = clCreateKernel(prog, "mandel_iters", &rc);
        assert(rc == CL_SUCCESS);

        int arg_xsteps = XSTEPS;
        int arg_maxiters = MAXITER;

        rc = clSetKernelArg(kernel, 0, sizeof(arg_xsteps), &arg_xsteps);
        assert(rc == CL_SUCCESS);
        rc = clSetKernelArg(kernel, 1, sizeof(arg_maxiters), &arg_maxiters);
        assert(rc == CL_SUCCESS);
        rc = clSetKernelArg(kernel, 2, sizeof(cl_mem), &reals_d); 
        assert(rc == CL_SUCCESS);
        rc = clSetKernelArg(kernel, 3, sizeof(cl_mem), &imags_d); 
        assert(rc == CL_SUCCESS);
        rc = clSetKernelArg(kernel, 4, sizeof(cl_mem), &iters_d);
        assert(rc == CL_SUCCESS);

        // enqueue job
        size_t local_size = 256; // max work group size (from clinfo)
        size_t global_size = XSTEPS * YSTEPS;
        rc = clEnqueueNDRangeKernel(cq, kernel, 1, NULL,
                                    &global_size, &local_size, 0, NULL, NULL);
        assert(rc == CL_SUCCESS);

        // wait for completion
        rc = clFinish(cq);
        assert(rc == CL_SUCCESS);

        // read back results
        rc = clEnqueueReadBuffer(cq, iters_d, CL_TRUE, 0,
                                 SIZE_ITERS, iters, 0, NULL, NULL);
        assert(rc == CL_SUCCESS);

        // release resources
        rc = clReleaseKernel(kernel);
        assert(rc == CL_SUCCESS);
        rc = clReleaseProgram(prog);
        assert(rc == CL_SUCCESS);
        rc = clReleaseMemObject(reals_d);
        assert(rc == CL_SUCCESS);
        rc = clReleaseMemObject(imags_d);
        assert(rc == CL_SUCCESS);
        rc = clReleaseMemObject(iters_d);
        assert(rc == CL_SUCCESS);
        rc = clReleaseCommandQueue(cq);
        assert(rc == CL_SUCCESS);
        rc = clReleaseContext(ctx);
        assert(rc == CL_SUCCESS);

        // write image
        FILE *out = fopen("mandel.ppm", "w");
        assert(out);
        fprintf(out, "P3\n%d %d\n255\n", XSTEPS, YSTEPS);
        for (int i = 0; i < XSTEPS*YSTEPS; i++) {
                int it = iters[i];
                int c = it == MAXITER ? 0 : it % 256;
                fprintf(out, "%d %d %d\n", c, c, c);
        }
        fclose(out);

        free(iters);
        free(reals);
        free(imags);

        return 0;
}
