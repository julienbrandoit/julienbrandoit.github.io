#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <omp.h>
#include <cuda_runtime.h>

#define ITER 250 

// ==================== Shared CPU op ====================
double op_cpu(double a, double b) { 
    for (int i = 0; i < ITER; ++i) {
        a = a*a - 0.5*a + 0.1;
        b = b*b - 0.5*b + 0.1;
    }
    return a + b;
}

// High-precision time in milliseconds (double)
double get_time_highres_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec * 1e-6; // ms with microsecond resolution
}

// ==================== CPU scan versions ====================
void naive_scan_On2(double *input, double *output, int n, double (*op)(double, double)) {
    for (int i = 0; i < n; i++) {
        output[i] = input[0];
        for (int j = 1; j <= i; j++) {
            output[i] = op(output[i], input[j]);
        }
    }
}

void naive_scan_On(double *input, double *output, int n, double (*op)(double, double)) {
    output[0] = input[0];
    for (int i = 1; i < n; i++) {
        output[i] = op(output[i - 1], input[i]);
    }
}

void hillis_steele(double *input, double *output, int n, double (*op)(double, double)) {
    double *buffer1 = (double *)malloc(n * sizeof(double));
    double *buffer2 = (double *)malloc(n * sizeof(double));
    memcpy(buffer1, input, n * sizeof(double));

    double *in = buffer1;
    double *out = buffer2;

    for (int d = 1; d < n; d *= 2) {
        #pragma omp parallel for
        for (int i = 0; i < n; i++) {
            if (i >= d)
                out[i] = op(in[i - d], in[i]);
            else
                out[i] = in[i];
        }
        double *tmp = in;
        in = out;
        out = tmp;
    }

    memcpy(output, in, n * sizeof(double));

    free(buffer1);
    free(buffer2);
}

void blelloch_scan_inclusive(double *input, double *output, int n, double (*op)(double, double)) {
    memcpy(output, input, n * sizeof(double));

    for (int d = 1; d < n; d *= 2) {
        #pragma omp parallel for
        for (int i = 0; i < n; i += 2 * d) {
            if (i + 2 * d - 1 < n) {
                output[i + 2 * d - 1] = op(output[i + d - 1], output[i + 2 * d - 1]);
            }
        }
    }

    for (int d = n / 2; d > 1; d /= 2) {
        #pragma omp parallel for
        for (int i = d; i < n; i += d) {
            output[i + d / 2 - 1] = op(output[i - 1], output[i + d / 2 - 1]);
        }
    }
}

// ==================== CUDA Version of Blelloch ====================
__device__ __forceinline__
double op_dev(double a, double b) {
    #pragma unroll 1
    for (int i = 0; i < ITER; i++) {
        a = a*a - 0.5*a + 0.1;
        b = b*b - 0.5*b + 0.1;
    }
    return a + b;
}

__global__ void upsweep_kernel(double* data, int n, int d) {
    int stride = 2 * d;
    long long tid = blockIdx.x * 1LL * blockDim.x + threadIdx.x;
    long long base = tid * 1LL * stride;

    if (base + stride - 1 < n) {
        double a = data[base + d - 1];
        double b = data[base + stride - 1];
        data[base + stride - 1] = op_dev(a, b);
    }
}

__global__ void downsweep_kernel(double* data, int n, int d) {
    int stride = d;
    long long tid = blockIdx.x * 1LL * blockDim.x + threadIdx.x;
    long long i = (tid + 1LL) * stride;

    if (i < n) {
        int left  = int(i - 1);
        int right = int(i + d/2 - 1);
        if (right < n) {
            double a = data[left];
            double b = data[right];
            data[right] = op_dev(a, b);
        }
    }
}

void blelloch_scan_inclusive_cuda(double *input, double *output, int n) {
    double *d_in, *d_out;
    cudaMalloc(&d_in,  n * sizeof(double));
    cudaMalloc(&d_out, n * sizeof(double));

    cudaMemcpy(d_in, input, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_out, input, n * sizeof(double), cudaMemcpyHostToDevice);

    const int block = 256;

    for (int d = 1; d < n; d <<= 1) {
        int numSeg = (n + (2*d) - 1) / (2*d);
        int grid = (numSeg + block - 1) / block;
        upsweep_kernel<<<grid, block>>>(d_out, n, d);
    }

    for (int d = n >> 1; d > 1; d >>= 1) {
        int numI = (n + d - 1) / d;
        int grid = (numI + block - 1) / block;
        downsweep_kernel<<<grid, block>>>(d_out, n, d);
    }

    cudaMemcpy(output, d_out, n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_in);
    cudaFree(d_out);
}

// ==================== Misc helpers ====================
void fill_random(double *array, int n) {
    for (int i = 0; i < n; i++)
        array[i] = (double)rand() / (double)RAND_MAX * 10.0;
}

// ==================== Benchmark harness ====================
int main() {
    int max_threads = omp_get_max_threads();
    printf("Maximum OpenMP threads: %d\n", max_threads);

    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        printf("No CUDA devices found! Running on CPU only.\n");
    } else {
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, dev);
        printf("Running on GPU: %s (Compute Capability %d.%d)\n",
            prop.name, prop.major, prop.minor);
    }

    int N = 25;  // Max power
    int M = 5;   // repetitions
    int On2_limit = 13;

    FILE *fp = fopen("scan_benchmark_runs.csv", "w");
    if (!fp) {
        perror("Failed to open CSV file");
        return 1;
    }
    fprintf(fp, "n,run,On2_time_ms,On_time_ms,HillisSteele_time_ms,Blelloch_time_ms,BlellochGPU_time_ms,correct_HS,correct_Blelloch,correct_BlellochGPU\n");

    for (int i = 5; i <= N; i++) {
        int n = pow(2, i);
        double *input = (double *)malloc(n * sizeof(double));
        double *output_On2 = (double *)malloc(n * sizeof(double));
        double *output_On = (double *)malloc(n * sizeof(double));
        double *output_HS = (double *)malloc(n * sizeof(double));
        double *output_Blelloch = (double *)malloc(n * sizeof(double));
        double *output_BlellochGPU = (double *)malloc(n * sizeof(double));

        for (int m = 1; m <= M; m++) {
            fill_random(input, n);

            double time_On2 = -1.0;
            double start, end;

            if (i <= On2_limit) {
                start = get_time_highres_ms();
                naive_scan_On2(input, output_On2, n, op_cpu);
                end = get_time_highres_ms();
                time_On2 = end - start;
            }

            start = get_time_highres_ms();
            blelloch_scan_inclusive(input, output_Blelloch, n, op_cpu);
            end = get_time_highres_ms();
            double time_Blelloch = end - start;

            start = get_time_highres_ms();
            blelloch_scan_inclusive_cuda(input, output_BlellochGPU, n);
            end = get_time_highres_ms();
            double time_BlellochGPU = end - start;

            start = get_time_highres_ms();
            naive_scan_On(input, output_On, n, op_cpu);
            end = get_time_highres_ms();
            double time_On = end - start;

            start = get_time_highres_ms();
            hillis_steele(input, output_HS, n, op_cpu);
            end = get_time_highres_ms();
            double time_HS = end - start;

            int correct_HS = 1, correct_Blelloch = 1, correct_BlellochGPU = 1;
            for (int j = 0; j < n; j++) {
                if (fabs(output_On[j] - output_HS[j]) > 1e-6) correct_HS = 0;
                if (fabs(output_On[j] - output_Blelloch[j]) > 1e-6) correct_Blelloch = 0;
                if (fabs(output_On[j] - output_BlellochGPU[j]) > 1e-6) correct_BlellochGPU = 0;
            }

            fprintf(fp, "%d,%d,%.9f,%.9f,%.9f,%.9f,%.9f,%d,%d,%d\n",
                    n, m, time_On2, time_On, time_HS, time_Blelloch, time_BlellochGPU,
                    correct_HS, correct_Blelloch, correct_BlellochGPU);
        }

        free(input);
        free(output_On2);
        free(output_On);
        free(output_HS);
        free(output_Blelloch);
        free(output_BlellochGPU);
    }

    fclose(fp);
    return 0;
}
