
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <jpeglib.h>

#define WIDTH 800*32
#define HEIGHT 800*32
#define MAX_ITERATIONS 10000
#define THRESHOLD_SQUARED 4.0f
#define BLOCK_SIZE 16

__global__ void mandelbrot_kernel(unsigned char *image, 
                                    int width, int height, 
                                    float x_min, float x_max, 
                                    float y_min, float y_max, 
                                    int max_iterations, 
                                    float threshold_squared) {
    // Calculate the pixel coordinates
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;

    // Check if the pixel is within the image bounds
    if (px < width && py < height) {
        // Map pixel coordinates to complex number c
        float real = x_min + (px / (float)width) * (x_max - x_min);
        float imag = y_min + (py / (float)height) * (y_max - y_min);
        
        float z_real = 0.0f;
        float z_imag = 0.0f;
        int iterations = 0;

        // Iterate to determine if the point is in the Mandelbrot set
        while (z_real * z_real + z_imag * z_imag <= threshold_squared && iterations < max_iterations) {
            float temp_real = z_real * z_real - z_imag * z_imag + real;
            z_imag = 2.0f * z_real * z_imag + imag;
            z_real = temp_real;
            iterations++;
        }

        // Determine the color based on the number of iterations
        unsigned char color;
        if (iterations == max_iterations) {
            color = 0; // Black for points likely in the Mandelbrot set
        } else {
            color = (unsigned char)(255 - 255 * iterations / max_iterations); // Gradient from black to white
        }

        // Set the pixel color in the image
        image[py * width + px] = color;
    }
}

void save_jpeg(const char *filename, unsigned char *image, int width, int height) {
    struct jpeg_compress_struct cinfo;
    struct jpeg_error_mgr jerr;
    FILE *outfile;
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_compress(&cinfo);
    if ((outfile = fopen(filename, "wb")) == NULL) {
        fprintf(stderr, "can't open %s\n", filename);
        exit(1);
    }
    jpeg_stdio_dest(&cinfo, outfile);
    cinfo.image_width = width;
    cinfo.image_height = height;
    cinfo.input_components = 1; // Grayscale
    cinfo.in_color_space = JCS_GRAYSCALE;
    jpeg_set_defaults(&cinfo);
    jpeg_set_quality(&cinfo, 100, TRUE); // 100 = best quality, minimal compression
    jpeg_start_compress(&cinfo, TRUE);
    JSAMPROW row_pointer[1];
    while (cinfo.next_scanline < cinfo.image_height) {
        row_pointer[0] = &image[cinfo.next_scanline * width];
        jpeg_write_scanlines(&cinfo, row_pointer, 1);
    }
    jpeg_finish_compress(&cinfo);
    fclose(outfile);
    jpeg_destroy_compress(&cinfo);
}

int main() {
    unsigned char *d_image;
    unsigned char *h_image = (unsigned char *)malloc(WIDTH * HEIGHT);
    
    // Allocate device memory
    cudaMalloc((void **)&d_image, WIDTH * HEIGHT);
    
    // Define the region of the complex plane to visualize
    float x_min = -2.0f, x_max = 2.0f;
    float y_min = -2.0f, y_max = 2.0f;
    
    // Define block and grid sizes
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((WIDTH + BLOCK_SIZE - 1) / BLOCK_SIZE, (HEIGHT + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    // Launch the kernel
    mandelbrot_kernel<<<gridSize, blockSize>>>(d_image, WIDTH, HEIGHT, x_min, x_max, y_min, y_max, MAX_ITERATIONS, THRESHOLD_SQUARED);
    
    // Copy the result back to host
    cudaMemcpy(h_image, d_image, WIDTH * HEIGHT, cudaMemcpyDeviceToHost);
    
    // Save the image as a JPEG file
    save_jpeg("mandelbrot_cuda.jpg", h_image, WIDTH, HEIGHT);
    
    // Free device memory
    cudaFree(d_image);
    free(h_image);
    
    return 0;
}