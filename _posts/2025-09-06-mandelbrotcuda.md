---
title: Mandelbrot set with CUDA
description: A simple implementation of the Mandelbrot set using CUDA for parallel computation.
date: 2025-09-06 00:00:00 +0000
categories: [Blog post, short]
tags: [parallel computing, cuda, mandelbrot]
math: true
---

In this short blog post, we explore a simple implementation that renders a static image of the Mandelbrot set using CUDA for parallel computation. The Mandelbrot set is a famous fractal defined by an interative rule applied to complex numbers.

## The math behind the Mandelbrot set

The Mandelbrot set is defined as the set of the complex numbers $c \in \mathbb{C}$ for which the sequence defined by the iteration
$$\begin{equation} \begin{cases} z_{n+1} = z_n^2 + c, \\ z_0 = 0 \end{cases} \end{equation}$$
remains bounded. The sequence is bounded if the magnitude of $z_n$ does not blow up to infinity.

## Numerical implementation

The objective of the implementation is to build an image where each pixel corresponds to a point in the complex plane. The color of each pixel is determined by how quickly the sequence diverges for that point. We will use the following rules for coloring.


If the sequence does not diverge after a maximum number of iterations, the pixel is colored black; *It is likely that the point belongs to the Mandelbrot set*, at least within the limits of our computation.


If the sequence diverges, the pixel is colored based on the number of iterations it took to diverge, creating a gradient effect. We will use a simple linear gradient from black (max iterations) to white (immediate divergence), such that the color is given by:

$$\text{color} = 255 \times \frac{\text{iterations to diverge}}{\text{max_iterations}},$$

where 0 is black and 255 is white.

We define divergence through a *threshold* value. If the magnitude of $z_n$ exceeds this threshold, we consider the sequence to have diverged.

### The threshold barrier

There exist a few mathematical results that help us choose a suitable threshold. **It can be shown that if the magnitude of $z_n$ exceeds 2, the sequence will diverge to infinity.**[^1]

[^1]: Munafo, Robert P. “Escape Radius.” *Mu-Ency*, 14 June 2023, [https://mrob.com/pub/muency/escaperadius.html](https://mrob.com/pub/muency/escaperadius.html).

### The objective

Given a rectangular region of the complex plane defined by its corners $(x_{min}, y_{min})$ and $(x_{max}, y_{max})$, we want to compute the color of each pixel in an image of size $(W, H)$.

A pixel at position $(p_x, p_y)$ in the image corresponds to the complex number:

$$
\begin{equation}
c = \underbrace{x_{min} + \frac{p_x}{W} (x_{max} - x_{min})}_{\text{real part}} + i \underbrace{\left(y_{min} + \frac{p_y}{H} (y_{max} - y_{min})\right)}_{\text{imaginary part}}.
\end{equation}
$$

The magnitude of such a complex number $c = a + bi$ is given by:
$$
\begin{align}
|c| &= \sqrt{a^2 + b^2},\\
&= \sqrt{\left(x_{min} + \frac{p_x}{W} (x_{max} - x_{min})\right)^2 + \left(y_{min} + \frac{p_y}{H} (y_{max} - y_{min})\right)^2},
\end{align}
$$

however, since we only need to compare the magnitude against a threshold, we can avoid the computational cost of the square root by comparing the squared magnitude against the squared threshold $T^2$. We use $T^2 = 4$ in our implementation.

We want to output a jpeg image of size $(W, H)$, where each pixel is represented by a single byte (grayscale). We will code this in `C`, using `CUDA` for parallel computation.

### CUDA implementation

The first thing to understand is: *what is a CUDA kernel?* A CUDA kernel is a function that runs on the GPU and is executed by many threads in parallel. This is very useful for problems that can be broken down into smaller, independent tasks, such as computing the color of each pixel in our Mandelbrot set image. Moreover, the computation for each pixel is independent of the others, making it an ideal candidate for parallelization, where each thread can handle the computation for a single pixel.

The way to think about a CUDA kernel is that it is a shared function that will be executed by many threads in parallel, working on different parts of the data. **Each thread has its own unique identifier, which it can use to determine which part of the data it should work on.**

The way CUDA organizes threads is through a grid of blocks, where each block contains a certain number of threads. The total number of threads is determined by the product of the number of blocks and the number of threads per block.

Memory management is also crucial in CUDA programming. There are different types of memory (global, shared, local), each with its own performance characteristics. In our case, we will primarily use global memory to store the image data, as it needs to be accessible by all threads. This memory contains the image array.

In our case, we want a 2D grid of blocks, where each block contains a 2D array of threads. Each thread will compute the color of a single pixel in the image.


Here is the CUDA kernel that computes the Mandelbrot set:

```c
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
```

This kernel is what each thread will execute. It calculates the pixel coordinates based on the block and thread indices, maps those coordinates to a complex number, iterates to determine if the point is in the Mandelbrot set, and finally sets the pixel color in the image.

We now need to set up the host code (that is, the code that runs on the CPU) to allocate memory, launch the kernel, and handle the image output. Here is a simple implementation:

```c
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
                                    float threshold_squared); // WE ALREADY DEFINED THIS ABOVE


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
```

This host code allocates memory for the image on both the host and device, defines the region of the complex plane to visualize, sets up the block and grid sizes, launches the kernel, copies the result back to the host, and saves the image as a JPEG file. We use the `libjpeg` library to handle JPEG compression.

We can compile this code using `nvcc`, the NVIDIA CUDA compiler, and link against the `libjpeg` library. Here is an example compilation command:

```bash
nvcc -o mandelbrot mandelbrotdrawer.cu -ljpeg
```

After compiling, we can run the program, and it will generate a file named `mandelbrot_cuda.jpg` containing the rendered Mandelbrot set.

![Here is the resulting image.](assets/mandelbrotcuda/mandelbrot_cuda.jpg)
***Figure 1: Mandelbrot set rendered with CUDA.** The maximum number of iterations is 10000, and the image size is 25600x25600 pixels.*

You can experiment with different parameters, such as the region of the complex plane, the image size, and the maximum number of iterations, to explore different aspects and approximations of the Mandelbrot set. Enjoy exploring the fascinating world of fractals with CUDA!

## Appendix : code
The code can be found on GitHub: [mandelbrot_cuda.cu](https://github.com/julienbrandoit/julienbrandoit.github.io/tree/main/assets/mandelbrotcuda/mandelbrotdrawer.cu)

### footnote
