# Understanding the scan operation

This folder contains the code and benchmark results for the scan operation. 

- The main code implementing the scan operation is in the file `benchmark_scan_cuda.cu`. It includes:
1. A sequential O(n^2) implementation for reference.
2. A sequential O(n) implementation.
3. A CPU Hillis-Steele parallel implementation.
4. A CPU Blelloch parallel implementation.
5. A GPU (cuda) Blelloch parallel implementation.

I only focus on the *inclusive* scan operation in this code. This is why the Blelloch implementation may look a bit different from the usual one, which is for the *exclusive* scan operation. Also, the purpose was not to provide the most optimized implementation, but rather a quick and dirty one that works to illustrate the blog post.

- The benchmark results are stored in the file `scan_benchmark_runs.csv`. It contains the execution times for different input sizes and different implementations. The input sizes are powers of two, from 2^5 to 2^25. Each implementation is run multiple times (5 by default) to get an average execution time.

- The benchmark can be run using the `benchmark_cuda.sh` script. It compiles the `benchmark_scan_cuda.cu` file and runs the resulting executable, which generates the `scan_benchmark_runs.csv` file. 

I'm not sure if the code will compile and run on your machine, as it depends on your CUDA setup.

Feel free to explore the code and the benchmark results, and let me know if you have any questions or suggestions!

- The notebook `notebook.ipynb` is a Python notebook that reads the benchmark results from the CSV file and generates plots to visualize the performance of the different implementations.

- The folder `img` contains the generated plots in PNG format.

- The folder `diagrams` contains the diagrams used in the blog post, generated using TikZ.