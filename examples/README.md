This directory contains basic examples and timing benchmarks compared
to existing packages such as `iisignature`, `signatory`, `esig`
and `sigkernel`.

### Examples

- `small_example.py` - validates pySigLib signatures against iisignature (single and batch)
- `signature_example.py` - times pySigLib signatures against iisignature
- `sig_combine_example.py` - times pySigLib sig_combine against iisignature sigcombine
- `sig_kernel_example.py` - times pySigLib sig_kernel against sigkernel (CPU and CUDA)

### Timing benchmarks

- `sig_times.py` - runtime for signatures (serial and parallel)
- `sig_backprop_times.py` - runtime for backprop through signatures
- `sig_kernel_times.py` - runtime for signature kernels (CPU and CUDA)
- `sig_kernel_backprop_times.py` - runtime for backprop through signature kernels (CPU and CUDA)

### Timing plots (runtime vs degree/path length)

- `sig_times_plot.py` - signatures (serial and parallel, Direct vs Horner)
- `sig_backprop_times_plot.py` - signature backprop
- `log_sig_times_plot.py` - log signatures (serial and parallel)
- `sig_kernel_times_plot.py` - signature kernels vs path length
- `sig_kernel_backprop_times_plot.py` - signature kernel backprop vs path length

### CPU vs CUDA comparisons (`cpu_vs_cuda/`)

- `sig_times_cpu_vs_cuda_plot.py` - signatures (Direct and Horner)
- `sig_backprop_times_cpu_vs_cuda_plot.py` - signature backprop
- `log_sig_times_cpu_vs_cuda_plot.py` - log signatures
- `sig_coef_times_cpu_vs_cuda_plot.py` - signature coefficients
- `sig_combine_times_cpu_vs_cuda_plot.py` - signature combine
- `sig_kernel_times_cpu_vs_cuda_plot.py` - signature kernels vs path length
- `sig_kernel_backprop_times_cpu_vs_cuda_plot.py` - signature kernel backprop vs path length

### Utilities

- `timing_utils.py` - shared timing functions for all benchmarks
- `plotting_params.py` - shared matplotlib style settings