#include <iostream>

// Since C++ 11
template<typename T>
using func_t = T (*) (T, T);

template <typename T> 
__device__ T add_func (T x, T y)
{
    return x + y;
}

template <typename T> 
__device__ T mul_func (T x, T y)
{
    return x * y;
}

// Required for functional pointer argument in kernel function
// Static pointers to device functions
template <typename T> 
__device__ func_t<T> p_add_func = add_func<T>;
template <typename T> 
__device__ func_t<T> p_mul_func = mul_func<T>;


template <typename T> 
__global__ void kernel(func_t<T> op, T * d_x, T * d_y, T * result)
{
 printf("op %p size %lX\n", op, sizeof(op));
    *result = (*op)(*d_x, *d_y);
}

template <typename T> 
void test(T x, T y)
{
    func_t<T> h_add_func = nullptr;
    func_t<T> h_mul_func = nullptr;

    T * d_x, * d_y;
    cudaMalloc(&d_x, sizeof(T));
    cudaMalloc(&d_y, sizeof(T));
    cudaMemcpy(d_x, &x, sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, &y, sizeof(T), cudaMemcpyHostToDevice);

    T result;
    T * d_result, * h_result;
    cudaMalloc(&d_result, sizeof(T));
    h_result = &result;

    // Copy device function pointer to host side
    cudaMemcpyFromSymbol(&h_add_func, p_add_func<T>, sizeof(func_t<T>));
    cudaMemcpyFromSymbol(&h_mul_func, p_mul_func<T>, sizeof(func_t<T>));

printf("h_add_func %p size %lX h_mul_func %p\n", h_add_func, sizeof(h_add_func), h_mul_func);
    void *d_addr = nullptr;
    auto err = cudaMemcpy(&d_addr, (const void *)h_add_func, sizeof(d_addr), cudaMemcpyDeviceToHost);
printf("d_addr %p, err %d (%s)\n", d_addr, err, cudaGetErrorString(err));

    kernel<T><<<1,1>>>(h_add_func, d_x, d_y, d_result);
    cudaDeviceSynchronize();
    cudaMemcpy(h_result, d_result, sizeof(T), cudaMemcpyDeviceToHost);
    std::cout << "Sum: " << result << std::endl;

    kernel<T><<<1,1>>>(h_mul_func, d_x, d_y, d_result);
    cudaDeviceSynchronize();
    cudaMemcpy(h_result, d_result, sizeof(T), cudaMemcpyDeviceToHost);
    std::cout << "Product: " << result << std::endl;

    cudaFree(d_result);
    cudaFree(d_x); cudaFree(d_y);
}

int main()
{
    std::cout << "Test int for type int ..." << std::endl;
    test<int>(2.05, 10.00);

    std::cout << "Test float for type float ..." << std::endl;
    test<float>(2.05, 10.00);

    std::cout << "Test double for type double ..." << std::endl;
    test<double>(2.05, 10.00);
}
