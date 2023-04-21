#ifdef  __cplusplus
extern "C" {
#endif

#if defined(GGML_USE_HIPBLAS)
void dequantize_row_q4_0_hip(const void * vx, float * y, int k, hipStream_t stream);
void dequantize_row_q4_1_hip(const void * vx, float * y, int k, hipStream_t stream);
void dequantize_row_q4_2_hip(const void * vx, float * y, int k, hipStream_t stream);
void dequantize_row_q4_3_hip(const void * vx, float * y, int k, hipStream_t stream);
#else
void dequantize_row_q4_0_cuda(const void * vx, float * y, int k, cudaStream_t stream);
void dequantize_row_q4_1_cuda(const void * vx, float * y, int k, cudaStream_t stream);
void dequantize_row_q4_2_cuda(const void * vx, float * y, int k, cudaStream_t stream);
void dequantize_row_q4_3_cuda(const void * vx, float * y, int k, cudaStream_t stream);
#endif

#ifdef  __cplusplus
}
#endif
