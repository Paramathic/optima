#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
// #include <cuda_f16.h>

#include <vector>

#include <iostream>

#define MAX(a, b) ((abs(a) > abs(b) ? (a) : (b)))
#define MIN(a, b) ((abs(a) < abs(b) ? (a) : (b)))


typedef struct {
   at::Half data;
   int index;
} indexed_half;


__global__ void prune_kernel(
        const float* __restrict__ input,
        float* __restrict__ output,
        bool* __restrict__ mask,
        size_t row_size) {
    const int column = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
    const int index = blockIdx.y * row_size + column;
    if (column < row_size) {
        reinterpret_cast<float4*>(&output[index])[0] = reinterpret_cast<const float4*>(&input[index])[0];
        if(abs(output[index]) > abs(output[index + 1])){
            output[index + 1] = 0.;
            mask[index + 1] = true;
        } else {
            output[index] = 0.;
            mask[index] = true;
        }
        if(abs(output[index + 2]) > abs(output[index + 3])){
            output[index + 3] = 0.;
            mask[index + 3] = true;
        } else {
            output[index + 2] = 0.;
            mask[index + 2] = true;
        }
  }
}


__global__ void prune_kernel(
        const at::Half* __restrict__ input,
        at::Half* __restrict__ output,
        bool* __restrict__ mask,
        size_t row_size) {
    const int column = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
    const int index = blockIdx.y * row_size + column;
    if (column < row_size) {
        reinterpret_cast<float4*>(&output[index])[0] = reinterpret_cast<const float4*>(&input[index])[0];
        at::Half min1, min2;
        int min_idx1, min_idx2;
        min1 = output[index];
        min_idx1 = index;
        if(MIN(min1, output[index + 1]) == abs(output[index + 1])){
            min1 = output[index + 1];
            min_idx1 = index + 1;
        }
        if(MIN(min1, output[index + 2]) == abs(output[index + 2])){
            min1 = output[index + 2];
            min_idx1 = index + 2;
        }
        if(MIN(min1, output[index + 3]) == abs(output[index + 3])){
            min1 = output[index + 3];
            min_idx1 = index + 3;
        }
        min2 = min_idx1 == index ? output[index + 1] : output[index];
        min_idx2 = min_idx1 == index ? index + 1 : index;
        if((MIN(min2, output[index + 1]) == abs(output[index + 1])) && min_idx1 != index + 1){
            min2 = output[index + 1];
            min_idx2 = index + 1;
        }
        if((MIN(min2, output[index + 2]) == abs(output[index + 2])) && min_idx1 != index + 2){
            min2 = output[index + 2];
            min_idx2 = index + 2;
        }
        if((MIN(min2, output[index + 3]) == abs(output[index + 3])) && min_idx1 != index + 3){
            min2 = output[index + 3];
            min_idx2 = index + 3;
        }
        output[min_idx1] = 0.; mask[min_idx1] = true;
        output[min_idx2] = 0.; mask[min_idx2] = true;

        min1 = output[index + 4];
        min_idx1 = index + 4;
        if(MIN(min1, output[index + 5]) == abs(output[index + 5])){
            min1 = output[index + 5];
            min_idx1 = index + 5;
        }
        if(MIN(min1, output[index + 6]) == abs(output[index + 6])){
            min1 = output[index + 6];
            min_idx1 = index + 6;
        }
        if(MIN(min1, output[index + 7]) == abs(output[index + 7])){
            min1 = output[index + 7];
            min_idx1 = index + 7;
        }
        min2 = min_idx1 == index + 4 ? output[index + 5] : output[index + 4];
        min_idx2 = min_idx1 == index + 4 ? index + 5 : index + 4;
        if((MIN(min2, output[index + 5]) == abs(output[index + 5])) && min_idx1 != index + 5){
            min2 = output[index + 5];
            min_idx2 = index + 5;
        }
        if((MIN(min2, output[index + 6]) == abs(output[index + 6])) && min_idx1 != index + 6){
            min2 = output[index + 6];
            min_idx2 = index + 6;
        }
        if((MIN(min2, output[index + 7]) == abs(output[index + 7])) && min_idx1 != index + 7){
            min2 = output[index + 7];
            min_idx2 = index + 7;
        }

        output[min_idx1] = 0.; mask[min_idx1] = true;
        output[min_idx2] = 0.; mask[min_idx2] = true;

//            output[index] = input[index];
//            output[index + 1] = input[index + 1];
//            output[index + 2] = input[index + 2];
//            output[index + 3] = input[index + 3];
//            output[index + 4] = input[index + 4];
//            output[index + 5] = input[index + 5];
//            output[index + 6] = input[index + 6];
//            output[index + 7] = input[index + 7];

//
//         at::Half o0 = output[index];
//         at::Half o1 = output[index + 1];
//         at::Half o2 = output[index + 2];
//         at::Half o3 = output[index + 3];
//         indexed_half top1_1, top2_1, top1_2, top2_2;
//         if(abs(o0) > abs(o1))
//         {
//             top1_1.data = o0; top1_1.index = 0;
//             top2_1.data = o1; top2_1.index = 1;
//         } else {
//             top1_1.data = o1; top1_1.index = 1;
//             top2_1.data = o0; top2_1.index = 0;
//         }
//         if(abs(top2_1.data) < abs(o2))
//         {
//             if(abs(top1_1.data) < abs(o2))
//             {
//                 top2_1.data = top1_1.data; top2_1.index = top1_1.index;
//                 top1_1.data = o2; top1_1.index = 2;
//             } else {
//                 top2_1.data = o2; top2_1.index = 2;
//             }
//         }
//         if(abs(top2_1.data) < abs(o3))
//         {
//             if(abs(top1_1.data) < abs(o3))
//             {
//                 top2_1.data = top1_1.data; top2_1.index = top1_1.index;
//                 top1_1.data = o3; top1_1.index = 3;
//             } else {
//                 top2_1.data = o3;
//                 top2_1.index = 3;
//             }
//         }
//         output[top1_1.index] = 0.; mask[top1_1.index] = false;
//         output[top2_1.index] = 0.; mask[top2_1.index] = false;
//
//         at::Half o4 = output[index + 4];
//         at::Half o5 = output[index + 5];
//         at::Half o6 = output[index + 6];
//         at::Half o7 = output[index + 7];
//         if(abs(o4) > abs(o5))
//         {
//             top1_2.data = o4; top1_2.index = 4;
//             top2_2.data = o5; top2_2.index = 5;
//         } else {
//             top1_2.data = o5; top1_2.index = 5;
//             top2_2.data = o4; top2_2.index = 4;
//         }
//         if(abs(top2_2.data) < abs(o6))
//         {
//             if(abs(top1_2.data) < abs(o6))
//             {
//                 top2_2.data = top1_2.data; top2_2.index = top1_2.index;
//                 top1_2.data = o6; top1_2.index = 6;
//             } else {
//                 top2_2.data = o6; top2_2.index = 6;
//             }
//         }
//         if(abs(top2_2.data) < abs(o7))
//         {
//             if(abs(top1_2.data) < abs(o7))
//             {
//                 top2_2.data = top1_2.data; top2_2.index = top1_2.index;
//                 top1_2.data = o7; top1_2.index = 7;
//             } else {
//                 top2_2.data = o7;
//                 top2_2.index = 7;
//             }
//         }
//         output[top1_2.index] = 0.; mask[top1_2.index] = false;
//         output[top2_2.index] = 0.; mask[top2_2.index] = false;
  }
}


std::vector<torch::Tensor> prune_cuda(
    torch::Tensor input) {

    auto output = torch::zeros_like(input);
    auto options = torch::TensorOptions().dtype(torch::kBool);
    auto mask = torch::zeros_like(input, options);

    const auto batch_size = input.size(0);
    const auto row_size = input.size(1);

    const int threads = 1024;

    switch (input.type().scalarType()) {
        case torch::ScalarType::Float:
        {
            const dim3 blocks(((row_size / 4) + threads - 1) / threads, batch_size);
            prune_kernel<<<blocks, threads>>>(
                input.data<float>(),
                output.data<float>(),
                mask.data<bool>(),
                row_size);
             break;
        }
        case torch::ScalarType::Half:
        {
            const dim3 blocks(((row_size / 8) + threads - 1) / threads, batch_size);
            prune_kernel<<<blocks, threads>>>(
                input.data<at::Half>(),
                output.data<at::Half>(),
                mask.data<bool>(),
                row_size);
        }
    }

//     AT_DISPATCH_ALL_TYPES(input.type(), "prune_cuda", ([&] {
//         prune_kernel<scalar_t><<<blocks, threads>>>(
//             input.data<scalar_t>(),
//             output.data<scalar_t>(),
//             mask.data<scalar_t>(),
//             row_size);
//   }));

  return {output, mask};
}


