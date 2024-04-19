//Pytorch扩展头文件的引用
#include <torch/extension.h> 
#include <vector>
#include <cmath>
using namespace std;

//hsigmoid_cpu函数的具体实现
torch::Tensor hsigmoid_cpu(const torch::Tensor & dets) {
    //TODO: 将输入的tensor转化为浮点类型的vector
    const float* data_ptr = dets.data_ptr<float>();
    std::vector<float> input_data(data_ptr, data_ptr + dets.numel());

    int input_size = input_data.size();
    //TODO: 创建一个浮点类型的output_data，output_data为大小与输入相同的vector
    vector<float> output_data(input_size);
    //TODO: 对于输入向量的每个元素计算hsigmoid: hsigmoid(x)=1/(1+e^(-x))
    for (int i = 0; i < input_size; ++i) {
        output_data[i] = 1.0 / (1.0 + exp(-input_data[i]));
    }
    // Create tensor options with dtype float32
    auto opts = torch::TensorOptions().dtype(torch::kFloat32);
    // Create a tensor from the output vector
    auto foo = torch::from_blob(output_data.data(), {int64_t(output_data.size())}, opts).clone();
    //TODO: 将得到的tensor reshape为[1,3,512,512]
    auto output = foo.reshape({1, 3, 512, 512});
    return output;
} 
//TODO: 算子绑定为Pytorch的模块
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {	
    m.def("hsigmoid_cpu", &hsigmoid_cpu, "HSigmoid activation function");
}       
