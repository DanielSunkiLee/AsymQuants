#include <iostream>
#include <fstream>
#include <cstdint>
#include <cstring>

struct ToyHeader {
    char magic[4] = {'T','O','Y','_'};
    int32_t version = 1;
    uint32_t tensor_count = 2;
};

int main() {
    std::ofstream file("toy_model.bin", std::ios::binary);
    if (!file) {
        std::cerr << "fail to create file." << '\n';
        return 1;
    }

    ToyHeader header;
    file.write(reinterpret_cast<char*>(&header), sizeof(ToyHeader));

    char model_name[32] = {0};
    std::strncpy(model_name, "Gemma-Toy-Quant-v1", sizeof(model_name)-1);
    file.write(model_name, sizeof(model_name));

    float weight_1[6] = {0.1f,-0.5f,1.2f,3.4f,-2.1f,0.0f};
    file.write(reinterpret_cast<char*>(weight_1), sizeof(weight_1));

    float weight_2[8] = {9.9f,8.8f,7.7f,6.6f,5.5f,4.4f,3.3f,2.2f};
    file.write(reinterpret_cast<char*>(weight_2), sizeof(weight_2));

    file.close();
    std::cout << "complete to create binary file." << std::endl;
    return 0;
}

