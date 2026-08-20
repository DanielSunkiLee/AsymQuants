#include <iostream>
#include <fstream>
#include <cstdint>
#include <string>

struct GGUFHeader{
    char magic[4];
    uint32_t version;
    uint64_t tensor_count;
    uint64_t kv_count;
};

int main(){
    std::string file_path = "/mnt/d/my_model/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf";

    std::cout << "try to open gguf file.." << std::endl;

    std::ifstream file(file_path,std::ios::binary);
    if (!file) {
        std::cerr << "fail to load file. check again." << file_path << std::endl;
        return 1;
    }

    std::cout << "success!" << std::endl;

    GGUFHeader header;
    file.read(reinterpret_cast<char*>(&header),sizeof(GGUFHeader));

    if (std::string(header.magic,4) != "GGUF"){
	    std::cerr << "not correct gguf file format" << std::endl;
	    return 1;
    }

    std::cout << "gguf header info" << std::endl;
    std::cout << "ver: " << header.version << std::endl;
    std::cout << "total num of weights: " << header.tensor_count << std::endl;
    std::cout << "num of metadat: " << header.kv_count << std::endl;

    return 0;
}
