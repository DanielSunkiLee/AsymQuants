#include <iostream>
#include <sstream>
#include <string>
#include <fstream>

int main() {
   std::ifstream file("toy_model.bin",std::ios::binary);
   if (!file) {
      std::cerr << "Error opening file!" << std::endl;
      return 1;
   }
   
   float my_copied_weights[8] = {0};

   int targetIndex = 68;
   file.seekg(targetIndex,std::ios::beg);

   file.read(reinterpret_cast<char*>(&my_copied_weights[0]), sizeof(my_copied_weights));
   for(int i = 0; i <8; ++i) {
      std::cout << my_copied_weights[i] << " ";
   }
   std::cout << std::endl;

   file.close();
   return 0;
}
