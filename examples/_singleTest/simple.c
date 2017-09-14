#include <stdio.h>
#include <stdint.h>
#include "ebnn.h"
#include "simple.h"

int main()
{
  float input[784];
  uint8_t output[1];
   
  //simulate a 28 by 28 greyscale image
  for(int i = 0; i < 28*28; ++i) {
    input[i] = i;
  }
  
  std::cout << "input size is : " << (sizeof(input)/sizeof(input[0])) << std::endl;
  
  ebnn_compute(input, output);
  printf("result is : %d\n", output[0]);
   
  return 0;
}