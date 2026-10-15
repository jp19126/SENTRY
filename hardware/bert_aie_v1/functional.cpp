#include "model.hpp"
#include <vector>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <stdexcept>
#include <cstring>
template<class T>std::vector<T> read(const std::string& path,size_t count) {
    std::ifstream f(path,std::ios::binary|std::ios::ate);
    if(!f||size_t(f.tellg())!=count*sizeof(T))throw std::runtime_error("wrong/missing size: "+path);
    std::vector<T> x(count);f.seekg(0);f.read(reinterpret_cast<char*>(x.data()),count*sizeof(T));return x;
}
int main(int argc,char** argv) {
    using namespace bert_aie;
    try {
        if(argc!=4)throw std::runtime_error("usage: functional EXPORT_DIR INPUT.bin|--stdin resident|external|both; stdin is consecutive 3072-byte window records");
        std::string path=argv[1];
        std::string mapping=argv[3];
        if(mapping!="resident"&&mapping!="external"&&mapping!="both")throw std::runtime_error("unknown mapping");
        auto p=read<float>(path+"/parameters_f32.bin",PARAM_COUNT);
        auto w=read<int8_t>(path+"/weights_i8.bin",LAYERS*LAYER_WEIGHTS);
        auto s=read<float>(path+"/scales_f32.bin",LAYERS*LAYER_SCALES);
        auto is=read<float>(path+"/input_scales_f32.bin",24);
        const bool stream=std::string(argv[2])=="--stdin";
        std::vector<int32_t> inputs(768);
        if(!stream)inputs=read<int32_t>(argv[2],768);
        std::vector<float> scratch(SCRATCH_FLOATS);
        std::cout<<std::setprecision(10);
        size_t index=0;
        while(true) {
            if(stream) {
                std::cin.read(reinterpret_cast<char*>(inputs.data()),768*sizeof(int32_t));
                if(std::cin.gcount()==0&&std::cin.eof())break;
                if(std::cin.gcount()!=768*sizeof(int32_t))throw std::runtime_error("partial stdin window record");
            }
            auto execute=[&](bool resident,float* logits) {
                CpuTiles e;auto start=std::chrono::steady_clock::now();
                int status=resident?
                    model<true>(e,inputs.data(),inputs.data()+256,inputs.data()+512,p.data(),w.data(),s.data(),is.data(),scratch.data(),logits):
                    model<false>(e,inputs.data(),inputs.data()+256,inputs.data()+512,p.data(),w.data(),s.data(),is.data(),scratch.data(),logits);
                double sec=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
                std::cout<<"{\"status\":"<<status<<",\"logits\":["<<logits[0]<<","<<logits[1]
                    <<"],\"integer_tiles\":"<<e.integer_tiles<<",\"bf16_tiles\":"<<e.attention_tiles<<",\"seconds\":"<<sec<<"}";
                return status;
            };
            float resident_logits[2]={},external_logits[2]={};int status=0;
            if(mapping=="both") {
                std::cout<<"{\"window_index\":"<<index<<",\"resident\":";status=execute(true,resident_logits);
                std::cout<<",\"external\":";status|=execute(false,external_logits);
                std::cout<<",\"mapping_bit_identical\":"<<(std::memcmp(resident_logits,external_logits,sizeof(resident_logits))==0?"true":"false")
                    <<",\"boundary\":\"portable C++ full-model functional execution; no AIE/RTL/device timing\"}"<<std::endl;
            } else {
                status=execute(mapping=="resident",resident_logits);std::cout<<std::endl;
            }
            if(status)return status;
            ++index;if(!stream)break;
        }
        return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 2;}
}
