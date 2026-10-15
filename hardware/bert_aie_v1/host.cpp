// Board continuation only. This program programs a device when explicitly run.
// It is never invoked by build_hardware.py or the offline functional harness.
#include "model.hpp"
#include <experimental/xrt_kernel.h>
#include <experimental/xrt_graph.h>
#include <fstream>
#include <iostream>
#include <chrono>
#include <stdexcept>
#include <algorithm>
#include <limits>
#include <iomanip>
static void load(const std::string& name,void* dest,size_t bytes) {
    std::ifstream f(name,std::ios::binary|std::ios::ate);
    if(!f||size_t(f.tellg())!=bytes)throw std::runtime_error("Wrong file size: "+name);
    f.seekg(0);f.read(static_cast<char*>(dest),bytes);
}
int main(int argc,char** argv) {
    if(argc<5){std::cerr<<"Usage (physical authorization required): host XCLBIN EXPORT_DIR THRESHOLD DOC_WINDOWS.bin [DOC_WINDOWS.bin ...]\nOr replace document paths by --stdin for persistent document-by-document execution. Each input path contains complete consecutive3072-byte window records (ids,mask,types).\n";return 2;}
    using namespace bert_aie;
    auto cold=std::chrono::steady_clock::now();
    xrt::device device(0);auto uuid=device.load_xclbin(argv[1]);
    xrt::kernel kernel(device,uuid,"bert_pl");xrt::graph graph(device,uuid,"bert_graph");
    xrt::bo in(device,768*4,kernel.group_id(0));
    xrt::bo parameters(device,PARAM_COUNT*4,kernel.group_id(1));
    xrt::bo weights(device,LAYERS*LAYER_WEIGHTS,kernel.group_id(2));
    xrt::bo scales(device,LAYERS*LAYER_SCALES*4,kernel.group_id(3));
    xrt::bo inscales(device,24*4,kernel.group_id(4));
    xrt::bo scratch(device,SCRATCH_FLOATS*4,kernel.group_id(5));
    xrt::bo out(device,3*4,kernel.group_id(6));
    std::string p=argv[2];
    load(p+"/parameters_f32.bin",parameters.map<char*>(),PARAM_COUNT*4);
    load(p+"/weights_i8.bin",weights.map<char*>(),LAYERS*LAYER_WEIGHTS);
    load(p+"/scales_f32.bin",scales.map<char*>(),LAYERS*LAYER_SCALES*4);
    load(p+"/input_scales_f32.bin",inscales.map<char*>(),24*4);
    parameters.sync(XCL_BO_SYNC_BO_TO_DEVICE);weights.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    scales.sync(XCL_BO_SYNC_BO_TO_DEVICE);inscales.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    graph.run(-1);
    std::cout<<"{\"cold_setup_seconds\":"<<std::chrono::duration<double>(std::chrono::steady_clock::now()-cold).count()<<"}"<<std::endl;
    double threshold=std::stod(argv[3]);
    if(!std::isfinite(threshold))throw std::runtime_error("Nonfinite margin threshold");
    std::cout<<std::setprecision(10);
    bool stdin_mode=argc==5&&std::string(argv[4])=="--stdin";
    for(int d=0;;++d) {
        std::string document;
        if(stdin_mode){if(!std::getline(std::cin,document))break;if(document.empty())throw std::runtime_error("Empty document path");}
        else {if(d+4>=argc)break;document=argv[d+4];}
        auto start=std::chrono::steady_clock::now();
        std::ifstream f(document,std::ios::binary|std::ios::ate);
        if(!f||f.tellg()<=0||size_t(f.tellg())%(768*4))throw std::runtime_error("Invalid document windows");
        size_t windows=size_t(f.tellg())/(768*4);f.seekg(0);float margin=-std::numeric_limits<float>::infinity();
        for(size_t w=0;w<windows;++w) {
            f.read(in.map<char*>(),768*4);in.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            auto run=kernel(in,parameters,weights,scales,inscales,scratch,out);run.wait();
            out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);float* v=out.map<float*>();
            if(v[2]!=0.f)throw std::runtime_error("Input rejected by hardware");
            if(!std::isfinite(v[0])||!std::isfinite(v[1]))throw std::runtime_error("Nonfinite hardware logits");
            float wm=v[1]-v[0];if(!std::isfinite(wm))throw std::runtime_error("Nonfinite hardware margin");margin=std::max(margin,wm);
            std::cout<<"{\"document_index\":"<<d<<",\"window_index\":"<<w<<",\"logits\":["<<v[0]<<","<<v[1]<<"],\"margin\":"<<wm<<"}\n";
        }
        std::cout<<"{\"document_index\":"<<d<<",\"windows\":"<<windows<<",\"max_margin\":"<<margin
            <<",\"blocked\":"<<(margin>threshold?"true":"false")<<",\"seconds_excluding_tokenization\":"
            <<std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()<<"}"<<std::endl;
    }
    graph.end();return 0;
}
