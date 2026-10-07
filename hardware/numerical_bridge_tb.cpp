// One selected real-development numerical bridge, not a synthesizable controller.
// FP arithmetic calls the current common service. Encoder matrix products use a
// checked native int64 reference of the already-validated W8/A8 integer contract.
// CPU orchestration/gathers are testbench work, never FPGA-offload/timing evidence.
#include <ap_int.h>
#include <algorithm>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include "fixed_fp32_service.hpp"

extern "C" void gate_fixed_fp32_top(
    int,const float*,const float*,const float*,const ap_int<32>*,
    float*,ap_int<8>*,int,int,int,float,bool,int*);

namespace {
using F = std::vector<float>;
constexpr int L=256, D=256, H=4, HD=64, FF=1024, CAP=32768;
std::string directory, output_directory;
bool first_divergence_diagnostic=false;
long long calls[9]={0};
long long same_input_codes_checked=0, exact_accumulators_checked=0;
struct Error {std::string name; size_t count; double maximum, mean, rms;};
struct IntegerCheck {std::string name; long long input_codes, original_code_differences, accumulators;};
std::vector<Error> errors;
std::vector<IntegerCheck> integer_checks;

template<class T> std::vector<T> read_binary(const std::string& name,size_t count) {
    std::ifstream file(directory+"/"+name+".bin",std::ios::binary);
    if(!file) throw std::runtime_error("missing binary "+name);
    std::vector<T> result(count);
    file.read(reinterpret_cast<char*>(result.data()),count*sizeof(T));
    if(file.gcount()!=static_cast<std::streamsize>(count*sizeof(T)) || file.peek()!=std::char_traits<char>::eof())
        throw std::runtime_error("binary extent mismatch "+name);
    return result;
}
void write_binary(const std::string& name,const F& values) {
    std::ofstream file(output_directory+"/"+name+".bin",std::ios::binary);
    file.write(reinterpret_cast<const char*>(values.data()),values.size()*sizeof(float));
    if(!file) throw std::runtime_error("cannot save "+name);
}
F slice(const F& values,size_t start,size_t count) {
    if(start+count>values.size()) throw std::runtime_error("slice extent");
    return F(values.begin()+start,values.begin()+start+count);
}
const float* pointer(const F& values) {
    static const float unused=0.0f;
    return values.empty()?&unused:values.data();
}
void finite(const F& values,const std::string& name) {
    for(float value:values) if(!std::isfinite(value)) throw std::runtime_error("nonfinite "+name);
}
F fp(int op,const F& x,const F& y,const F& z,int rows,int width,int outputs=0,
     float scale=1.0f,bool bias=false,const std::vector<ap_int<32>>& acc={}) {
    if(op==gate::FP32_QUANTIZE) throw std::runtime_error("use quantize helper");
    const int count=rows*(op==gate::FP32_DOT?outputs:width);
    F result(count,-777.0f);
    ap_int<32> unused_acc=0;
    ap_int<8> unused_code=0;
    int status=-1;
    gate_fixed_fp32_top(op,pointer(x),pointer(y),pointer(z),
        acc.empty()?&unused_acc:acc.data(),result.data(),&unused_code,
        rows,width,outputs,scale,bias,&status);
    ++calls[op];
    if(status!=0) throw std::runtime_error("FP service status op="+std::to_string(op));
    finite(result,"service output");
    return result;
}
F elements(int op,const F& x,const F& y,const F& z,int rows,int width,bool per_channel=false) {
    F result(rows*width);
    for(int row=0;row<rows;) {
        const int count_rows=std::min(rows-row,CAP/width), count=count_rows*width, begin=row*width;
        const F xx=slice(x,begin,count);
        const F yy=y.empty()?F{}:(per_channel?y:slice(y,begin,count));
        const F zz=z.empty()?F{}:(per_channel?z:slice(z,begin,count));
        F current=fp(op,xx,yy,zz,count_rows,width);
        std::copy(current.begin(),current.end(),result.begin()+begin);
        row+=count_rows;
    }
    return result;
}
F norm(const F& x,const std::string& name) {
    const F parameters=read_binary<float>(name,2*D);
    return elements(gate::FP32_LAYER_NORM,x,slice(parameters,0,D),slice(parameters,D,D),L,D,true);
}
void compare(const std::string& name,const F& actual) {
    const F reference=read_binary<float>("reference_"+name,actual.size());
    finite(reference,"reference");
    double maximum=0.0,sum=0.0,squared=0.0;
    for(size_t i=0;i<actual.size();++i) {
        const double delta=std::abs(double(actual[i])-double(reference[i]));
        maximum=std::max(maximum,delta);sum+=delta;squared+=delta*delta;
    }
    errors.push_back({name,actual.size(),maximum,sum/actual.size(),std::sqrt(squared/actual.size())});
    write_binary("bridge_"+name,actual);
    std::cout<<"BRIDGE_STAGE "<<name<<" values="<<actual.size()<<" max_abs="<<maximum
             <<" mean_abs="<<sum/actual.size()<<std::endl;
}
std::vector<std::int8_t> quantize(const F& x,int rows,int width,float scale) {
    std::vector<std::int8_t> result(x.size());
    for(int row=0;row<rows;) {
        const int count_rows=std::min(rows-row,CAP/width), count=count_rows*width, begin=row*width;
        std::vector<ap_int<8>> codes(count);
        ap_int<32> unused_acc=0;float unused=0;int status=-1;
        gate_fixed_fp32_top(gate::FP32_QUANTIZE,x.data()+begin,&unused,&unused,&unused_acc,
                           &unused,codes.data(),count_rows,width,0,scale,false,&status);
        ++calls[gate::FP32_QUANTIZE];
        if(status!=0) throw std::runtime_error("quantize service status");
        for(int i=0;i<count;++i) {
            // Independent same-input reference with one rounded FP32 division.
            volatile float quotient=x[begin+i]/scale;
            if(std::isnan(quotient)) throw std::runtime_error("NaN quantization quotient");
            int expected=quotient>=127.0f?127:quotient<=-128.0f?-128:int(std::nearbyint(double(quotient)));
            const int actual=codes[i].to_int();
            if(actual!=expected) throw std::runtime_error("same-input quantize mismatch index="+std::to_string(begin+i));
            result[begin+i]=static_cast<std::int8_t>(actual);
            ++same_input_codes_checked;
        }
        row+=count_rows;
    }
    return result;
}
F linear(const std::string& name,const F& x,int k,int n) {
    const auto weights=read_binary<std::int8_t>(name+"_w8",n*k);
    const F parameters=read_binary<float>(name+"_params",1+2*n);
    const auto a=quantize(x,L,k,parameters[0]);
    const auto original=read_binary<std::int8_t>(name+"_reference_a8",L*k);
    long long changed=0;
    for(size_t i=0;i<a.size();++i) {
        changed+=a[i]!=original[i];
        if(first_divergence_diagnostic && name=="layer1_attention_output" && a[i]!=original[i]) {
            volatile float scaled=x[i]/parameters[0];
            const double boundary=(int(a[i])+int(original[i]))/2.0;
            const float ulp=std::nextafter(float(scaled),std::numeric_limits<float>::infinity())-scaled;
            std::cout<<"GATE_BRIDGE_CODE_DIFFERENCE {\"name\":\""<<name<<"\",\"index\":"<<i
                     <<",\"token\":"<<i/k<<",\"channel\":"<<i%k
                     <<",\"scale\":"<<parameters[0]<<",\"input\":"<<x[i]
                     <<",\"scaled\":"<<scaled<<",\"original_code\":"<<int(original[i])
                     <<",\"service_code\":"<<int(a[i])<<",\"boundary\":"<<boundary
                     <<",\"scaled_ulp\":"<<ulp<<",\"distance_from_boundary_ulps\":"
                     <<(double(scaled)-boundary)/ulp<<"}"<<std::endl;
        }
    }
    if(first_divergence_diagnostic && name=="layer1_attention_output")
        write_binary("diagnostic_layer1_attention_context",x);
    std::vector<std::int32_t> accumulators(L*n);
    // Optimized ordinary C++ loops, not another full ap_int hardware simulation.
    // Direct signed dot and signed-high/unsigned-low decomposition must agree.
    for(int r=0;r<L;++r) for(int o=0;o<n;++o) {
        std::int64_t direct=0,high_sum=0,low_sum=0;
        for(int j=0;j<k;++j) {
            const int av=a[r*k+j], w=weights[o*k+j];
            const int high=w<0?(w-15)/16:w/16;
            const int low=w-16*high;
            direct+=std::int64_t(av)*w;
            high_sum+=std::int64_t(av)*high;
            low_sum+=std::int64_t(av)*low;
        }
        if(direct!=16*high_sum+low_sum || direct<std::numeric_limits<std::int32_t>::min()
                || direct>std::numeric_limits<std::int32_t>::max())
            throw std::runtime_error("integer contract failure "+name);
        accumulators[r*n+o]=static_cast<std::int32_t>(direct);
        ++exact_accumulators_checked;
    }
    F result(L*n);
    const F weight_scale=slice(parameters,1,n), bias=slice(parameters,1+n,n);
    for(int row=0;row<L;) {
        const int count_rows=std::min(L-row,CAP/n), count=count_rows*n, begin=row*n;
        std::vector<ap_int<32>> current(count);
        for(int i=0;i<count;++i)current[i]=accumulators[begin+i];
        const F block=fp(gate::FP32_RESCALE_BIAS,{},weight_scale,bias,count_rows,n,0,parameters[0],false,current);
        std::copy(block.begin(),block.end(),result.begin()+begin);
        row+=count_rows;
    }
    if(first_divergence_diagnostic && (name=="layer1_query" || name=="layer1_key" || name=="layer1_value"))
        write_binary("diagnostic_"+name+"_output",result);
    integer_checks.push_back({name,L*k,changed,L*n});
    std::cout<<"BRIDGE_LINEAR "<<name<<" original_software_a8_differences="<<changed<<"/"<<L*k<<std::endl;
    return result;
}
F attention(const F& q,const F& k,const F& v,const F& key_mask) {
    F context(L*D);
    constexpr int M=CAP/L; //128 rows, full sequence retained.
    for(int head=0;head<H;++head) {
        F keys(L*HD), transposed_v(HD*L);
        for(int token=0;token<L;++token) for(int d=0;d<HD;++d) {
            keys[token*HD+d]=k[token*D+head*HD+d];
            transposed_v[d*L+token]=v[token*D+head*HD+d];
        }
        for(int first=0;first<L;first+=M) {
            const int rows=std::min(M,L-first);
            F queries(rows*HD), scores(rows*L), mask(rows*L);
            for(int r=0;r<rows;++r) for(int d=0;d<HD;++d)
                queries[r*HD+d]=q[(first+r)*D+head*HD+d];
            for(int key=0;key<L;key+=64) {
                const F panel=slice(keys,key*HD,64*HD);
                const F tile=fp(gate::FP32_DOT,queries,panel,{},rows,HD,64);
                for(int r=0;r<rows;++r) for(int o=0;o<64;++o)
                    scores[r*L+key+o]=tile[r*64+o];
            }
            for(int r=0;r<rows;++r)std::copy(key_mask.begin(),key_mask.end(),mask.begin()+r*L);
            const F probabilities=fp(gate::FP32_SOFTMAX,scores,mask,{},rows,L,0,8.0f);
            const F block=fp(gate::FP32_DOT,probabilities,transposed_v,{},rows,L,HD);
            for(int r=0;r<rows;++r) for(int d=0;d<HD;++d)
                context[(first+r)*D+head*HD+d]=block[r*HD+d];
        }
    }
    return context;
}
void write_report(const F& logits,float risk,float threshold,bool same_decision) {
    const F reference_logits=read_binary<float>("reference_logits",2);
    const float reference_risk=read_binary<float>("reference_risk",1)[0];
    std::ofstream file(output_directory+"/bridge_summary.json");
    file<<std::setprecision(17);
    file<<"{\n  \"completed\":true,\n  \"case\":\"aeslc:train:panus-s_inbox_1.subject:clean\",\n"
        <<"  \"source_revision\":\""<<GATE_FIXED_FP32_SCHEDULE_REVISION<<"\",\n"
        <<"  \"same_input_quantization_exact\":true,\n  \"integer_accumulators_exact\":true,\n"
        <<"  \"same_input_codes_checked\":"<<same_input_codes_checked<<",\n"
        <<"  \"exact_accumulators_checked\":"<<exact_accumulators_checked<<",\n"
        <<"  \"logits\":["<<logits[0]<<","<<logits[1]<<"],\n"
        <<"  \"reference_logits\":["<<reference_logits[0]<<","<<reference_logits[1]<<"],\n"
        <<"  \"risk\":"<<risk<<",\n  \"reference_risk\":"<<reference_risk<<",\n"
        <<"  \"risk_absolute_error\":"<<std::abs(double(risk)-reference_risk)<<",\n"
        <<"  \"threshold\":"<<threshold<<",\n  \"strict_reject\":"<<(risk>threshold?"true":"false")<<",\n"
        <<"  \"reference_strict_reject\":"<<(reference_risk>threshold?"true":"false")<<",\n"
        <<"  \"strict_decision_agrees\":"<<(same_decision?"true":"false")<<",\n"
        <<"  \"end_to_end_score_tolerance\":null,\n"
        <<"  \"scope\":\"One numerical C-sim bridge, CPU exact integer contract, actual common FP32 service; not full RTL, FPGA orchestration, latency or board evidence\",\n"
        <<"  \"service_calls_by_op\":[";
    for(int op=0;op<9;++op)file<<(op?",":"")<<calls[op];
    file<<"],\n  \"intermediate_errors\":[\n";
    for(size_t i=0;i<errors.size();++i) {
        const auto& e=errors[i];
        file<<(i?",\n":"")<<"    {\"name\":\""<<e.name<<"\",\"values\":"<<e.count
            <<",\"maximum_absolute_error\":"<<e.maximum<<",\"mean_absolute_error\":"<<e.mean
            <<",\"rms_error\":"<<e.rms<<"}";
    }
    file<<"\n  ],\n  \"integer_layers\":[\n";
    for(size_t i=0;i<integer_checks.size();++i) {
        const auto& c=integer_checks[i];
        file<<(i?",\n":"")<<"    {\"name\":\""<<c.name<<"\",\"input_codes\":"<<c.input_codes
            <<",\"original_software_code_differences\":"<<c.original_code_differences
            <<",\"accumulators_checked\":"<<c.accumulators<<"}";
    }
    file<<"\n  ]\n}\n";
    if(!file)throw std::runtime_error("cannot save summary");
}
}

int main() {
    try {
        static_assert(sizeof(float)==4,"FP32 required");
        const std::uint32_t endian=1;
        if(*reinterpret_cast<const unsigned char*>(&endian)!=1)throw std::runtime_error("little-endian host required");
        const char* location=std::getenv("GATE_NUMERICAL_BRIDGE_DIR");
        if(!location)throw std::runtime_error("GATE_NUMERICAL_BRIDGE_DIR missing");
        directory=location;
        const char* diagnostic=std::getenv("GATE_NUMERICAL_BRIDGE_DIAGNOSTIC_DIR");
        first_divergence_diagnostic=diagnostic!=nullptr;
        output_directory=diagnostic?diagnostic:directory;
        std::fesetround(FE_TONEAREST);
        std::cout<<std::setprecision(17);
        const F word=read_binary<float>("embedding_word",L*D);
        const F type=read_binary<float>("embedding_type",L*D);
        const F position=read_binary<float>("embedding_position",L*D);
        const F mask=read_binary<float>("key_mask",L);
        F hidden=norm(elements(gate::FP32_EMBED_ADD,word,type,position,L,D),"embedding_ln");
        compare("embedding_ln",hidden);
        for(int layer=0;layer<4;++layer) {
            const std::string prefix="layer"+std::to_string(layer);
            const F q=linear(prefix+"_query",hidden,D,D);
            const F k=linear(prefix+"_key",hidden,D,D);
            const F v=linear(prefix+"_value",hidden,D,D);
            const F context=attention(q,k,v,mask);
            const F projected=linear(prefix+"_attention_output",context,D,D);
            const F normalized=norm(elements(gate::FP32_RESIDUAL_ADD,projected,hidden,{},L,D),prefix+"_attention_ln");
            const F intermediate=linear(prefix+"_ffn_input",normalized,D,FF);
            const F activated=elements(gate::FP32_GELU,intermediate,{},{},L,FF);
            const F output=linear(prefix+"_ffn_output",activated,FF,D);
            hidden=norm(elements(gate::FP32_RESIDUAL_ADD,output,normalized,{},L,D),prefix+"_output_ln");
            compare(prefix,hidden);
        }
        const F pool_weights=read_binary<float>("pooler_weight",D*D);
        const F pool_bias=read_binary<float>("pooler_bias",D);
        const F cls=slice(hidden,0,D);
        F pooled(D);
        for(int o=0;o<D;o+=64) {
            const F tile=fp(gate::FP32_DOT,cls,slice(pool_weights,o*D,64*D),
                            slice(pool_bias,o,64),1,D,64,1.0f,true);
            std::copy(tile.begin(),tile.end(),pooled.begin()+o);
        }
        pooled=fp(gate::FP32_TANH,pooled,{},{},1,D);
        const F logits=fp(gate::FP32_DOT,pooled,read_binary<float>("classifier_weight",2*D),
                          read_binary<float>("classifier_bias",2),1,D,2,1.0f,true);
        compare("logits",logits);
        const F probabilities=fp(gate::FP32_SOFTMAX,logits,F(2,0.0f),{},1,2);
        const F risk{probabilities[1]};
        compare("risk",risk);
        long long total_calls=0;for(auto count:calls)total_calls+=count;
        if(total_calls!=411 || integer_checks.size()!=24)
            throw std::runtime_error("fixed schedule call count differs");
        const float threshold=read_binary<float>("threshold",1)[0];
        const float reference_risk=read_binary<float>("reference_risk",1)[0];
        const bool same_decision=(risk[0]>threshold)==(reference_risk>threshold);
        write_report(logits,risk[0],threshold,same_decision);
        std::cout<<"GATE_NUMERICAL_BRIDGE_COMPLETE calls="<<total_calls
                 <<" strict_decision_agrees="<<(same_decision?1:0)
                 <<" no_end_to_end_tolerance_invented=1"<<std::endl;
        return same_decision?0:1;
    } catch(const std::exception& error) {
        std::cerr<<"GATE_NUMERICAL_BRIDGE_FAIL "<<error.what()<<std::endl;
        return 1;
    }
}

