#include "span_guard.hpp"
#include <fstream>
#include <iostream>
#include <vector>
#include <cstring>
using namespace sentry_span;
static std::vector<uint8_t> bytes(const char *p){std::ifstream f(p,std::ios::binary);if(!f)throw std::runtime_error("Cannot open input");return {std::istreambuf_iterator<char>(f),{}};}
static uint32_t u32(const std::vector<uint8_t>& b,size_t& p){if(p+4>b.size())throw std::runtime_error("Short cases");uint32_t x=0;for(int i=0;i<4;++i)x|=uint32_t(b[p++])<<(8*i);return x;}
static uint16_t u16(const std::vector<uint8_t>& b,size_t& p){if(p+2>b.size())throw std::runtime_error("Short IDs");uint16_t x=b[p++];x|=uint16_t(b[p++])<<8;return x;}
template<class T>static void dump(std::ofstream& f,const T *data,size_t n){f.write(reinterpret_cast<const char*>(data),n*sizeof(T));}
int main(int argc,char **argv){
    try{
        if(argc<3){std::cerr<<"usage: span_reference model.hls.bin cases.bin [trace-prefix]\n";return 2;}
        auto model=bytes(argv[1]);const auto cases=bytes(argv[2]);const uint32_t model_bytes=uint32_t(model.size());size_t p=0;
        // Match declared RTL port extents even for a short/empty transaction.
        model.resize(2100000,0);
        if(u32(cases,p)!=0x53505431u||u32(cases,p)!=1)throw std::runtime_error("Cases version");
        const uint32_t count=u32(cases,p);static Model m;if(!load_model(model.data(),model_bytes,m))throw std::runtime_error("Invalid model");
        static uint16_t cold_ids[CHUNK*MAX_TOKENS]={};static Window cold_windows[CHUNK]={};
        Result initialized;span_guard_top(model.data(),model_bytes,cold_ids,cold_windows,0,1,0,0,0,&initialized);
        if(initialized.error)throw std::runtime_error("Cold model loading failed");
        uint32_t failures=0;
        for(uint32_t c=0;c<count;++c){
            const uint32_t n=u32(cases,p),empty=u32(cases,p),previous=u32(cases,p),threshold=u32(cases,p),expected=u32(cases,p),anchors_expected=u32(cases,p);
            if(n>CHUNK)throw std::runtime_error("Split input into bounded chunks");
            std::vector<Window> ws(CHUNK);std::vector<uint16_t> ids(CHUNK*MAX_TOKENS);
            for(uint32_t i=0;i<n;++i){Window& w=ws[i];w.length=u16(cases,p);w.left_edge=uint8_t(u16(cases,p));w.right_edge=uint8_t(u16(cases,p));(void)u16(cases,p);
                w.offset=i*MAX_TOKENS;w.content_start=u32(cases,p);w.document_length=u32(cases,p);
                for(int t=0;t<MAX_TOKENS;++t)ids[w.offset+t]=u16(cases,p);
            }
            Result actual;span_guard_top(model.data(),model_bytes,ids.data(),ws.data(),n,0,empty,previous,threshold,&actual);
            uint32_t ref=previous,anchors=0;
            if(empty)ref=risk_lookup(m,m.head_bias);
            for(uint32_t i=0;i<n;++i){const Window w=ws[i];Result independent;
                if(argc>=4){static Trace trace;std::memset(&trace,0,sizeof(trace));independent=window_trace(m,ids.data()+w.offset,w,threshold,trace);
                    const std::string prefix=std::string(argv[3])+"_c"+std::to_string(c)+"_w"+std::to_string(i);
                    std::ofstream f(prefix+".bin",std::ios::binary);const uint32_t header[4]={0x53505231u,1,w.length,m.kind};dump(f,header,4);
                    dump(f,&trace.z[0][0],w.length*Z);dump(f,&trace.p[0][0],w.length*2*R);dump(f,&trace.a[0][0],w.length*R);dump(f,&trace.b[0][0],w.length*R);dump(f,trace.g,w.length);dump(f,trace.eligible,w.length);
                }else independent=m.kind==3?window_sopa(m,ids.data()+w.offset,w,threshold):window_reference(m,ids.data()+w.offset,w,threshold);
                if(independent.error)throw std::runtime_error("Reference rejects window");
                if(independent.risk_bits>ref)ref=independent.risk_bits;anchors+=independent.anchors;
            }
            const bool pass=actual.error==OK&&actual.risk_bits==expected&&actual.risk_bits==ref&&actual.anchors==anchors_expected&&anchors==anchors_expected&&actual.blocked==(expected>threshold);
            failures+=!pass;
            std::cout<<"{\"case\":"<<c<<",\"risk_bits\":"<<actual.risk_bits<<",\"expected_risk_bits\":"<<expected<<",\"anchors\":"<<actual.anchors<<",\"error\":"<<actual.error<<",\"pass\":"<<(pass?"true":"false")<<"}\n";
        }
        if(p!=cases.size())throw std::runtime_error("Trailing case bytes");return failures?1:0;
    }catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 2;}
}
