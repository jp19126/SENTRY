#define main existing_driver_tests
#include "host_test.cpp"
#undef main
#include "host_service.hpp"
#include "host_sha256.hpp"
#include "deployment.hpp"
struct Stream {std::vector<uint8_t> input,output;size_t at=0;};
static bool recv_wire(void*p,uint8_t*d,size_t n){auto&s=*(Stream*)p;if(n>s.input.size()-s.at)return false;std::memcpy(d,s.input.data()+s.at,n);s.at+=n;return true;}
static bool send_wire(void*p,const uint8_t*d,size_t n){auto&s=*(Stream*)p;s.output.insert(s.output.end(),d,d+n);return true;}
static uint32_t get32(const uint8_t*p){return uint32_t(p[0])|uint32_t(p[1])<<8|uint32_t(p[2])<<16|uint32_t(p[3])<<24;}
int main(int argc,char**argv){
 if(argc!=3)return 2;std::ifstream input(argv[1],std::ios::binary);std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(input)),{});int32_t margin=std::stoi(argv[2]);
 const char*messages[]={"","abc","abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"};
 const char*expected[]={"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad","248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"};
 for(unsigned t=0;t<3;++t){Sha256 h;uint8_t digest[32];for(const char*p=messages[t];*p;++p)h.update(p,1);h.finish(digest);char hex[65];for(unsigned i=0;i<32;++i)std::snprintf(hex+2*i,3,"%02x",digest[i]);if(std::strcmp(hex,expected[t]))return 3;}
 for(int mode=0;mode<9;++mode){
 Stream stream{bytes,{}};int32_t selected=mode==1?margin-1:margin;for(unsigned i=0;i<4;++i)stream.input[kdcnn::PARAM_BYTES+48+i]=uint8_t(uint32_t(selected)>>(8*i));Sha256 request_hash;request_hash.update(stream.input.data()+kdcnn::PARAM_BYTES,52);request_hash.update(stream.input.data()+kdcnn::PARAM_BYTES+84,stream.input.size()-kdcnn::PARAM_BYTES-84);request_hash.finish(stream.input.data()+kdcnn::PARAM_BYTES+52);if(mode==2)stream.input[0]^=1;if(mode==3)stream.input.back()^=1;if(mode==4)stream.input.resize(stream.input.size()-1);if(mode==5)stream.input[kdcnn::PARAM_BYTES+8+2]=0xff;
 if(mode==7)stream.input[kdcnn::PARAM_BYTES+16]^=1;if(mode==8)stream.input[kdcnn::PARAM_BYTES+48]^=1;
 Fake f;Platform io={&f,wr,rd,memwrite,memread,flush,invalidate,barrier,release};Driver driver(io,regs,buffers,CONTROL);
 std::vector<uint8_t> model(kdcnn::PARAM_BYTES),raw(8),masks(17*256);std::vector<uint16_t> ids(17*256);
 GuardService guard({&stream,recv_wire,send_wire},{model.data(),raw.data(),ids.data(),masks.data(),8,17},driver,mode==1?margin-1:margin,8);
 if(mode==6)f.timeout=true;
 bool ready=guard.receive_model(deployment_model_sha);if(ready&&(std::memcmp(stream.output.data()+16,deployment_model_sha,32)||get32(stream.output.data()+48)!=uint32_t(selected)))return 8;bool done=ready&&guard.document();
 if(mode<2){if(!done||stream.output.size()!=size_t(mode==1?128:136)||get32(stream.output.data()+64+8)!=uint32_t(mode==1)||f.releases!=1||f.calls!=2||f.saw_partial_release)return 4;if(mode==0&&std::memcmp(stream.output.data()+128,"retained",8))return 5;}
 else {if(done||f.releases||stream.output.size()!=size_t(mode==2?64:128)||get32(stream.output.data()+stream.output.size()-56)!=2||guard.document())return 6;if(mode!=6&&f.calls)return 7;}
 }
 std::cout<<"{\"passed\":true,\"sha_vectors\":3,\"service_scenarios\":9,\"windows_success_case\":17,\"chunks\":2,\"raw_emitted_only_allow\":true,\"model_digest_checked\":true,\"document_digest_checked\":true,\"timeout_retained\":true,\"evidence\":\"native service plus fake MMIO invoking full C CNN; no UART/board execution\"}\n";
}
