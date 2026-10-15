#include "host_a72.hpp"
#include "host_service.hpp"
#include "deployment.hpp"
#include "xuartpsv.h"
#include "xil_cache.h"
#include "xstatus.h"
#include "xparameters.h"
static XUartPsv uart;
static bool receive(void*,uint8_t*p,size_t n){uint32_t idle=0;while(n){unsigned ask=n>4096?4096:unsigned(n),got=XUartPsv_Recv(&uart,p,ask);if(got){p+=got;n-=got;idle=0;}else if(++idle==100000000)return false;}return true;}
static bool send(void*,const uint8_t*p,size_t n){uint32_t idle=0;while(n){unsigned ask=n>4096?4096:unsigned(n),got=XUartPsv_Send(&uart,const_cast<uint8_t*>(p),ask);if(got){p+=got;n-=got;idle=0;}else if(++idle==100000000)return false;}for(uint32_t i=0;i<100000000;++i)if(!XUartPsv_IsSending(&uart))return true;return false;}
// Driver finalization notifies this owner; GuardService alone forwards raw bytes.
static void retain_owner(void*,const uint8_t*,size_t,bool){}
int main(){
 Xil_ICacheEnable();Xil_DCacheEnable();auto *config=XUartPsv_LookupConfig(XPAR_XUARTPSV_0_BASEADDR);
 if(!config||XUartPsv_CfgInitialize(&uart,config,config->BaseAddress)!=XST_SUCCESS||XUartPsv_SetBaudRate(&uart,115200)!=XST_SUCCESS)return 1;
 kdhost::A72Context context{0x40000000,0x04000000,nullptr,retain_owner};
 kdhost::Driver driver(kdhost::a72_platform(context),kdhost::synthesized_registers,{0x40000000,0x40300000,0x40302000,0x40304000},0xA4000000);
 kdhost::Storage data{(uint8_t*)0x41000000,(uint8_t*)0x41400000,(uint16_t*)0x41500000,(uint8_t*)0x41700000,1048576,4096};
 kdhost::GuardService guard({nullptr,receive,send},data,driver,deployment_threshold,1000000000);
 if(guard.receive_model(deployment_model_sha))while(guard.document()){}
 // Never reset or overwrite an unresolved request automatically.
 for(;;)asm volatile("wfe");
}
