/* Required offline host protocol check. Fake devices provide no detector
 * arithmetic, physical timing, throughput, power or platform feasibility. */
#include "fake_vendor.h"
#include "../sentry_supervisor.h"
#include <assert.h>
#include <setjmp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static SentryDeployment d;
static uint8_t input[1+3*(64+9*512)],output[4*64];
static uint8_t *responses=output+64;
static size_t input_size,input_pos,response_pos;
static uint32_t regfile[64],maximum;
static unsigned starts,flushes,invalidates;
static int mode;static jmp_buf eof;
void XTime_GetTime(XTime *p){static XTime value;value+=100;*p=value;}
static uint32_t get32(const uint8_t *p){uint32_t x;memcpy(&x,p,4);return x;}
static void put32(uint8_t *p,uint32_t x){memcpy(p,&x,4);}
static void put64(uint8_t *p,uint64_t x){memcpy(p,&x,8);}
static uint64_t pair(unsigned n){return regfile[n/4]|((uint64_t)regfile[n/4+1]<<32);}
static void setpair(unsigned n,uint64_t x){regfile[n/4]=(uint32_t)x;regfile[n/4+1]=(uint32_t)(x>>32);}
XUartPsv_Config *XUartPsv_LookupConfig(UINTPTR n){static XUartPsv_Config c;c.BaseAddress=n;return &c;}
int XUartPsv_CfgInitialize(XUartPsv *u,XUartPsv_Config *c,UINTPTR n){(void)u;assert(c->BaseAddress==n);return 0;}
int XUartPsv_SetBaudRate(XUartPsv *u,unsigned n){(void)u;assert(n==115200);return 0;}
unsigned XUartPsv_Recv(XUartPsv *u,uint8_t *p,unsigned n){(void)u;if(input_pos==input_size)longjmp(eof,1);if(n>7)n=7;if(n>input_size-input_pos)n=(unsigned)(input_size-input_pos);memcpy(p,input+input_pos,n);input_pos+=n;return n;}
unsigned XUartPsv_Send(XUartPsv *u,uint8_t *p,unsigned n){(void)u;if(n>11)n=11;assert(response_pos+n<=sizeof output);memcpy(output+response_pos,p,n);response_pos+=n;return n;}
int XUartPsv_IsSending(XUartPsv *u){(void)u;return 0;}
XSdPs_Config *XSdPs_LookupConfig(uint32_t n){static XSdPs_Config c;c.BaseAddress=n;return &c;}
int XSdPs_CfgInitialize(XSdPs *s,XSdPs_Config *c,UINTPTR n){assert(c->BaseAddress==n);s->HCS=mode!=7;return 0;}
int XSdPs_CardInitialize(XSdPs *s){(void)s;return 0;}
int XSdPs_ReadPolled(XSdPs *s,uint32_t sector,unsigned count,uint8_t *p){if(!s->HCS){assert(!(sector%512));sector/=512;}assert(count<=32);memset(p,0,count*512);if(sector==d.arena_sd_lba){assert(count==1);memcpy(p,"SENTRYA1",8);put32(p+8,1);put32(p+12,d.tokens);put64(p+16,d.model_id);put32(p+24,d.w8_mask);put32(p+28,d.threshold_bits);put32(p+32,d.arena_bytes);if(mode==6)p[16]^=1;}else assert(sector>=d.arena_sd_lba+1&&sector+count<=d.arena_sd_lba+1+d.arena_bytes/512);return 0;}
void Xil_DCacheFlushRange(INTPTR p,unsigned n){assert(p&&n);flushes++;}
void Xil_DCacheInvalidateRange(INTPTR p,unsigned n){assert(p&&n);invalidates++;}
uint32_t Xil_In32(UINTPTR address){assert(address>=d.mailbox_base&&address<d.mailbox_base+256);return regfile[(address-d.mailbox_base)/4];}
void Xil_Out32(UINTPTR address,uint32_t x){unsigned off=(unsigned)(address-d.mailbox_base);assert(off<256&&!(off%4));regfile[off/4]=x;if(off)return;
 if(x==1){assert(regfile[0x10/4]==d.w8_mask&&regfile[0x14/4]==d.threshold_bits&&pair(0x18)==d.arena_base&&pair(0x20)==d.model_id);regfile[1]=18;}
 if(x==2){assert(regfile[1]==18);unsigned index=regfile[0x40/4],valid=regfile[0x44/4];uint32_t *ids=(uint32_t *)(uintptr_t)(d.arena_base+d.token_offset),*types=(uint32_t *)(uintptr_t)(d.arena_base+d.type_offset);uint8_t *mask=(uint8_t *)(uintptr_t)(d.arena_base+d.mask_offset);for(unsigned i=0;i<d.tokens;i++)assert(ids[i]==(i<valid?101U:0U)&&types[i]==0&&mask[i]==(i<valid));starts++;if(mode==5){regfile[1]=24;return;}uint32_t risk=starts==1?d.threshold_bits-1:starts==2?d.threshold_bits+1:d.threshold_bits;if(!index)maximum=risk;else if(risk>maximum)maximum=risk;regfile[0x80/4]=mode==4?1:0;regfile[0x84/4]=risk;regfile[0x88/4]=maximum;regfile[0x8c/4]=regfile[0x48/4]|((regfile[0x48/4]&&maximum>d.threshold_bits)?2:0);setpair(0x90,pair(0x30)+(mode==1));setpair(0x98,pair(0x38));regfile[0xa0/4]=index;setpair(0xa4,pair(0x20));regfile[1]=20;}
 if(x==4)regfile[1]=18;
}
static void packet(unsigned slot,uint64_t doc,unsigned index,unsigned last){uint8_t *p=input+1+slot*(64+9*d.tokens);memset(p,0,64+9*d.tokens);memcpy(p,"SENTRY01",8);put32(p+8,1);put32(p+12,d.tokens);put32(p+16,3);put32(p+20,index);put64(p+24,doc);put64(p+32,100+slot);put64(p+40,d.model_id);put32(p+48,last);put32(p+52,9*d.tokens);put32(p+56,d.threshold_bits);for(unsigned i=0;i<3;i++){put32(p+64+4*i,101);p[64+8*d.tokens+i]=1;}}
static int run_supervisor(void){if(setjmp(eof))return 777;return sentry_supervisor(&d);}
int main(int argc,char **argv){void *arena=aligned_alloc(4096,67108864);assert(arena);d=(SentryDeployment){.mailbox_base=0xa4000000,.arena_base=(uintptr_t)arena,.uart_base=0xff000000,.sd_base=0xf1050000,.model_id=0xbf04010000000001ULL,.mailbox_poll_limit=8,.tokens=256,.arena_bytes=67108864,.arena_sd_lba=2048,.token_offset=4096,.type_offset=8192,.mask_offset=12288,.threshold_bits=0x3b955a95,.w8_mask=65535};
 const char *names[]={"two_document_max_reset","captured_identity","request_order","invalid_payload","engine_error","completion_timeout","cold_model_identity","sdsc_addressing"};int expected[]={777,-15,-11,-12,-16,-14,-17,777};
 for(unsigned length=128;length<=512;length*=2){d.tokens=length;for(mode=0;mode<8;mode++){memset(regfile,0,sizeof regfile);regfile[1]=1;maximum=0;starts=flushes=invalidates=0;input_pos=response_pos=0;input_size=1+3*(64+9*d.tokens);input[0]=0xa5;packet(0,10,0,0);packet(1,10,1,1);packet(2,11,0,1);if(mode==2)put32(input+21,1);if(mode==3)put32(input+65,30522);int result=run_supervisor();assert(result==expected[mode]);if(mode==0||mode==7){assert(starts==3&&response_pos==256);assert(!memcmp(output,"SENTRYR1",8)&&get32(output+8)==1&&get32(output+12)==d.tokens&&get32(output+24)==d.threshold_bits&&get32(output+28)==COUNTS_PER_SECOND);for(unsigned i=0;i<3;i++){uint8_t *r=responses+64*i;assert(!memcmp(r,"SENTRES1",8)&&get32(r+8)==1&&get32(r+12)==0&&get32(r+40)==(i==1));assert(get32(r+44)==(i==0?d.threshold_bits-1:i==1?d.threshold_bits+1:d.threshold_bits));assert(get32(r+48)==(i==0?d.threshold_bits-1:i==1?d.threshold_bits+1:d.threshold_bits));assert(get32(r+52)==(i==0?0:i==1?3:1));assert(get32(r+56)==1000&&get32(r+60)==0);}assert(invalidates==4097&&flushes>=8200);if(mode==0&&argc==2){char path[1024];assert(snprintf(path,sizeof path,"%s_l%u.bin",argv[1],length)>0);FILE *f=fopen(path,"wb");assert(f&&fwrite(output,1,256,f)==256);assert(!fclose(f));}}if(mode==4)assert(response_pos==128&&get32(responses+12)==1);printf("PASS L%u %s result=%d starts=%u responses=%zu\n",length,names[mode],result,starts,response_pos/64);}}
 free(arena);puts("HOST_PROTOCOL_CHECK_PASS arithmetic_evidence=false physical_measurement=false");return 0;
}
