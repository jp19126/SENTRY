/* Bare-metal Cortex-A72 supervisor. No detector arithmetic runs on this CPU.
 * The generated XSA/BSP supplies all physical addresses at deployment build.
 * Offline functional verification uses narrow fake peripheral adapters.
 */
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include "xuartpsv.h"
#include "xsdps.h"
#include "xil_cache.h"
#include "xil_io.h"
#include "xstatus.h"
#include "xiltimer.h"
#include "sentry_supervisor.h"

static XUartPsv uart;
static XSdPs sd;
static uint8_t sd_chunk[16384] __attribute__((aligned(64)));
static uint8_t request[64+9*512] __attribute__((aligned(64)));
static uint32_t le32(const uint8_t *p){return (uint32_t)p[0]|(uint32_t)p[1]<<8|(uint32_t)p[2]<<16|(uint32_t)p[3]<<24;}
static uint64_t le64(const uint8_t *p){return le32(p)|((uint64_t)le32(p+4)<<32);}
static void put32(uint8_t *p,uint32_t x){for(unsigned i=0;i<4;i++)p[i]=(uint8_t)(x>>(8*i));}
static void put64(uint8_t *p,uint64_t x){put32(p,(uint32_t)x);put32(p+4,(uint32_t)(x>>32));}
static void barrier(void){
#ifdef SENTRY_HOST_OFFLINE
    __asm__ volatile("" ::: "memory");
#else
    __asm__ volatile("dsb sy" ::: "memory");
#endif
}
static uint32_t rd(const SentryDeployment *d,unsigned off){return Xil_In32((UINTPTR)(d->mailbox_base+off));}
static void wr(const SentryDeployment *d,unsigned off,uint32_t x){Xil_Out32((UINTPTR)(d->mailbox_base+off),x);}
static void wr64(const SentryDeployment *d,unsigned off,uint64_t x){wr(d,off,(uint32_t)x);wr(d,off+4,(uint32_t)(x>>32));}
static uint64_t rd64(const SentryDeployment *d,unsigned off){return rd(d,off)|((uint64_t)rd(d,off+4)<<32);}
static int receive(uint8_t *p,unsigned size){unsigned n=0;while(n<size)n+=XUartPsv_Recv(&uart,p+n,size-n);return 0;}
static void send(uint8_t *p,unsigned size){unsigned n=0;while(n<size)n+=XUartPsv_Send(&uart,p+n,size-n);while(XUartPsv_IsSending(&uart)){} }
static int wait_status(const SentryDeployment *d,uint32_t mask,uint32_t wanted){
    /* A finite supervisor iteration cap is a failure bound, not a timing measurement. */
    for(uint64_t n=0;n<d->mailbox_poll_limit;n++)if((rd(d,4)&mask)==wanted)return 0;
    return -1;
}
static int initialize(const SentryDeployment *d){
    XUartPsv_Config *uc=XUartPsv_LookupConfig((UINTPTR)d->uart_base);
    XSdPs_Config *sc=XSdPs_LookupConfig((uint32_t)d->sd_base);
    if(!uc||!sc||XUartPsv_CfgInitialize(&uart,uc,uc->BaseAddress)!=XST_SUCCESS||
       XUartPsv_SetBaudRate(&uart,115200)!=XST_SUCCESS||
       XSdPs_CfgInitialize(&sd,sc,sc->BaseAddress)!=XST_SUCCESS||
       XSdPs_CardInitialize(&sd)!=XST_SUCCESS)return -2;
    uint32_t meta_sector=d->arena_sd_lba;
    if(!sd.HCS)meta_sector*=512;
    Xil_DCacheFlushRange((INTPTR)sd_chunk,sizeof sd_chunk);barrier();
    if(XSdPs_ReadPolled(&sd,meta_sector,1,sd_chunk)!=XST_SUCCESS)return -3;
    Xil_DCacheInvalidateRange((INTPTR)sd_chunk,512);
    if(memcmp(sd_chunk,"SENTRYA1",8)||le32(sd_chunk+8)!=1||le32(sd_chunk+12)!=d->tokens||
       le64(sd_chunk+16)!=d->model_id||le32(sd_chunk+24)!=d->w8_mask||
       le32(sd_chunk+28)!=d->threshold_bits||le32(sd_chunk+32)!=d->arena_bytes)return -17;
    /* Read-only raw SD slot. One metadata sector precedes the matching arena.
     * Bounce through OCM so SD DMA need not address the fabric-memory aperture. */
    for(uint32_t off=0;off<d->arena_bytes;off+=sizeof sd_chunk){
        uint32_t bytes=d->arena_bytes-off;if(bytes>sizeof sd_chunk)bytes=sizeof sd_chunk;
        uint32_t sector=d->arena_sd_lba+1+off/512;
        if(!sd.HCS)sector*=512;
        Xil_DCacheFlushRange((INTPTR)sd_chunk,sizeof sd_chunk);barrier();
        if(XSdPs_ReadPolled(&sd,sector,bytes/512,sd_chunk)!=XST_SUCCESS)return -3;
        Xil_DCacheInvalidateRange((INTPTR)sd_chunk,bytes);
        memcpy((void *)(UINTPTR)(d->arena_base+off),sd_chunk,bytes);
        Xil_DCacheFlushRange((INTPTR)(d->arena_base+off),bytes);
    }
    barrier();
    if(wait_status(d,1,1))return -4;
    wr(d,0x10,d->w8_mask);wr(d,0x14,d->threshold_bits);
    wr64(d,0x18,d->arena_base);wr64(d,0x20,d->model_id);barrier();wr(d,0,1);
    if(wait_status(d,16,16))return -5;
    return 0;
}
int sentry_supervisor(const SentryDeployment *d){
    if(!d || (d->tokens!=128&&d->tokens!=256&&d->tokens!=512) ||
       d->arena_bytes!=67108864 || (d->arena_base&4095) || !d->mailbox_poll_limit ||
       d->threshold_bits>0x3f800000U || d->token_offset+4*d->tokens>d->arena_bytes ||
       d->type_offset+4*d->tokens>d->arena_bytes || d->mask_offset+d->tokens>d->arena_bytes)return -10;
    int init=initialize(d);if(init)return init;
    /* One-byte hello can wait safely during the SD load. The host waits for
     * this identity before transmitting a full packet; no cold-load guess. */
    uint8_t hello;receive(&hello,1);if(hello!=0xa5)return -18;
    uint8_t ready[64]={0};memcpy(ready,"SENTRYR1",8);put32(ready+8,1);
    put32(ready+12,d->tokens);put64(ready+16,d->model_id);
    put32(ready+24,d->threshold_bits);put32(ready+28,COUNTS_PER_SECOND);
    send(ready,sizeof ready);
    uint64_t document=0;uint32_t next_window=0;int active=0;
    for(;;){
        receive(request,64);
        uint32_t valid=le32(request+16),index=le32(request+20),flags=le32(request+48);
        uint64_t doc=le64(request+24),sample=le64(request+32),model=le64(request+40);
        if(memcmp(request,"SENTRY01",8)||le32(request+8)!=1||le32(request+12)!=d->tokens||
           valid<1||valid>d->tokens||flags>1||le32(request+52)!=9*d->tokens||
           le32(request+56)!=d->threshold_bits||le32(request+60)!=0||model!=d->model_id||
           (active?(doc!=document||index!=next_window):index!=0))return -11;
        receive(request+64,9*d->tokens);
        for(uint32_t i=0;i<d->tokens;i++){
            uint32_t id=le32(request+64+4*i),type=le32(request+64+4*d->tokens+4*i);
            uint8_t mask=request[64+8*d->tokens+i];
            if(id>=30522||type>1||mask!=(i<valid?1:0))return -12;
        }
        if(wait_status(d,14,2))return -13; /* idle, no captured result, window_ready */
        memcpy((void *)(UINTPTR)(d->arena_base+d->token_offset),request+64,4*d->tokens);
        memcpy((void *)(UINTPTR)(d->arena_base+d->type_offset),request+64+4*d->tokens,4*d->tokens);
        memcpy((void *)(UINTPTR)(d->arena_base+d->mask_offset),request+64+8*d->tokens,d->tokens);
        Xil_DCacheFlushRange((INTPTR)(d->arena_base+d->token_offset),4*d->tokens);
        Xil_DCacheFlushRange((INTPTR)(d->arena_base+d->type_offset),4*d->tokens);
        Xil_DCacheFlushRange((INTPTR)(d->arena_base+d->mask_offset),d->tokens);
        wr64(d,0x30,sample);wr64(d,0x38,doc);wr(d,0x40,index);wr(d,0x44,valid);wr(d,0x48,flags);
        barrier();XTime kernel_begin,kernel_end;XTime_GetTime(&kernel_begin);wr(d,0,2);
        if(wait_status(d,4,4))return -14;
        XTime_GetTime(&kernel_end);
        /* Complete PL check plus A72 submit/poll overhead; excludes UART,
         * input copying and cache maintenance. BSP frequency is recorded. */
        uint64_t ticks=kernel_end-kernel_begin;
        uint64_t kernel_ns=(ticks/COUNTS_PER_SECOND)*1000000000ULL+
            ((ticks%COUNTS_PER_SECOND)*1000000000ULL)/COUNTS_PER_SECOND;
        uint32_t status=rd(d,0x80),result_flags=rd(d,0x8c);
        if(rd64(d,0x90)!=sample||rd64(d,0x98)!=doc||rd(d,0xa0)!=index||
           rd64(d,0xa4)!=model||(result_flags&1)!=flags)return -15;
        uint8_t response[64]={0};memcpy(response,"SENTRES1",8);put32(response+8,1);put32(response+12,status);
        put64(response+16,sample);put64(response+24,doc);put64(response+32,model);put32(response+40,index);
        put32(response+44,rd(d,0x84));put32(response+48,rd(d,0x88));put32(response+52,result_flags);put64(response+56,kernel_ns);
        send(response,sizeof response);barrier();wr(d,0,4);
        if(status)return -16; /* sequencer errors need explicit reset before reuse */
        active=!flags;document=doc;next_window=index+1;
    }
}
