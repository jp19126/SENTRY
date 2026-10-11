#ifndef SENTRY_FAKE_VENDOR_H
#define SENTRY_FAKE_VENDOR_H
#include <stdint.h>
#include <stddef.h>
typedef uintptr_t UINTPTR;typedef intptr_t INTPTR;
#define XST_SUCCESS 0
typedef struct {UINTPTR BaseAddress;} XUartPsv_Config;
typedef struct {unsigned unused;} XUartPsv;
typedef struct {uint32_t BaseAddress;} XSdPs_Config;
typedef struct {unsigned HCS;} XSdPs;
XUartPsv_Config *XUartPsv_LookupConfig(UINTPTR);
int XUartPsv_CfgInitialize(XUartPsv *,XUartPsv_Config *,UINTPTR);
int XUartPsv_SetBaudRate(XUartPsv *,unsigned);
unsigned XUartPsv_Recv(XUartPsv *,uint8_t *,unsigned);
unsigned XUartPsv_Send(XUartPsv *,uint8_t *,unsigned);
int XUartPsv_IsSending(XUartPsv *);
XSdPs_Config *XSdPs_LookupConfig(uint32_t);
int XSdPs_CfgInitialize(XSdPs *,XSdPs_Config *,UINTPTR);
int XSdPs_CardInitialize(XSdPs *);
int XSdPs_ReadPolled(XSdPs *,uint32_t,unsigned,uint8_t *);
void Xil_DCacheFlushRange(INTPTR,unsigned);
void Xil_DCacheInvalidateRange(INTPTR,unsigned);
uint32_t Xil_In32(UINTPTR);
void Xil_Out32(UINTPTR,uint32_t);
typedef uint64_t XTime;
#define COUNTS_PER_SECOND 100000000ULL
void XTime_GetTime(XTime *);
#endif
