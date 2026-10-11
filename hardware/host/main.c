/* deployment.h is generated only from the selected system XSA/address map,
 * frozen arena and command plan. It defines const SentryDeployment deployment. */
#include "sentry_supervisor.h"
#include "deployment.h"
#include "xil_cache.h"
int main(void){
    Xil_ICacheEnable();Xil_DCacheEnable();
    volatile int failure=sentry_supervisor(&deployment);
    (void)failure;
    /* Retain failed state; do not silently reset/reuse an invalid document. */
    for(;;)__asm__ volatile("wfe");
}
