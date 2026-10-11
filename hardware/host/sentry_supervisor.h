#ifndef SENTRY_SUPERVISOR_H
#define SENTRY_SUPERVISOR_H
#include <stdint.h>
/* Values are generated from the selected XSA, plan and frozen arena record.
 * No physical addresses or media layout are defaulted by this interface. */
typedef struct {
    uint64_t mailbox_base, arena_base, uart_base, sd_base;
    uint64_t model_id, mailbox_poll_limit;
    uint32_t tokens, arena_bytes, arena_sd_lba;
    uint32_t token_offset, type_offset, mask_offset, threshold_bits;
    uint16_t w8_mask;
} SentryDeployment;
int sentry_supervisor(const SentryDeployment *deployment);
#endif
