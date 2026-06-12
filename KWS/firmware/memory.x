/* nRF54LM20B memory map (bare-metal, whole-chip).
 *
 * From the nRF54LM20B MDK (nrf54lm20b_xxaa_application_memory.h):
 *   FLASH (RRAM): base 0x00000000, size 0x1FD000 = 2036 KB
 *   RAM:          two contiguous 256K banks (0x20000000 + 0x20040000) = 512 KB
 *
 * Model constants/weights live in RRAM (FLASH region); the Axon engine reads
 * them during inference.
 *
 * RAM is declared as 510K, NOT the headline 512K: the last ~512 B
 * (0x2007FF00..0x20080000) bus-fault on this silicon (verified via SWD reads
 * on the DK; protected KMU/reserved words at the top of RAM). cortex-m-rt
 * places the initial stack pointer at the END of RAM, so declaring the full
 * 512K hard-faults on the first stack push, before main() ever runs.
 */
MEMORY
{
  FLASH (rx) : ORIGIN = 0x00000000, LENGTH = 2036K
  RAM  (rwx) : ORIGIN = 0x20000000, LENGTH = 510K
}
