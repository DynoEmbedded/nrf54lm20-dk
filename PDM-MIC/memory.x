/* nRF54LM20B memory map (bare-metal application core).
 *
 * From the nRF54LM20B MDK (nrf54lm20b_xxaa_application_memory.h; identical to the
 * A variant):
 *   FLASH (RRAM): base 0x00000000, size 0x1FD000 = 2036 KB
 *   RAM:          0x20000000 (256K) + RAM2 0x20040000 (256K), contiguous = 512 KB
 *
 * We only need a small fraction; declaring the first 256K bank is plenty and keeps
 * EasyDMA buffers in the lower bank. Bump to 512K if you need it.
 */
MEMORY
{
  FLASH (rx) : ORIGIN = 0x00000000, LENGTH = 2036K
  RAM  (rwx) : ORIGIN = 0x20000000, LENGTH = 256K
}
