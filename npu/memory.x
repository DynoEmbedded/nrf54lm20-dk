/* nRF54LM20B memory map (bare-metal, whole-chip).
 *
 * Confirmed from the nRF54LM20B MDK (nrf54lm20b_xxaa_application_memory.h):
 *   FLASH (RRAM): base 0x00000000, size 0x1FD000 = 2036 KB
 *   RAM:          two contiguous 256K banks (0x20000000 + 0x20040000) = 512 KB
 *
 * Adjust ORIGIN/LENGTH if you reserve space for a bootloader (e.g. MCUboot) or a
 * secure/non-secure split.
 *
 * NOTE: model constants/weights live in RRAM (FLASH region). The Axon engine
 * reads them during inference (see the RRAM note in src/platform.rs).
 */
MEMORY
{
  FLASH (rx) : ORIGIN = 0x00000000, LENGTH = 2036K
  RAM  (rwx) : ORIGIN = 0x20000000, LENGTH = 512K
}
