# Accessing the Axon NPU (nRF54LM20B) from Rust

## What the NPU is

The "NPU" is Nordic's **Axon** block. It is a **command-buffer executor**, not a
general-purpose / GPGPU compute unit. You do not write kernels for it. You express
compute as an **int8-quantized TFLite model**, compile it offline, and the firmware
hands the resulting command buffer to the driver.

Flow:
TFLite (int8) -> Axon Compiler -> generated C header -> link into firmware -> infer.

The Axon Compiler (`tools/axon/compiler`, Python + a host static lib, runs in Docker)
emits a C header, e.g. `nrf_axon_model_<name>_.h`, containing:
- a const struct of weights/biases,
- `cmd_buffer_<name>[]` (the compiled "Axon machine code"),
- a `nrf_axon_nn_compiled_model_s model_<name>` descriptor.

Firmware then calls `nrf_axon_nn_model_infer_sync()` / `_infer_async()`.

## Why there is no pure-Rust register driver / "direct" access

"Direct" access would mean defeating TWO undocumented layers:

- **Register interface.** The product spec AXONS page
  (docs.nordicsemi.com/r/bundle/ps_nrf54LM20A/page/axons.html) documents ONLY two
  registers: `ENABLE` and `STATUS`. The cmd-buffer pointer / start register is NOT
  public. So you can power it on and read busy/done, but cannot tell it what to do.
- **Command-buffer ISA.** The work is an opaque stream of 32-bit words (compiled
  "Axon machine code", e.g. `0x1fff0044, 0x02000080, ...` with buffer addresses
  patched at link time). Encoding is proprietary; only the (blob) compiler emits it.

- The Rust PAC `nrf-pac` has an `nrf54lm20a` target but it exposes **no AXON
  peripheral** (not in the public SVD). So no register-level access from Rust
  without reverse-engineering both the register map and the opaque cmd-buffer ISA.
- The driver blob also encodes silicon errata workarounds (changelog: "1.2.1 Remove
  RRAM work-around") -- a moving target (driver May 2026, chip GA Q2 2026).

Semi-direct option: keep only the host compiler (to make cmd buffers), then from
Rust write ENABLE / DMA cmd buffer / kick / poll STATUS -- drops the *runtime* blob
but still needs the (non-public) cmd-buffer register + inherits link-time address
patching. A reverse-engineering project, not configuration.

Conclusion: the C driver IS the programming model. FFI into it = the direct access.
- The low-level driver ships as a **precompiled blob**:
  `lib/axon/bin/arm/libnrf-axon-driver-internal.a` (+ `-fpu` variant), license
  `LicenseRef-Nordic-5-Clause` (proprietary, usable on Nordic devices).
- Header `nrf_axon_driver.h` states explicitly: "for the pre-compiled driver library".

Conclusion: a pure-Rust driver is off the table. Rust access = **FFI into Nordic's C**.

## The C ABI you bind to (clean, bindgen-friendly)

High level (`include/drivers/axon/nrf_axon_nn_infer.h`):
- `nrf_axon_nn_model_validate(model)`
- `nrf_axon_nn_model_infer_sync(model, input_vec, out_buf)`
- `nrf_axon_nn_model_async_init` / `nrf_axon_nn_model_infer_async`
- `nrf_axon_nn_get_classification(...)`

Low level (`nrf_axon_driver.h`, for non-NN math / DSP intrinsics):
- `nrf_axon_init_command_buffer_info`, `nrf_axon_run_cmd_buf_sync`, `nrf_axon_queue_cmd_buf`
- plus `nrf_axon_dsp_intrinsics.h` (FIR, matrix mult, etc.) -- closest thing to
  "use the NPU as a raw compute accelerator".

Platform glue you must provide (`nrf_axon_platform_interface.h`):
- `nrf_axon_driver_init(base_addr)` -- AXON must be powered + clocked first
- `nrf_axon_driver_power_on/off`, IRQ handler, reserve/free (mutex), event signalling.

On Zephyr this is implemented in `lib/axon/platform/src/zephyr/nrf_axon_platform_zephyr.c`:
base addr + IRQ from devicetree `&axon` node, `IRQ_CONNECT`, `nrf_sys_event_register`
(keeps RRAM powered during inference), `onoff_manager` for power, semaphores for
reservation. The `&axon` DT node/binding comes from `sdk-nrf` v3.3.0.

## Rust options, ranked

1. **NCS/Zephyr app, Rust application logic via FFI (recommended).**
   Build the Edge AI add-on app in NCS (C driver blob + Zephyr platform layer +
   generated model). Write app logic in Rust using Zephyr's Rust support
   (`zephyr-lang-rust`), calling the C inference API via `bindgen`. Zephyr handles
   power/clock/IRQ/DT. Only path that gets the NPU running quickly.

2. **Bare-metal Rust (embassy/cortex-m) + FFI to the driver blob. [chosen path -- Zephyr ruled out]**
   The driver blob is **RTOS-agnostic** (verified: `nm libnrf-axon-driver-internal-fpu.a`
   has NO Zephyr symbols; only undefined deps are `memcpy`/`memset` + the ~10
   `nrf_axon_platform_*` shims you implement). Zephyr is just one platform impl.
   - Power/clock bring-up is trivial (from `nrf_axon_platform_zephyr.c`):
     `*(uint32_t*)(AXON_BASE_ADDR + 0x400) |= 1;` to enable (ENABLE @ base+0x400),
     plus a vote to keep RRAM in standby (`nrf_sys_event_register(0,true)` on Zephyr;
     on bare-metal just don't power-gate RRAM, or replicate the POWER/MEMORY poke).
   - Platform shims: interrupt mask -> PRIMASK; reserve/free -> trivial (single owner);
     user/driver events -> flag spin, or call `nrf_axon_process_driver_event()` directly
     (header permits this bare-metal). Use SYNC + POLLING mode to avoid IRQ wiring.
   - Link the `.a` + repo source `nrf_axon_nn_infer.c` / `nrf_axon_nn_op_extensions.c`
     (these define the remaining internal symbols). bindgen the headers.
   - Two real unknowns: AXONS base address (preliminary datasheet v0.8); exact RRAM
     standby vote. Everything else mechanical. Keeps 2 blobs: runtime driver .a +
     host compiler. Drops Zephyr, NOT the driver.

   Why not fully blob-free: ENABLE/STATUS are public, but the cmd-buffer submission
   register + the instruction ISA are not (Nordic versions HW+compiler+driver
   together; datasheet is preliminary; their own source asks "@fixme will this be in
   the public mdk?"). The driver IS the only thing that knows those.

3. **Pure-Rust register driver.** Not feasible now (no public AXON registers, opaque ISA).

## Bare-metal Rust scaffold (built & linking)

Crate builds for `thumbv8m.main-none-eabihf`, links the full FFI graph with ZERO
undefined symbols (verified `arm-none-eabi-nm -u`). Key wiring:

- `build.rs`: `cc` (arm-none-eabi-gcc, flags derived from target -- do NOT also pass
  `-mcpu`, it conflicts with cc's `-march=armv8-m.main+fp`) compiles the two vendored
  `.c`; links `vendor/lib/libnrf-axon-driver-internal-fpu.a`; bindgen with
  `EnumVariation::NewType` so result enums are `repr(transparent)` i32 (compare `.0`).
- bindgen needs `nrf_axon_platform_interface.h` in `wrapper.h` -- that's where
  `nrf_axon_driver_init`/`_power_on`/`_process_driver_event` are declared (NOT in
  nrf_axon_platform.h).
- libc/libm deps of the C: `memcpy`/`memset` + `__aeabi_*` come from Rust
  `compiler_builtins`; `exp/expf/round/roundf` shimmed from the `libm` crate
  (`src/libm_shims.rs`) -- no newlib linked.
- Axon buffers (`nrf_axon_interlayer_buffer`/`_psum_buffer`) defined as `#[no_mangle]`
  Rust statics in `main.rs`; C side only `extern`-declares them (size via `-D`).
- Variadic `nrf_axon_platform_printf` lives in `csrc/glue.c` (stable Rust can't define
  variadic fns); the other ~10 platform shims are Rust in `src/platform.rs`.
- `--gc-sections` strips uncalled API; `main.rs` uses `black_box(fn as *const ())`
  link anchors to retain + prove the inference path links.

### Hardware values (confirmed from nrfx MDK 8.75.3, `nrf54lm20b_*` headers)

- AXONS base: `NRF_AXONS_S_BASE = 0x50056000` (secure), NS alias `0x40056000`.
  Use secure -- CPU boots secure without TrustZone/SPU setup.
- ENABLE register @ offset `0x400`, EN = bit 0 (`AXONS_ENABLE_EN_Msk = 0x1`);
  STATUS @ `0x404`. (NRF_AXONS_Type = RESERVED[256] then ENABLE, STATUS.)
- `AXONS_IRQn = 86` (only needed for async/IRQ mode; polling path doesn't use it).
- Memory map (`nrf54lm20b_xxaa_application_memory.h`): FLASH/RRAM base `0x00000000`
  size `0x1FD000` = 2036 KB; RAM two contiguous 256K banks (`0x20000000` +
  `0x20040000`) = 512 KB. memory.x matches.
- SVD (`nrf54lm20b_application.svd`) includes an AXONS peripheral but only
  ENABLE/STATUS -- a generated svd2rust PAC would expose those, NOT the
  command-buffer/ISA registers (consistent with the datasheet).
- RRAM standby: active after reset (code runs from RRAM), so no action needed
  unless entering low-power modes that power-gate it.

Both former TODOs are now resolved in src/platform.rs / memory.x.

## Model compile flow (TFLite -> Axon header -> linked)

- Axon Compiler is a containerized python3.11 + tensorflow==2.19 tool
  (`tools/axon-compiler/`) wrapping a cffi-loaded host `.so`
  (`bin/Linux/libnrf-axon-nn-compiler-lib-amd64.so`). Input: a YAML pointing at an
  int8 `.tflite`. Output: `nrf_axon_model_<name>_.h` (weights + cmd buffer +
  `model_<name>` descriptor) into the mounted workspace.
- `tools/compile-model.sh <model.tflite> <name> [il] [psum]` writes the YAML,
  builds the image, runs the container, installs the header into
  `vendor/include/generated/`. Engine defaults to **podman**
  (`CONTAINER_ENGINE=docker` overrides). For rootless podman it uses
  `--userns=keep-id` + `:z` volume relabel so the container can write the output
  header back as our uid.
- Podman caveat (this env): VS Code's snap terminal sets
  `XDG_DATA_HOME=.../snap/code/243/.local/share`, so podman misses its real
  storage DB ("database configuration mismatch"). The script resets
  `XDG_DATA_HOME=$HOME/.local/share` when it sees a `/snap/` path. Verified podman
  works after that (rootless, overlay).
- First real compile DONE (podman, this env): built the image (TF 2.19, ~645 MB),
  generated a small int8 sine model in-container (`models/gen_sine_model.py` ->
  `sine.tflite`), compiled it -> 3 FC layers, cmd_buffer 1848 B, constants 420 B,
  interlayer needed 84 B. Linked into firmware ("Axon model linked: sine"): final
  ELF has `model_sine` + `cmd_buffer_sine` (.rodata), zero undefined symbols.
- Gotchas fixed in compile-model.sh: (a) the compiler writes outputs under
  `<workspace>/outputs/nrf_axon_model_<name>_.h` (NOT the workspace root); (b)
  copying a model already in models/ onto itself (cp same-file) -> guard added.
- The bundled sample yamls reference `.tflite` files (e.g. `kws_ref_model.tflite`)
  that are not shipped, so they can't be compiled as-is; bring your own model.
- The generated header is NOT self-contained: it relies on the includer for the
  Axon headers + `<assert.h>` (static_assert) + `<stddef.h>` (NULL), and the
  `NRF_AXON_INTERLAYER_BUFFER_SIZE` macro (a static_assert checks it). So build.rs
  auto-detects the header, emits a glue TU (with those includes) exposing
  `axon_active_model()` -> `&model_<name>`, compiles it (`-std=c11`), and sets
  `cargo:rustc-cfg=has_model`. The `model_<name>` descriptor has EXTERNAL linkage
  (`const`, not `static`), so it links; the weight blob `axon_model_const_*` is
  `const static` (TU-local) but reached via the descriptor's pointer.
- Validated end-to-end at link time with Nordic's `hello_axon` (sin regression):
  final ELF contains `model_hello_axon` + `cmd_buffer_hello_axon` (.rodata) and
  `nrf_axon_nn_model_infer_sync`, zero undefined symbols, bss += interlayer buffer.

## Toolchain / env state

- Edge AI add-on v2.0 requires NCS **v3.3.0** (`west.yml` pins `nrf` @ v3.3.0).
- Board target: `nrf54lm20dk/nrf54lm20b/cpuapp`.
- Local env: standalone `west v1.5.0`, **no NCS workspace yet** (`west topdir` empty,
  no `~/ncs`). Full NCS install is multi-GB.
- Repo inspected at `/tmp/sdk-edge-ai` (github.com/nrfconnect/sdk-edge-ai).
- Samples: `samples/axon/hello_axon`, `samples/axon/axon_low_power`,
  apps `person_detection`, `ww_kws` (keyword spotting), `gesture_recognition`.
