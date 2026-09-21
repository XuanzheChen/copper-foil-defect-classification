# Inference Speed Benchmark

Generated: `2026-09-16T21:23:31+08:00`

This file was generated locally by `temp/infer_direct.py`. The script does not upload or commit this report to Git.

## Results

All measurements use `batch_size=1`, sequential requests, and exclude model/checkpoint loading time.

| Mode | Scope | FPS | Mean ms | P50 ms | P95 ms | P99 ms | Min ms | Max ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| CPU FP32 | Model forward only | 64.54 | 15.495 | 14.765 | 15.916 | 17.304 | 14.542 | 530.190 |
| CPU FP32 | End-to-end | 60.91 | 16.418 | 16.066 | 17.714 | 19.246 | 15.790 | 65.836 |
| GPU FP32 | Model forward only | 463.38 | 2.158 | 2.158 | 2.165 | 2.170 | 2.147 | 2.190 |
| GPU FP32 | End-to-end | 284.07 | 3.520 | 3.284 | 3.383 | 12.359 | 3.138 | 33.647 |
| GPU AMP (FP16 autocast) | Model forward only | 422.88 | 2.365 | 2.354 | 2.416 | 2.447 | 2.313 | 2.834 |
| GPU AMP (FP16 autocast) | End-to-end | 252.56 | 3.959 | 3.710 | 3.936 | 10.767 | 3.605 | 26.121 |

- **Model forward only:** a preprocessed `[1, 3, H, W]` tensor already resides on the inference device; only `model(tensor)` is timed.
- **End-to-end:** file open/decode → RGB → resize/crop/normalize → device transfer → model forward → argmax → scalar result on CPU.
- CPU mode has no CPU→GPU transfer. GPU modes include the CPU→GPU transfer in end-to-end timing.
- GPU AMP uses CUDA FP16 autocast; CPU mode is FP32.
- Repeated source files may benefit from the operating-system page cache when timed requests exceed the number of unique images.

## Benchmark Configuration

| Item | Value |
|---|---|
| Model | ConvNeXt V2 Tiny |
| Checkpoint | `/home/cxz/projects/copper-foil-defect-classification/pretrained-weights/convnextv2_tiny_1k_224_ema.pt` |
| Input size | 224 × 224 |
| Batch size | 1 |
| Request mode | Sequential |
| Benchmark modes | CPU FP32, GPU FP32, GPU AMP (FP16 autocast) |
| Unique source images | 5486 |
| Timed requests per mode/scope | 1000 |
| Warmup requests per mode/scope | 50 |
| Source directory | `/share/data/cxz/copper-foil-defect-classification` |

## Hardware and Software

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 5080 |
| GPU VRAM | 15.46 GiB |
| GPU SM count | 84 |
| GPU compute capability | 12.0 |
| NVIDIA driver | 595.84 |
| Driver-supported CUDA | 13.2 |
| CPU | Intel(R) Core(TM) Ultra 7 265KF |
| Logical CPUs | 20 |
| PyTorch CPU intra-op threads | 20 |
| PyTorch CPU inter-op threads | 20 |
| OS | Linux-7.0.0-31-generic-x86_64-with-glibc2.43 |
| Python | 3.10.21 |
| PyTorch | 2.7.1+cu128 |
| TorchVision | 0.22.1+cu128 |
| PyTorch CUDA runtime | 12.8 |
| cuDNN | 90701 |

## Raw System Information

### `nvidia-smi`

```text
Wed Sep 16 21:23:31 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 595.84                 Driver Version: 595.84         CUDA Version: 13.2     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 5080        Off |   00000000:02:00.0 Off |                  N/A |
| 33%   50C    P1            112W /  365W |     501MiB /  16303MiB |     38%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A            6413      G   /usr/bin/gnome-shell                     87MiB |
|    0   N/A  N/A            7427      G   /usr/bin/Xwayland                         6MiB |
|    0   N/A  N/A           10864      G   /usr/bin/clash-verge                      4MiB |
|    0   N/A  N/A           62691      C   python                                  358MiB |
+-----------------------------------------------------------------------------------------+
```

### `lscpu`

```text
Architecture:                            x86_64
CPU op-mode(s):                          32-bit, 64-bit
Address sizes:                           42 bits physical, 48 bits virtual
Byte Order:                              Little Endian
CPU(s):                                  20
On-line CPU(s) list:                     0-19
Vendor ID:                               GenuineIntel
Model name:                              Intel(R) Core(TM) Ultra 7 265KF
CPU family:                              6
Model:                                   198
Thread(s) per core:                      1
Core(s) per socket:                      20
Socket(s):                               1
Stepping:                                2
CPU(s) scaling MHz:                      33%
CPU max MHz:                             5500.0000
CPU min MHz:                             800.0000
BogoMIPS:                                7756.80
Flags:                                  fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx pdpe1gb rdtscp lm constant_tsc art arch_perfmon pebs bts rep_good nopl xtopology nonstop_tsc cpuid aperfmperf tsc_known_freq pni pclmulqdq dtes64 monitor ds_cpl vmx smx est tm2 ssse3 sdbg fma cx16 xtpr pdcm pcid sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand lahf_lm abm 3dnowprefetch cpuid_fault ssbd ibrs ibpb stibp ibrs_enhanced tpr_shadow flexpriority ept vpid ept_ad fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid rdt_a rdseed adx smap clflushopt clwb intel_pt sha_ni xsaveopt xsavec xgetbv1 xsaves split_lock_detect user_shstk avx_vnni lam wbnoinvd dtherm ida arat pln pts hwp hwp_notify hwp_act_window hwp_epp hwp_pkg_req hfi vnmi umip pku ospke waitpkg gfni vaes vpclmulqdq rdpid bus_lock_detect movdiri movdir64b fsrm md_clear serialize arch_lbr ibt flush_l1d arch_capabilities
Virtualization:                          VT-x
L1d cache:                               704 KiB (18 instances)
L1i cache:                               1.1 MiB (18 instances)
L2 cache:                                36 MiB (11 instances)
L3 cache:                                30 MiB (1 instance)
NUMA node(s):                            1
NUMA node0 CPU(s):                       0-19
Vulnerability Gather data sampling:      Not affected
Vulnerability Ghostwrite:                Not affected
Vulnerability Indirect target selection: Not affected
Vulnerability Itlb multihit:             Not affected
Vulnerability L1tf:                      Not affected
Vulnerability Mds:                       Not affected
Vulnerability Meltdown:                  Not affected
Vulnerability Mmio stale data:           Not affected
Vulnerability Old microcode:             Not affected
Vulnerability Reg file data sampling:    Not affected
Vulnerability Retbleed:                  Not affected
Vulnerability Spec rstack overflow:      Not affected
Vulnerability Spec store bypass:         Mitigation; Speculative Store Bypass disabled via prctl
Vulnerability Spectre v1:                Mitigation; usercopy/swapgs barriers and __user pointer sanitization
Vulnerability Spectre v2:                Mitigation; Enhanced / Automatic IBRS; IBPB conditional; PBRSB-eIBRS Not affected; BHI BHI_DIS_S
Vulnerability Srbds:                     Not affected
Vulnerability Tsa:                       Not affected
Vulnerability Tsx async abort:           Not affected
Vulnerability Vmscape:                   Mitigation; IBPB before exit to userspace
```

### `free -h`

```text
total        used        free      shared  buff/cache   available
Mem:            61Gi       5.1Gi        32Gi       108Mi        24Gi        56Gi
Swap:          8.0Gi          0B       8.0Gi
```

### `lsblk -o NAME,MODEL,SIZE,ROTA,TYPE,MOUNTPOINT`

```text
NAME        MODEL                 SIZE ROTA TYPE MOUNTPOINT
loop0                              20M    0 loop /snap/desktop-security-center/151
loop1                            66.8M    0 loop /snap/core24/2124
loop2                            19.6M    0 loop /snap/desktop-security-center/150
loop3                           261.1M    0 loop /snap/firefox/8803
loop4                            66.8M    0 loop /snap/core24/1643
loop5                           260.8M    0 loop /snap/firefox/8863
loop6                            16.5M    0 loop /snap/firmware-updater/226
loop7                               4K    0 loop /snap/bare/5
loop8                            91.7M    0 loop /snap/gtk-common-themes/1535
loop9                           614.5M    0 loop /snap/gnome-46-2404/164
loop10                          606.1M    0 loop /snap/gnome-46-2404/153
loop11                            395M    0 loop /snap/mesa-2404/1165
loop12                            402M    0 loop /snap/mesa-2404/1839
loop13                           18.8M    0 loop /snap/prompting-client/204
loop14                           18.8M    0 loop /snap/prompting-client/222
loop15                           15.7M    0 loop /snap/snap-store/1367
loop16                           11.8M    0 loop /snap/snap-store/1390
loop17                           50.1M    0 loop /snap/snapd/27710
loop18                           50.3M    0 loop /snap/snapd/27738
loop19                            828K    0 loop /snap/snapd-desktop-integration/391
loop20                            828K    0 loop /snap/snapd-desktop-integration/387
nvme0n1     KINGSTON SNV3S2000G   1.8T    0 disk 
|-nvme0n1p1                         1G    0 part /boot/efi
`-nvme0n1p2                       1.8T    0 part /
```

### `df -hT /share/data/cxz/copper-foil-defect-classification`

```text
Filesystem     Type  Size  Used Avail Use% Mounted on
/dev/nvme0n1p2 ext4  1.8T  657G  1.1T  38% /
```

### `nvidia-smi topo -m`

```text
GPU0	CPU Affinity	NUMA Affinity	GPU NUMA ID
GPU0	 X 	0-19	0		N/A

Legend:

  X    = Self
  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
  NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node
  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
  PXB  = Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
  PIX  = Connection traversing at most a single PCIe bridge
  NV#  = Connection traversing a bonded set of # NVLinks
```
