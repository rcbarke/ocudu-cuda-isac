# Deployment and measurement plan

All three modes replay the same held-out uplink episodes through the same receiver and sensing pipeline. Mode 1 is the first Spark milestone and is completed for every ablation before Modes 2 and 3 begin. Aerial Framework and Aerial-OCUDU over the air follow. [Build order](../dev/DEVELOPMENT.md#build-order) · [Aerial ports](../integration/AERIAL.md) Hardware timings, successful UE decoding, link rates and resource isolation require measurements from the named machines. The portable replay fixture can be exercised before any of that equipment is available.

| Mode | Placement of communications and sensing | Observation handoff | Question |
|---|---|---|---|
| 1. One DGX Spark | CPU hosts CU/DU control, virtual radio and optional 5GC. GB10 GPU hosts CUDA PHY and sensing in separate MPS client processes. GB10 on DGX Spark does not support MIG. | Bounded local ring, one producer and an independently paced consumer | How much do concurrent DSP and inference affect each other's tails? |
| 2. Two DGX Sparks | Spark 2 hosts CU/DU, virtual radio/channel replay and CUDA DSP. Spark 1 hosts the 5GC on CPU and sensing on its GB10. A single direct CX7 DAC links them. | Framed observations carried between the hosts, with independent sequence, timestamp and drop counters | What does removing same-GPU contention cost in transport and observation age? |
| 3. MSI XpertStation WS300, GB300 | CPU hosts CU/DU control, virtual radio and 5GC. One GB300 GPU instance (GI) serves DSP. A second GI serves sensing. Each GI has its own compute instance (CI). | Local host-memory ring with explicit GPU-to-host and host-to-GPU completion | What does hardware isolation buy under the same workload and fixed GI capacities? |

The demo can start with a known, scheduled PUSCH and a decoder CRC. Add core traffic when exercising registration, user-plane throughput and the full CU/DU load. Report which ran. A virtual RU is part of the simulated radio path. The network joining two Sparks carries observations and, when the core is active, its N2/N3 traffic. CU–DU F1 and simulated fronthaul traffic stay on Spark 2 in this placement.

## Mode 1 · Sharing the GB10 with MPS

Start the CUDA PHY and sensing worker as distinct clients of the same MPS service. Verify the server/client attachment using the control interface provided with the installed MPS version. Preserve separate process IDs, GPU identity, CUDA and driver versions, per-client settings and queue limits. Compare DSP alone, sensing replay alone, then both under the same offered load. Launching another CUDA process by itself does not demonstrate that MPS is active. [NVIDIA MPS deployment guide](https://docs.nvidia.com/deploy/mps/latest/when-to-use-mps.html)

Use ordinary MPS scheduling for the initial comparison. An active-thread-percentage ceiling constrains available execution resources but does not reserve a particular set of SMs. Static SM partitioning is a separate option on supported configurations that can assign disjoint SM sets. MPS v3 also supports memory-capacity limits under CUDA 13.4+, Linux cgroup v2 and its documented UVM restrictions. Record these as distinct experiments if used. Neither a concurrency ceiling nor an allocation-capacity limit establishes dedicated memory bandwidth or a latency bound. [MPS execution provisioning](https://docs.nvidia.com/deploy/mps/latest/when-to-use-mps.html) · [MPS v3 memory partitioning](https://docs.nvidia.com/deploy/mps/latest/mpsv3-memory-partitioning.html)

Begin with an owned host-visible buffer and explicit producer-complete and consumer-release events. Later test a supported shared mapping on the actual GB10 stack if it improves the measured tail. Unified physical memory alone says nothing about address validity across processes, visibility to the NIC or release ordering. The critical PHY thread publishes without waiting for inference. When the ring is full, record a dropped sensing observation and continue the communications path. [Spark CUDA porting guide](https://docs.nvidia.com/dgx/dgx-spark-porting-guide/latest/porting/cuda.html) The buffer, its ownership rules and its control channel are the `obs-ring/v1` contract. [Observation contract](../interfaces/OBSERVATION.md)

## Mode 2 · One 100 Gb/s link between two Sparks

Use one direct CX7 connection and bind the transport and measurements to its actual interface. Record NICs, ports, firmware, cable, MTU, `ethtool` link speed, routing, and the bandwidth the application sustains in each direction. NVIDIA documents two 200 Gb/s-capable ports and recommends compatible cabling. Qualify the intended 100 Gb/s DAC and negotiation on the two machines. If using a 200 Gb/s link rate-limited to 100 Gb/s, report both rates and the shaping method. A configured cap and a negotiated link describe different experiments. [DGX Spark ConnectX-7 networking](https://docs.nvidia.com/dgx/dgx-spark/spark-clustering.html)

Begin with an application-framed socket transfer. Bound sender and receiver queue depths, transmit sequence ID, slot/UE/allocation identity, stage, extent, units and source time with each payload. Count gaps, stale frames, duplicates and invalid lengths before inference. Keep slow consumers off the DU thread. For a verbs comparison, NVIDIA recommends `cudaHostAlloc` buffers registered with `ibv_reg_mr` on Spark. Its porting guide says the NIC cannot coherently access memory from `cudaMalloc`. Measure GPU-to-staging, NIC, and destination staging-to-GPU costs and publish the actual memory path. [Spark CUDA and RDMA buffer guidance](https://docs.nvidia.com/dgx/dgx-spark-porting-guide/latest/porting/cuda.html)

For an illustrative upper bound, 4 RX ports × 14 symbols × 273 PRBs × 12 subcarriers × 8 bytes per complex64 value is 1,467,648 bytes per slot. At 2,000 such slots/s the payload alone needs 23.48 Gb/s. At 100 Gb/s it takes at least 117.4 μs to serialize a single grid. At 16 RX ports the offered payload becomes 93.93 Gb/s. The pinned receiver itself stores the grid as complex bf16, four bytes per value (S2.1), so the same grid in its stored form is 733,824 bytes per slot. TDD uplink duty cycle, grants, representation precision, headers, retries and time spent staging change these figures. Calculate the actual bytes per observation and report measured bandwidth plus age. Transport the same receiver representation in all modes for the topology comparison. Test DMRS-only observations or compression separately.

N2 connects CU-CP to AMF using NGAP/SCTP and N3 connects CU-UP to UPF using GTP-U. Give the sensing transport its own socket or queue and record the impact of core traffic on observation age. Use synchronized host clocks and measure residual offset and uncertainty before and after the run. If synchronization is insufficient, report same-clock stage durations and round-trip measurements. Cross-host CUDA events cannot establish one-way elapsed time.

## Mode 3 · Separate GB300 GPU instances

The WS300 is a DGX Station system using the GB300 Grace Blackwell Ultra Desktop Superchip. The MIG guide lists CUDA 13 with R595 ≥ 595.58.03 for GB300 DGX Station. Apply MIG to the GB300 device, identifying its UUID explicitly if another GPU is installed. NVIDIA lists profiles for 252 GB of GPU memory. The system's larger coherent-memory figure includes CPU memory. [MSI WS300](https://www.msi.com/Landing/NVIDIA-DGX-STATION) · [MIG deployment requirements](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/latest/deployment-considerations.html) · [GB300 profiles and placement diagram](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/latest/supported-mig-profiles.html)

A starting placement is `4g.126gb` for DSP and `3g.126gb` for sensing. NVIDIA's profile diagram permits the complementary placement. It assigns four and three sevenths of the SM capacity respectively, with half the available memory slices for each. With workloads stopped and the administrator's agreed MIG configuration in place, inspect `nvidia-smi mig -lgip`, `nvidia-smi mig -lgipp` and `nvidia-smi -L`. Save GI/CI configuration and verify one distinct MIG device UUID per worker. Profile names describe capacity. Record application-visible free memory, clocks, context and observed interference. NVIDIA's general MIG page lists preliminary instance sizes for the GB300 NVL72 (7 × 34 GB, 4 × 69 GB, 2 × 139 GB, 1 × 279 GB per GPU). That is a different system from the station, so take the profiles from the station's own listing. The CPU, host services, platform power envelope and the publication ring remain shared. [MIG profiles](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/latest/supported-mig-profiles.html) · [GPU and compute instance isolation](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/latest/concepts.html)

Use the host-staged path for the first cross-GI experiment. NVIDIA documents same-physical-GPU peer-to-peer access between MIG instances on R570 and later. CUDA IPC across separate GPU instances is unsupported, so passing a `cudaIpcMemHandle` from the PHY process to the sensing process is invalid here. A separately measured broker that can enumerate both devices and establish peer access could test a direct peer copy after checking capability and data integrity. Report each copy, synchronization and process boundary. [MIG application considerations](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/latest/deployment-considerations.html)

## Functional splits on any placement

A placement says where the work runs. A split says where the DU's radio boundary is. This project deploys three of the pin's targets, and S5.4 gives each a launch profile and one ingress adapter that reads the same episode store.

| Split | Targets at the pin | Boundary that replay enters | Use |
|---|---|---|---|
| Staged harness | The `pusch_e2e_pipeline_test` fixture path | Upper-PHY resource grid | Demonstration (S2–S4) |
| Monolithic CU/DU | `gnb` | As configured: dummy RU, SDR or Open Fronthaul | Full stack in one process with scheduler pacing |
| Split 8 | `gnb_split_8`, `odu_split_8` | Baseband samples at the ZeroMQ radio | The DU alone on Sionna RT channels in both directions, because the lower PHY is in the DU. Also where an OAI software UE with an OAI 5GC attaches (S6.10). |
| Split 7.2x | `gnb_split_7_2`, `odu_split_7_2` | U-plane IQ from the `ru_emulator` example (socket or DPDK transceiver) or an O-RU | Packetized fronthaul. The lower PHY is in the RU, so a channel can be injected on the uplink only. |

The `gnb_split_*` and `odu_split_*` targets are excluded from the default build. The accepted S1.1 build has ZeroMQ and UHD off, so S5.4 adds a build profile with ZeroMQ on for split-8 replay and software-UE testing. DPDK is off by default, and the RU emulator then uses its socket transceiver. ZeroMQ runs in lockstep and not in real time, so it supports correctness and sensing-quality comparisons and no latency claim. The dummy RU supplies timing and load and no signal. The pin also carries a split-6 unit that ends in a plugin seam. It is noted in the [inventory](../integration/OCUDU.md) and is not deployed here. Over the air, the FAPI boundary comes from the Aerial-OCUDU branch instead ([Aerial ports](../integration/AERIAL.md)).

## Memory paths and sharing controls

S5.5 varies one of these at a time on the collocated Spark and reads the effective setting back.

| Factor | Options |
|---|---|
| Uplink grid memory | `ul_cuda_visible_grid_mode`: `auto`, `managed`, `pinned`. On the GB10 `auto` resolves to managed memory. |
| Observation export | Host copy into the pinned ring, device-side copy before publication, a shared mapping where supported |
| Payload | The receiver's estimate (O3) in FP16 (19,584 bytes per slot at the demonstration profile) or complex64, comb LS in complex64 (19,584 bytes), or the full grid (274,176 bytes) |
| MPS | Default, active-thread percentage, static SM partition, client priority, memory limit |
| Stream priority | The pin's PUSCH and PDSCH CUDA stream priority overrides |

The GB10 shares memory between CPU and GPU. A GPU memory figure is therefore also a system memory figure, and both are recorded.

## Comparable runs and evidence

1. Fix the episode hashes, PUSCH allocation/schedule, TDD pattern, model and GPU precision, sensing runtime and engine backend, chosen observation stage and payload bytes. Record build flags, versions, CPU affinity, clocks, warm-up, powered hardware and number of concurrent cells/UEs. Keep a DSP-only control in each placement, including a DSP-only run in the selected MIG profile.
2. Time receiver publication, transport arrival, inference dispatch and decision. Retain counts for generated, offered, published, dropped, expired and scored observations. Report p50/p95/p99 with sample counts, error and false-alarm rates, timely correct decisions, transport backlog and the deadline used. A drop must remain visible in the offered-update denominator.
3. Report load-matched CRC/BLER, throughput, slot deadline misses and workload status alongside sensing performance. Use independent episodes/runs for uncertainty. GPU event time covers only its recorded device work. For power comparisons use a consistent measurement boundary that includes both Sparks in Mode 2.

Start with [the advisor demo](../demo/ADVISOR_DEMO.md). The linked upstream documents describe hardware capabilities. Record successful bring-up and measured outcomes per run.
