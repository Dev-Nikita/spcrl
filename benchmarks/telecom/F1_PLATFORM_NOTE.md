# Where the free5GC campaign can actually run

Read this before step 1 of `F1_RUNBOOK.md`. It is about one hard constraint,
and it decides the machine you run F1 on.

## The constraint

free5GC's user plane function (UPF) needs the **gtp5g kernel module** loaded on
the host; the project's own compose repository states this requirement, and only
the UPF container depends on it. A kernel module is Linux-kernel code: it cannot
be loaded into the LinuxKit VM that Docker Desktop runs on macOS, and there is
no Apple Silicon path around that. The control-plane network functions (AMF,
AUSF, UDM, UDR, NSSF, PCF, SMF, NRF) and MongoDB are ordinary containers and do
not need it.

So on the MacBook, with Docker Desktop alone: the control plane comes up, UE
registration works, and **PDU-session establishment fails**, because the SMF has
no UPF to associate with.

The images are also built for amd64; on Apple Silicon they run under emulation,
which is slow but has no effect on what the campaign measures (telemetry is
sampled from the containers, and the fault magnitudes are relative).

## The three ways forward, and what each costs

**A. A Linux machine or VM with gtp5g (recommended).** Any x86-64 Linux host you
can get root on — a spare machine, a lab server, a cloud VM — running Ubuntu,
`apt install` the kernel headers, build and load gtp5g, then follow
`F1_RUNBOOK.md` unchanged. This is the only option that runs F1 as
pre-registered, with PDU sessions. On Apple Silicon a local arm64 Ubuntu VM
(UTM, Lima, Multipass) can also work, but gtp5g and the free5GC images then have
to be built for arm64, which is a longer detour than using an x86-64 host.

**B. Control-plane-only campaign on the MacBook.** Everything runs except the
UPF, and the workload is registration and deregistration cycles rather than PDU
sessions. This is a *different experiment* from the one registered: the
dependencies it exercises are the SBI calls between control-plane functions
only, and the SMF and PCF see far less traffic. It is still a valid second
system — a different technology, a different prior, a different driver — but it
must be registered as such, in a dated amendment written **before** the data
exist, stating that the UPF is absent, that PDU sessions are not attempted, and
that the fault set is restricted to the functions that remain. Ask for that
amendment and it will be written; it will not be written afterwards to fit the
data.

**C. Skip F1.** Report the paper on the two Kubernetes campaigns and say plainly
that no second system was tested. This is honest and is the weakest of the
three, because the manuscript's own Discussion names the second system as what
would separate a property of descriptions from a property of this testbed.

## How to check which situation you are in, in two minutes

```bash
docker info --format '{{.OSType}}/{{.Architecture}}'   # linux/aarch64 on an M-series Mac
docker run --rm --privileged alpine sh -c 'lsmod | grep -c gtp5g'   # 0 means no module
```

A zero from the second command means option A or B, never the runbook as
written.
