# F1 end to end on a Hetzner Ubuntu server

Everything from a fresh server to the data archive you send back. Commands
marked `mac$` run in a terminal on your MacBook; commands marked `srv$` run over
SSH on the Hetzner machine. Nothing here needs to be typed twice: copy a block,
paste it, read what it prints.

Pre-registered in `docs/preregistration/PREREGISTRATION_EXTERNAL.md`. The
campaign is F1; a campaign that fails a validity gate is reported as failed and
is not analysed, so the smoke test in step 8 is not optional.

Rough time: 1–2 h of setup, ~7 h of unattended campaign, 10 min of collection.

---

## 0. The server

**Use Ubuntu 24.04 LTS, and use a server of its own.**

Two reasons, both learned the hard way on the first attempt:

- *Kernel.* The UPF needs gtp5g **v0.9.5** — that is what the current UPF
  supports — and v0.9.5 does not build against a 7.x kernel: the kernel build
  system changed how `$(src)` resolves for out-of-tree modules, so the module's
  own headers stop being found and the build dies with
  `fatal error: genl.h: No such file or directory`. Ubuntu 24.04 LTS ships
  6.8, which is inside the range that works. A newer gtp5g tag (v0.10.x) builds
  on 7.x but is not the version the UPF is pinned to, so it is not a shortcut.
- *Blast radius.* The first attempt was on `cybernord-prod-1`. This campaign
  loads a kernel module, runs privileged containers, injects packet delay and
  saturates CPU for seven hours. That does not belong on a production host.

In the Hetzner console: create a **new** Cloud server, image *Ubuntu 24.04*, at
least 4 vCPU / 8 GB RAM / 80 GB disk, then add it to `~/.ssh/config` on the Mac
so the commands below work as written:

```
mac$ cat >> ~/.ssh/config <<'CFG'
Host cybernord-f1
    HostName <NEW-SERVER-IP>
    User root
CFG
```

Replace `cybernord` with `cybernord-f1` in every command below if you keep both
hosts. Then check the basics:

`git clone https://github.com/free5gc/free5gc.git` is the
**source tree**, which we do not need: the campaign uses the published container
images. You can delete it (`rm -rf ~/free5gc`) or leave it; it is not used
below. And `cd gtp5g` failed because gtp5g is a separate repository — step 2.

```bash
srv$ apt update && apt -y upgrade && reboot      # reconnect after the reboot
srv$ uname -r                          # must start with 6. — 7.x will not work
srv$ grep -c avx /proc/cpuinfo         # must be > 0, MongoDB 5+ needs AVX
```

If `uname -r` prints a 7.x kernel, you are on the wrong image; rebuild the
server with Ubuntu 24.04 before going further. Everything after this point
assumes a 6.x kernel.

## 0b. Running it on your own PC instead of a rented server

A native Ubuntu install on your own x86-64 machine is **as valid as the rented
server** — same kernel, same module, same containers, nothing about the campaign
changes. It is the free option, and the only one that is free without
compromise (WSL2 is not: its kernel ships without distro headers, so gtp5g means
rebuilding the WSL kernel yourself, and SCTP for the gNB link may need the same
rebuild).

What the machine needs:

- x86-64 CPU with AVX (`grep -c avx /proc/cpuinfo` > 0 — MongoDB needs it),
  4 cores and 8 GB RAM are comfortable, 4 GB works if nothing else runs;
- ~40 GB free disk for Ubuntu, the images and the Prometheus data;
- to stay powered on and awake for about eight hours in one stretch.

Three ways to get Ubuntu 24.04 onto it, in order of how little they touch
Windows:

1. **External USB SSD install.** Install Ubuntu onto an external SSD and boot
   from it. Windows is untouched; unplug the disk and the machine is as it was.
   Slowest disk, which does not matter here — the campaign is CPU and clock
   bound, not disk bound.
2. **Dual boot.** Shrink the Windows partition, install alongside. Back up
   anything you cannot lose first; resizing a system partition is the one step
   in this whole document that can cost you data.
3. **Wipe and install**, if the machine is spare.

A live USB session without persistence also technically works, but a reboot or
a crash loses the campaign, and it has to survive seven hours — not worth it.

After installing:

```bash
pc$ sudo apt update && sudo apt -y upgrade && sudo reboot
pc$ uname -r                       # must start with 6.
pc$ sudo systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target
pc$ grep -c avx /proc/cpuinfo
```

The suspend line matters: a laptop that sleeps four hours into the campaign
produces a gap in the telemetry, and the collector will reject those windows.

Then follow steps 1 to 10 exactly as written, with `sudo` in front of the
commands that need it, and with two substitutions:

- **Step 4, getting the harness onto the machine.** There is no `scp` from a
  server; either copy `spcrl_testbed_v5.tar.gz` across on a USB stick, or, if
  the Mac and the PC are on the same network, from the Mac:
  `scp spcrl_testbed_v5.tar.gz <user>@<pc-ip>:~/` (the PC needs
  `sudo apt -y install openssh-server` first). Downloading it from this chat in
  a browser on the PC works too.
- **Step 11, getting the data back.** Same three ways in reverse: USB stick,
  `scp` from the Mac (`scp <user>@<pc-ip>:~/spcrl_f1_data.tar.gz .`), or upload
  the archive to the chat directly from the PC.

Everything else — gtp5g, compose, subscribers, workload, smoke test, campaign,
collection — is identical. The web console needs no SSH tunnel here: it is
http://localhost:5000 on the machine itself.

## 1. Docker

```bash
srv$ apt -y install ca-certificates curl gnupg
srv$ install -m 0755 -d /etc/apt/keyrings
srv$ curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
       | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
srv$ chmod a+r /etc/apt/keyrings/docker.gpg
srv$ echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo $VERSION_CODENAME) stable" \
       > /etc/apt/sources.list.d/docker.list
srv$ apt update && apt -y install docker-ce docker-ce-cli containerd.io \
       docker-buildx-plugin docker-compose-plugin
srv$ docker compose version            # v2.x
```

## 2. The gtp5g kernel module

The UPF needs it, and only version 0.9.5 is supported by the current UPF.

```bash
srv$ apt -y install gcc g++ make cmake autoconf libtool pkg-config \
       libmnl-dev libyaml-dev linux-headers-$(uname -r) dkms git
srv$ cd ~ && git clone --branch v0.9.5 --depth 1 https://github.com/free5gc/gtp5g.git
srv$ cd gtp5g && make clean && make && make install
srv$ modprobe gtp5g; lsmod | grep gtp5g      # must print a gtp5g line
```

`make install` must end without errors and `lsmod` must print the module. Two
failures are worth naming, because both have already happened:

- `fatal error: genl.h: No such file or directory` → you are on a 7.x kernel.
  Go back to step 0; no flag fixes this.
- `make` stops on a missing `Module.symvers` or complains that the running
  kernel differs from the headers → you upgraded the kernel and have not
  rebooted. `reboot`, then repeat this step.

Anything else: send me the last 30 lines before changing versions.

## 3. free5GC compose

```bash
srv$ cd ~ && git clone https://github.com/free5gc/free5gc-compose.git
srv$ cd free5gc-compose && docker compose pull
```

## 4. Copy the SP-CRL harness from your Mac

```bash
mac$ cd "/Users/nikitatarasov/Science Article 2025-/Structural Priors Enable Generalizable Causal Diagnosis and Counterfactual Recovery in Dynamic Systems"
mac$ scp spcrl_testbed_v5.tar.gz cybernord:~/
srv$ cd ~ && tar xzf spcrl_testbed_v5.tar.gz      # gives ~/telecom and ~/microservices
srv$ ls ~/telecom/scripts                          # campaign_compose.py, smoke_test_f1.sh, ...
srv$ apt -y install python3-yaml python3-numpy python3-requests
```

(If you prefer, `scp spcrl_repo.tar.gz` too and unpack it — but only the
`benchmarks/telecom` part is needed on the server; the analysis runs on my side.)

## 5. Start the core with the campaign's monitoring

```bash
srv$ cd ~/free5gc-compose
srv$ cp ~/telecom/compose/*.y*ml .
srv$ docker compose -f docker-compose.yaml -f docker-compose.spcrl.yaml config >/dev/null \
       && echo "overlay OK"
srv$ docker compose -f docker-compose.yaml -f docker-compose.spcrl.yaml up -d
srv$ docker ps --format '{{.Names}}'
```

If `config` complains about an unknown service, the compose file names its
functions differently from the overlay: edit the service keys in
`docker-compose.spcrl.yaml` to match, and the `container:` fields in
`~/telecom/f1_topology.yaml` to match `docker ps`. Do not touch `depends_on` —
that is the structural prior itself.

Check the radio link:

```bash
srv$ docker logs ueransim 2>&1 | grep -i "ng setup" | tail -3   # expect: successful
```

## 6. Subscribers (the one manual step)

40 subscribers with consecutive IMSIs, same key, OPc and slice as the UE config
the compose file ships. The web console runs on the server's port 5000; do not
expose it to the internet — tunnel it:

```bash
mac$ ssh -L 5000:localhost:5000 cybernord
```

Then open http://localhost:5000 on the Mac (default login is in the free5GC
"Create Subscriber via webconsole" guide). Take the SUPI from
`~/free5gc-compose/config/uecfg.yaml` (field `supi`, commonly
`imsi-208930000000001`) and create 40 consecutive IMSIs from it.

```bash
srv$ docker exec mongodb mongosh free5gc --quiet \
       --eval 'db.subscriptionData.authenticationData.authenticationSubscription.countDocuments()'
```

That count must be 40 before you go on.

## 7. Start the workload

```bash
srv$ docker cp ~/telecom/scripts/workload_ueransim.sh ueransim:/workload.sh
srv$ docker exec -d ueransim sh -c \
       'bash /workload.sh /ueransim/config/uecfg.yaml 208930000000001 40 2 20 > /tmp/workload.log 2>&1'
srv$ sleep 90 && docker exec ueransim tail -5 /tmp/workload.log
```

(Adjust the config path and the first IMSI to what step 6 actually used.)

## 8. Smoke test — must pass

```bash
srv$ cd ~/telecom && chmod +x scripts/*.sh
srv$ ./scripts/smoke_test_f1.sh http://localhost:9090
```

It stops at the first failed assumption and names it. **If it fails, stop and
send me the output.** Do not start the campaign on a failed smoke test.

## 9. The campaign (~7 h)

```bash
srv$ apt -y install tmux
srv$ tmux new -s f1
srv$ cd ~/telecom && python3 scripts/campaign_compose.py --dry-run     # 38 runs
srv$ python3 scripts/campaign_compose.py 2>&1 | tee ~/f1_campaign.out
```

Detach with `Ctrl-b d`; the campaign survives your SSH session. Reattach with
`tmux attach -t f1`. Check progress any time:

```bash
srv$ tail -5 ~/f1_campaign.out
```

## 10. Collect, immediately after it finishes

Prometheus keeps a limited window, so do this straight away.

```bash
srv$ cd ~ && python3 microservices/scripts/collect.py \
       --topology telecom/f1_topology.yaml --runtime compose \
       --campaign telecom/data/f1/campaign.jsonl \
       --out data/f1 --prometheus http://localhost:9090 --step 1
srv$ tar czf spcrl_f1_data.tar.gz data/f1
srv$ ls -la spcrl_f1_data.tar.gz
```

The collector prints a line like `variables=45 ... all windows contain
telemetry`. Keep it.

## 11. Bring the data back to the Mac

```bash
mac$ cd "/Users/nikitatarasov/Science Article 2025-/Structural Priors Enable Generalizable Causal Diagnosis and Counterfactual Recovery in Dynamic Systems"
mac$ scp cybernord:~/spcrl_f1_data.tar.gz .
mac$ scp cybernord:~/f1_campaign.out .
```

Send me `spcrl_f1_data.tar.gz`, `f1_campaign.out` and the smoke-test output.
I run the gates first — a failed gate means the campaign is reported as failed
and no model is run on it — and then only the pre-registered analyses
(`scripts/run_f1.sh`).

## 12. When it is over

The server is only needed for the campaign. Snapshot or destroy it once the
archive is on your Mac and the gates have passed; if a gate fails we may want it
again, so wait for my answer before destroying.

## Troubleshooting, shortest useful form

- `lsmod | grep gtp5g` empty → the UPF will not start; step 2 did not finish.
- UPF restarting in `docker ps` → check `docker logs upf`; usually gtp5g absent
  or a stale `upfgtp` interface from a previous run.
- No `registration is successful` in the ueransim log → subscribers (step 6) or
  the NGAP link (step 5), in that order.
- `tc: not found` in a function container → the smoke test says which one; tell
  me and I will register the netem-sidecar fallback as a dated amendment
  *before* any data is taken.
- Campaign died halfway → send `~/f1_campaign.out`; the runner is resumable and
  we restart it rather than starting a fresh log.
