// 直连 NSCQ（libnvidia-nscq）读 NVSwitch 的 per-port throughput_counters —— DCGM 没接的那条路。
// 目的：DCGM 走 NVSDM 拿到的是不跟随流量的静态 IB 计数器；NSCQ 有 /{nvswitch}/nvlink/{port}/throughput_counters
// 路径（struct nscq_link_throughput_t{rx,tx}，单位 Mibits）。测它在本机 H100 上是否给真流量。
//
// 读两遍（中间 sleep 秒数由 argv[1] 给），打印每个 (switch,port) 的 rx/tx 及两遍差分。
// 若差分随负载增大 -> NSCQ 这条路能拿到过交换机的流量（DCGM 没用而已）；若恒 0/不变 -> 本机连 NSCQ 也拿不到。
//
// build: g++ -I<DCGM>/sdk/nvidia/nscq nscq_throughput_probe.cpp -o probe -lnvidia-nscq
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <unistd.h>
#include "nscq.h"
#include "path.h"

struct Rec { char label[64]; uint32_t port; uint64_t rx, tx; int rc; };
struct Collector { std::vector<Rec> recs; int calls = 0; };

// per-port 回调：签名照 DCGM DcgmNscqManager 的 per-port 模式 (device, port, rc, value, user)
static void tput_cb(const nscq_uuid_t *device, uint32_t port, nscq_rc_t rc,
                    nscq_link_throughput_t val, Collector *c) {
    if (!c) return;
    c->calls++;
    Rec r{}; r.port = port; r.rc = rc; r.rx = val.rx; r.tx = val.tx;
    nscq_label_t lbl; std::memset(&lbl, 0, sizeof(lbl));
    if (device && nscq_uuid_to_label(device, &lbl, 0) == 0)
        std::snprintf(r.label, sizeof(r.label), "%s", lbl.data);
    else
        std::snprintf(r.label, sizeof(r.label), "?");
    c->recs.push_back(r);
}

static int read_once(nscq_session_t s, const char *path, Collector &c) {
    c.recs.clear(); c.calls = 0;
    nscq_rc_t ret = nscq_session_path_observe(s, path, NSCQ_FN(tput_cb), &c, 0);
    return (int)ret;
}

int main(int argc, char **argv) {
    int sleep_s = (argc > 1) ? atoi(argv[1]) : 5;
    const char *path = (argc > 2) ? argv[2] : nscq_nvswitch_nvlink_port_throughput_counters;
    printf("NSCQ path = %s ; sleep between reads = %ds\n", path, sleep_s);

    nscq_session_result_t sr = nscq_session_create(NSCQ_SESSION_CREATE_MOUNT_DEVICES);
    if (NSCQ_ERROR(sr.rc)) { printf("session_create ERROR rc=%d\n", (int)sr.rc); return 1; }
    if (NSCQ_WARNING(sr.rc)) printf("session_create WARNING rc=%d (driver/nscq mismatch?)\n", (int)sr.rc);
    nscq_session_t s = sr.session;

    Collector a, b;
    int r1 = read_once(s, path, a);
    printf("read#1 rc=%d callback_hits=%d records=%zu\n", r1, a.calls, a.recs.size());
    sleep(sleep_s);
    int r2 = read_once(s, path, b);
    printf("read#2 rc=%d callback_hits=%d records=%zu\n", r2, b.calls, b.recs.size());

    // 打印每 (switch,port) 两遍值 + 差分；只打非全 0 的 + 汇总
    printf("\n%-22s %5s | %14s %14s | %14s %14s | %12s %12s\n",
           "switch", "port", "rx1(Mib)", "tx1(Mib)", "rx2(Mib)", "tx2(Mib)", "drx", "dtx");
    uint64_t sum_drx = 0, sum_dtx = 0; int nonzero = 0;
    for (size_t i = 0; i < a.recs.size() && i < b.recs.size(); i++) {
        Rec &x = a.recs[i]; Rec &y = b.recs[i];
        long long drx = (long long)y.rx - (long long)x.rx;
        long long dtx = (long long)y.tx - (long long)x.tx;
        if (x.rx || x.tx || y.rx || y.tx || drx || dtx) {
            printf("%-22s %5u | %14llu %14llu | %14llu %14llu | %12lld %12lld\n",
                   x.label, x.port, (unsigned long long)x.rx, (unsigned long long)x.tx,
                   (unsigned long long)y.rx, (unsigned long long)y.tx, drx, dtx);
            nonzero++;
        }
        if (drx > 0) sum_drx += drx;
        if (dtx > 0) sum_dtx += dtx;
    }
    printf("\nSUMMARY: nonzero_rows=%d  sum_drx=%llu Mib (%.1f GB)  sum_dtx=%llu Mib (%.1f GB) over %ds\n",
           nonzero, (unsigned long long)sum_drx, sum_drx / 8.0 / 1024.0,
           (unsigned long long)sum_dtx, sum_dtx / 8.0 / 1024.0, sleep_s);
    nscq_session_destroy(s);
    return 0;
}
