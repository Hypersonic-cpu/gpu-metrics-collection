/* nvswitch_traffic —— 实时打印本机 NVSwitch 过交换机的 NVLink 流量。
 *
 * 数据源 = 直连 libnvidia-nscq 读 per-port `throughput_counters`（累计计数器，单位 Mibits），
 * 自己差分成速率。这是本机唯一能拿到 NVSwitch 芯片流量的路——DCGM 的 nvswitch_throughput 字段
 * 在本机 H100 恒 0（NVSDM 后端 / 已知 DCGM issue #236），故本工具绕过 DCGM 直读 NSCQ。
 * 依据与实测见 experiments/nvswitch/。
 *
 * 架构：**后台 reader 线程**不停读 NSCQ（每次读全部端口 ~100ms），刷新一份"最新累计快照"；
 * 主线程按 -d 间隔取最新快照、与上次打印的快照差分成速率后打印。读与打印解耦——
 * 保证采集始终以 ~100ms 的读节拍进行（读延迟就是有效地板），打印再慢也不拖慢采集。
 *
 * 用法（参数风格参考 dcgmi dmon）：
 *   nvswitch_traffic [-e fields] [-i switches] [-l level] [-d ms] [-c count] [-r] [--csv] [-H]
 *     -e  字段(逗号): rx,tx,total     默认全部
 *     -i  switch 索引(逗号) 或 all     默认 all
 *     -l  粒度: total | switch | port  默认 switch
 *     -d  采样间隔 ms                  默认 1000
 *     -c  采几拍后退出 (0=Ctrl-C)      默认 0
 *     -r  打印原始累计增量(Mibits,整数) 而非速率(GB/s)
 *     --csv  CSV 输出(带 epoch)        -H 不打表头
 * 速率口径: GB/s(十进制)=ΔMibits*2^20/8/1e9/Δt。build: make（-lnvidia-nscq -lpthread，免 sudo）
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <signal.h>
#include <getopt.h>
#include <pthread.h>
#include "nscq_min.h"

#define MAXREC  2048
#define MAXSW   64
#define NPORTID 64      /* NVSwitch gen3 每颗 64 端口 */

enum { FLD_RX = 1, FLD_TX = 2, FLD_TOT = 4 };

typedef struct { char sw[64]; uint32_t port; uint64_t rx, tx; nscq_rc_t rc; } rec_t;

/* reader 线程的读缓冲（只有 reader 线程碰） */
static rec_t   g_cur[MAXREC];
static int     g_ncur;

/* reader -> 主线程 共享的"最新累计快照"（mutex 保护） */
static pthread_mutex_t g_mx = PTHREAD_MUTEX_INITIALIZER;
static rec_t   g_latest[MAXREC];
static int     g_latest_n;
static double  g_latest_t;
static uint64_t g_gen;          /* 每成功读一次 +1 */

/* switch 标签 -> 稳定索引（按标签排序） */
static char   g_labels[MAXSW][64];
static int    g_nsw;
/* per-(switch,port) -> 快照内记录下标；不存在 = -1 */
static int    g_ri_of[MAXSW][NPORTID];
static int    g_union_ports[NPORTID];   /* 出现过的端口 id（升序） */
static int    g_nunion;

/* 运行参数 */
static int    g_fields = FLD_RX | FLD_TX | FLD_TOT;
static int    g_level  = 1;             /* 0=total 1=switch 2=port */
static long   g_delay_ms = 1000;
static long   g_count  = 0;
static int    g_raw    = 0;
static int    g_csv    = 0;
static int    g_header = 1;
static int    g_isw_mask[MAXSW];
static int    g_isw_all = 1;

static volatile sig_atomic_t g_stop = 0;
static void on_sigint(int s) { (void)s; g_stop = 1; }

static double mib_to_GB(double mib) { return mib * 131072.0 / 1e9; }   /* Mibits -> GB(十进制) */
static double now_mono(void)  { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec/1e9; }
static double now_epoch(void) { struct timespec t; clock_gettime(CLOCK_REALTIME, &t); return t.tv_sec + t.tv_nsec/1e9; }
static void sleep_ms(long ms) { struct timespec t = { ms/1000, (ms%1000)*1000000L }; nanosleep(&t, NULL); }
static double delta(uint64_t cur, uint64_t prev) { return (cur >= prev) ? (double)(cur - prev) : 0.0; }
static int cmp_str(const void *a, const void *b) { return strcmp((const char *)a, (const char *)b); }
static int sw_index(const char *lbl) { for (int k=0;k<g_nsw;k++) if(!strcmp(g_labels[k],lbl)) return k; return -1; }
static int sw_shown(int idx) { return g_isw_all || (idx>=0 && idx<MAXSW && g_isw_mask[idx]); }

/* NSCQ 观察回调：按枚举顺序填 g_cur[] */
static void tput_cb(const nscq_uuid_t *dev, uint32_t port, nscq_rc_t rc,
                    nscq_link_throughput_t v, void *user) {
    (void)user;
    if (g_ncur >= MAXREC) return;
    rec_t *r = &g_cur[g_ncur++];
    r->port = port; r->rc = rc; r->rx = v.rx; r->tx = v.tx;
    nscq_label_t lbl; memset(&lbl, 0, sizeof lbl);
    if (dev && nscq_uuid_to_label(dev, &lbl, 0) == NSCQ_RC_SUCCESS)
        memcpy(r->sw, lbl.data, 64), r->sw[63] = '\0';
    else snprintf(r->sw, sizeof r->sw, "?");
}
static int read_once(nscq_session_t s) {
    g_ncur = 0;
    nscq_rc_t rc = nscq_session_path_observe(s, NSCQ_PATH_PORT_THROUGHPUT, NSCQ_FN(tput_cb), NULL, 0);
    return NSCQ_ERROR(rc) ? (int)rc : 0;
}

/* 后台 reader：不停读，刷新 g_latest */
static void *reader_main(void *arg) {
    nscq_session_t s = (nscq_session_t)arg;
    while (!g_stop) {
        if (read_once(s) != 0 || g_ncur == 0) { sleep_ms(50); continue; }
        double t = now_mono();
        pthread_mutex_lock(&g_mx);
        memcpy(g_latest, g_cur, sizeof(rec_t) * g_ncur);
        g_latest_n = g_ncur; g_latest_t = t; g_gen++;
        pthread_mutex_unlock(&g_mx);
        /* 无 sleep：背靠背读，节拍≈单次读延迟(~100ms) */
    }
    return NULL;
}

static void build_switch_index(rec_t *rec, int n) {
    g_nsw = 0;
    for (int i = 0; i < n; i++) {
        int f = 0; for (int k=0;k<g_nsw;k++) if(!strcmp(g_labels[k],rec[i].sw)){f=1;break;}
        if (!f && g_nsw < MAXSW) { memcpy(g_labels[g_nsw], rec[i].sw, 64); g_labels[g_nsw][63]='\0'; g_nsw++; }
    }
    qsort(g_labels, g_nsw, 64, cmp_str);
}
/* 建立 (switch,port)->下标 与 端口并集（用第一份快照） */
static void build_port_map(rec_t *rec, int n) {
    for (int k=0;k<MAXSW;k++) for (int p=0;p<NPORTID;p++) g_ri_of[k][p] = -1;
    int seen[NPORTID]; memset(seen, 0, sizeof seen);
    for (int i=0;i<n;i++) {
        int k = sw_index(rec[i].sw); int p = (int)rec[i].port;
        if (k>=0 && p>=0 && p<NPORTID) { g_ri_of[k][p] = i; seen[p] = 1; }
    }
    g_nunion = 0;
    for (int p=0;p<NPORTID;p++) if (seen[p]) g_union_ports[g_nunion++] = p;
}

/* 单值格式化：速率 5 位小数；raw = 整数 Mibits */
static void put_val(double mib, double dt) {
    if (g_raw) printf(" %14.0f", mib);
    else       printf(" %14.5f", mib_to_GB(mib) / dt);
}
static void put_val_csv(double mib, double dt) {
    if (g_raw) printf(",%.0f", mib);
    else       printf(",%.5f", mib_to_GB(mib) / dt);
}

static void print_header(void) {
    if (!g_header) return;
    const char *U = g_raw ? "Mib" : "GB/s";
    if (g_csv) {
        printf("epoch,dt_s,level,switch,port");
        if (g_fields & FLD_RX)  printf(",rx_%s", U);
        if (g_fields & FLD_TX)  printf(",tx_%s", U);
        if (g_fields & FLD_TOT) printf(",total_%s", U);
        printf("\n"); return;
    }
    printf("# nvswitch_traffic  src=NSCQ per-port throughput_counters  level=%s  unit=%s  interval=%ldms\n",
           g_level==0?"total":g_level==1?"switch":"port", U, g_delay_ms);
    if (g_level == 2) {   /* 宽表：每 switch 一行，端口做列 */
        printf("%-10s %-6s", "#time(s)", "switch");
        for (int j=0;j<g_nunion;j++) {
            int p = g_union_ports[j]; char lbl[24];
            if (g_fields & FLD_RX)  { snprintf(lbl,sizeof lbl,"p%d_rx",p);  printf(" %14s", lbl); }
            if (g_fields & FLD_TX)  { snprintf(lbl,sizeof lbl,"p%d_tx",p);  printf(" %14s", lbl); }
            if (g_fields & FLD_TOT) { snprintf(lbl,sizeof lbl,"p%d_tot",p); printf(" %14s", lbl); }
        }
        printf("\n"); return;
    }
    printf("%-10s %-6s", "#time(s)", g_level==1 ? "switch" : "fabric");
    if (g_fields & FLD_RX)  printf(" %14s", g_raw?"RX(Mib)":"RX(GB/s)");
    if (g_fields & FLD_TX)  printf(" %14s", g_raw?"TX(Mib)":"TX(GB/s)");
    if (g_fields & FLD_TOT) printf(" %14s", g_raw?"TOT(Mib)":"TOT(GB/s)");
    printf("\n");
}

int main(int argc, char **argv) {
    static struct option lo[] = { {"csv",no_argument,0,1000}, {"help",no_argument,0,'h'}, {0,0,0,0} };
    int opt;
    while ((opt = getopt_long(argc, argv, "e:i:l:d:c:rHh", lo, NULL)) != -1) {
        switch (opt) {
        case 'e': {
            g_fields = 0; char b[128]; snprintf(b,sizeof b,"%s",optarg);
            for (char *t=strtok(b,","); t; t=strtok(NULL,",")) {
                if(!strcmp(t,"rx"))g_fields|=FLD_RX; else if(!strcmp(t,"tx"))g_fields|=FLD_TX;
                else if(!strcmp(t,"total")||!strcmp(t,"tot"))g_fields|=FLD_TOT;
                else { fprintf(stderr,"unknown field '%s' (rx,tx,total)\n",t); return 2; }
            }
            if(!g_fields){fprintf(stderr,"no valid -e fields\n");return 2;} break; }
        case 'i': {
            if(!strcmp(optarg,"all")) g_isw_all=1;
            else { g_isw_all=0; memset(g_isw_mask,0,sizeof g_isw_mask);
                   char b[256]; snprintf(b,sizeof b,"%s",optarg);
                   for(char*t=strtok(b,",");t;t=strtok(NULL,",")){int x=atoi(t); if(x>=0&&x<MAXSW)g_isw_mask[x]=1;} }
            break; }
        case 'l':
            if(!strcmp(optarg,"total"))g_level=0; else if(!strcmp(optarg,"switch"))g_level=1;
            else if(!strcmp(optarg,"port"))g_level=2; else {fprintf(stderr,"bad -l\n");return 2;} break;
        case 'd': g_delay_ms = atol(optarg); if(g_delay_ms<1)g_delay_ms=1; break;
        case 'c': g_count = atol(optarg); break;
        case 'r': g_raw = 1; break;
        case 'H': g_header = 0; break;
        case 1000: g_csv = 1; break;
        case 'h':
            fputs("nvswitch_traffic [-e rx,tx,total] [-i sw|all] [-l total|switch|port] "
                  "[-d ms] [-c n] [-r] [--csv] [-H]\n"
                  "  实时打印过 NVSwitch 的 NVLink 流量(源: NSCQ per-port throughput_counters)。\n"
                  "  例: nvswitch_traffic -d 100            # 每 ~100ms 每台 switch 的 rx/tx/total GB/s\n"
                  "      nvswitch_traffic -l total -e total # 全 fabric 总带宽一行\n"
                  "      nvswitch_traffic -l port -i 1      # sw1 每端口(宽表)\n", stderr);
            return 0;
        default: return 2;
        }
    }

    nscq_session_result_t sr = nscq_session_create(NSCQ_SESSION_CREATE_MOUNT_DEVICES);
    if (NSCQ_ERROR(sr.rc)) { fprintf(stderr, "nscq_session_create failed rc=%d\n", (int)sr.rc); return 1; }
    nscq_session_t s = sr.session;
    signal(SIGINT, on_sigint); signal(SIGTERM, on_sigint);

    /* 起后台 reader；等它出第一份快照（含重试，避开与 nv-hostengine 抢 NSCQ 的偶发空读） */
    pthread_t rt; pthread_create(&rt, NULL, reader_main, (void *)s);
    rec_t prev[MAXREC]; int prev_n = 0; double prev_t = 0; uint64_t prev_gen = 0;
    for (int w=0; ; w++) {
        pthread_mutex_lock(&g_mx);
        if (g_gen > 0) { memcpy(prev, g_latest, sizeof(rec_t)*g_latest_n); prev_n=g_latest_n; prev_t=g_latest_t; prev_gen=g_gen; }
        pthread_mutex_unlock(&g_mx);
        if (prev_gen > 0) break;
        if (g_stop || w >= 40) {   /* ~40*100ms = 4s */
            fprintf(stderr, "no NVSwitch ports found（Fabric Manager 在跑吗? libnvidia-nscq 装了吗? "
                            "偶发与 nv-hostengine 抢 NSCQ——直接重跑一次通常就好）\n");
            g_stop = 1; pthread_join(rt, NULL); nscq_session_destroy(s); return 1;
        }
        sleep_ms(100);
    }
    build_switch_index(prev, prev_n);
    build_port_map(prev, prev_n);
    print_header();

    double t0 = now_mono();
    long tick = 0;
    static double sw_rx[MAXSW], sw_tx[MAXSW];

    while (!g_stop && (g_count == 0 || tick < g_count)) {
        sleep_ms(g_delay_ms);
        /* 取一份"比上次打印更新"的快照；若 -d < 读延迟则短暂等新数据(不打重复行) */
        rec_t cur[MAXREC]; int cur_n = 0; double cur_t = 0; uint64_t cur_gen = 0;
        for (;;) {
            pthread_mutex_lock(&g_mx);
            cur_gen = g_gen;
            if (cur_gen != prev_gen) { memcpy(cur, g_latest, sizeof(rec_t)*g_latest_n); cur_n=g_latest_n; cur_t=g_latest_t; }
            pthread_mutex_unlock(&g_mx);
            if (cur_gen != prev_gen || g_stop) break;
            sleep_ms(5);
        }
        if (g_stop) break;
        if (cur_n != prev_n) {   /* 端口数变了：重建、跳过 */
            build_switch_index(cur, cur_n); build_port_map(cur, cur_n);
            memcpy(prev, cur, sizeof(rec_t)*cur_n); prev_n=cur_n; prev_t=cur_t; prev_gen=cur_gen; continue;
        }
        double dt = cur_t - prev_t; if (dt <= 0) dt = g_delay_ms/1000.0;
        double epoch = now_epoch(), tsec = cur_t - t0;

        if (g_level == 2) {                          /* 宽表：每 switch 一行，端口做列 */
            for (int k=0;k<g_nsw;k++) {
                if (!sw_shown(k)) continue;
                if (g_csv) {                          /* CSV 仍用长表(每端口一行)便于解析 */
                    for (int j=0;j<g_nunion;j++) {
                        int p=g_union_ports[j], ri=g_ri_of[k][p]; if(ri<0)continue;
                        double drx=delta(cur[ri].rx,prev[ri].rx), dtx=delta(cur[ri].tx,prev[ri].tx);
                        printf("%.3f,%.3f,port,%d,%d", epoch, dt, k, p);
                        if(g_fields&FLD_RX)put_val_csv(drx,dt);
                        if(g_fields&FLD_TX)put_val_csv(dtx,dt);
                        if(g_fields&FLD_TOT)put_val_csv(drx+dtx,dt);
                        printf("\n");
                    }
                } else {
                    printf("%-10.2f %-6d", tsec, k);
                    for (int j=0;j<g_nunion;j++) {
                        int p=g_union_ports[j], ri=g_ri_of[k][p];
                        double drx=ri>=0?delta(cur[ri].rx,prev[ri].rx):0, dtx=ri>=0?delta(cur[ri].tx,prev[ri].tx):0;
                        if(g_fields&FLD_RX)put_val(drx,dt);
                        if(g_fields&FLD_TX)put_val(dtx,dt);
                        if(g_fields&FLD_TOT)put_val(drx+dtx,dt);
                    }
                    printf("\n");
                }
            }
        } else {                                     /* switch / total 聚合 */
            memset(sw_rx,0,sizeof sw_rx); memset(sw_tx,0,sizeof sw_tx);
            double trx=0, ttx=0;
            for (int i=0;i<cur_n;i++) {
                if (strcmp(prev[i].sw,cur[i].sw)||prev[i].port!=cur[i].port) continue;
                int k=sw_index(cur[i].sw); if(k<0)continue;
                double drx=delta(cur[i].rx,prev[i].rx), dtx=delta(cur[i].tx,prev[i].tx);
                sw_rx[k]+=drx; sw_tx[k]+=dtx; trx+=drx; ttx+=dtx;
            }
            if (g_level == 1) {
                for (int k=0;k<g_nsw;k++) {
                    if(!sw_shown(k))continue;
                    if (g_csv) { printf("%.3f,%.3f,switch,%d,-1", epoch, dt, k);
                        if(g_fields&FLD_RX)put_val_csv(sw_rx[k],dt); if(g_fields&FLD_TX)put_val_csv(sw_tx[k],dt);
                        if(g_fields&FLD_TOT)put_val_csv(sw_rx[k]+sw_tx[k],dt); printf("\n"); }
                    else { printf("%-10.2f %-6d", tsec, k);
                        if(g_fields&FLD_RX)put_val(sw_rx[k],dt); if(g_fields&FLD_TX)put_val(sw_tx[k],dt);
                        if(g_fields&FLD_TOT)put_val(sw_rx[k]+sw_tx[k],dt); printf("\n"); }
                }
            } else {
                if (g_csv) { printf("%.3f,%.3f,total,-1,-1", epoch, dt);
                    if(g_fields&FLD_RX)put_val_csv(trx,dt); if(g_fields&FLD_TX)put_val_csv(ttx,dt);
                    if(g_fields&FLD_TOT)put_val_csv(trx+ttx,dt); printf("\n"); }
                else { printf("%-10.2f %-6s", tsec, "ALL");
                    if(g_fields&FLD_RX)put_val(trx,dt); if(g_fields&FLD_TX)put_val(ttx,dt);
                    if(g_fields&FLD_TOT)put_val(trx+ttx,dt); printf("\n"); }
            }
        }
        fflush(stdout);
        memcpy(prev, cur, sizeof(rec_t)*cur_n); prev_n=cur_n; prev_t=cur_t; prev_gen=cur_gen;
        tick++;
    }

    g_stop = 1; pthread_join(rt, NULL);
    nscq_session_destroy(s);
    return 0;
}
