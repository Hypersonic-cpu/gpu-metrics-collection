/* nscq_min.h —— libnvidia-nscq 的最小 ABI 声明（只取本工具用到的部分），使工具自包含、
 * 不依赖 DCGM SDK 头文件路径。ABI 依据 NVIDIA NSCQ 头（driver 590.x，NSCQ_1.0）：
 *   ~/Repos/DCGM/sdk/nvidia/nscq/{nscq.h,path.h}。
 * 运行时链接系统的 libnvidia-nscq.so（随 nvidia 驱动/Fabric Manager 安装）。
 */
#ifndef NSCQ_MIN_H
#define NSCQ_MIN_H
#include <stdint.h>

typedef int8_t nscq_rc_t;                 /* <0 error, 0 success, >0 warning */
typedef struct nscq_session_st *nscq_session_t;
typedef void (*nscq_fn_t)(void);

typedef struct { uint8_t bytes[16]; } nscq_uuid_t;
typedef struct { char data[64]; }     nscq_label_t;
typedef struct { uint64_t rx; uint64_t tx; } nscq_link_throughput_t;  /* 单位 Mibits，累计计数器 */

/* 与 NSCQ 头里 _NSCQ_RESULT_TYPE(nscq_session_t, session) 展开一致（rc 后 8 字节对齐到 session） */
typedef struct { nscq_rc_t rc; nscq_session_t session; } nscq_session_result_t;

#define NSCQ_RC_SUCCESS 0
#define NSCQ_ERROR(r)   ((int8_t)(r) < 0)
#define NSCQ_WARNING(r) ((int8_t)(r) > 0)
#define NSCQ_SESSION_CREATE_MOUNT_DEVICES (0x1u)
#define NSCQ_FN(fn) ((nscq_fn_t)&fn)

/* NSCQ 路径：per-port 吞吐累计计数器（DCGM 未接的那条） */
#define NSCQ_PATH_PORT_THROUGHPUT "/{nvswitch}/nvlink/{port}/throughput_counters"

#ifdef __cplusplus
extern "C" {
#endif
nscq_session_result_t nscq_session_create(uint32_t flags);
void                  nscq_session_destroy(nscq_session_t session);
nscq_rc_t             nscq_session_path_observe(nscq_session_t session, const char *path,
                                                nscq_fn_t cb, void *user, uint32_t flags);
nscq_rc_t             nscq_uuid_to_label(const nscq_uuid_t *uuid, nscq_label_t *label, uint32_t flags);
#ifdef __cplusplus
}
#endif
#endif /* NSCQ_MIN_H */
