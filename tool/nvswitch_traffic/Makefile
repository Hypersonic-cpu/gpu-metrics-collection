# nvswitch_traffic —— NSCQ 直读 NVSwitch per-port 流量的 CLI
# 依赖：运行时 libnvidia-nscq.so（随 nvidia 驱动/Fabric Manager 安装）。头文件自包含(nscq_min.h)。
CC      ?= gcc
CFLAGS  ?= -O2 -Wall -Wextra -std=c11 -Wno-misleading-indentation
NSCQ_LIBDIR ?= /usr/lib/x86_64-linux-gnu
LDFLAGS ?= -L$(NSCQ_LIBDIR)
LDLIBS  ?= -lnvidia-nscq -lpthread

BIN     := nvswitch_traffic
PREFIX  ?= /usr/local

$(BIN): $(BIN).c nscq_min.h
	$(CC) $(CFLAGS) -o $@ $(BIN).c $(LDFLAGS) $(LDLIBS)

clean:
	rm -f $(BIN)

# 可选：装到 PATH，之后任何脚本可直接 `nvswitch_traffic ...`
install: $(BIN)
	install -d $(DESTDIR)$(PREFIX)/bin
	install -m 0755 $(BIN) $(DESTDIR)$(PREFIX)/bin/$(BIN)

uninstall:
	rm -f $(DESTDIR)$(PREFIX)/bin/$(BIN)

.PHONY: clean install uninstall
