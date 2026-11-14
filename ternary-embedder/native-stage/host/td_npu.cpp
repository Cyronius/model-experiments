// td_npu: a C API over XRT for the tiny planner's host driver (run_planner.py, ctypes).
// One handle per (xclbin, insts.bin); the dispatch is open_kernels/harness's classic path.
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <stdexcept>
#include <vector>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_hw_context.h"
#include "xrt/xrt_kernel.h"
#include "xrt/experimental/xrt_ext.h"

#ifdef _WIN32
#define TDN_EXPORT __declspec(dllexport)
#else
#define TDN_EXPORT __attribute__((visibility("default")))
#endif

namespace {
std::unique_ptr<xrt::device> g_dev;
thread_local std::string last_error;

struct Td {
  xrt::hw_context ctx;
  xrt::kernel kern;
  xrt::bo instr;
  int nwords = 0;
  std::vector<xrt::bo> bufs;
  std::unique_ptr<xrt::run> run;
};
}  // namespace

extern "C" {

TDN_EXPORT const char *tdn_last_error() { return last_error.c_str(); }

TDN_EXPORT void *tdn_open(const char *xclbin, const char *insts) {
  last_error.clear();
  try {
    if (!g_dev) g_dev = std::make_unique<xrt::device>(0u);
    auto t = std::make_unique<Td>();
    xrt::xclbin x{std::string(xclbin)};
    t->ctx = xrt::hw_context(*g_dev, g_dev->register_xclbin(x));
    t->kern = xrt::kernel(t->ctx, "MLIR_AIE");
    std::ifstream f(insts, std::ios::binary);
    std::vector<char> w((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (!f || w.empty() || w.size() % 4) throw std::runtime_error("Invalid instruction file");
    t->nwords = static_cast<int>(w.size() / 4);
    t->instr = xrt::bo(*g_dev, w.size(), xrt::bo::flags::cacheable, t->kern.group_id(1));
    std::memcpy(t->instr.map<void *>(), w.data(), w.size());
    t->instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    return t.release();
  } catch (const std::exception &e) {
    last_error = e.what();
    return nullptr;
  }
}

// a zeroed buffer at kernel argument 3 + its index; returns its host mapping
TDN_EXPORT void *tdn_buf(void *h, size_t size) {
  last_error.clear();
  try {
    auto *t = static_cast<Td *>(h);
    size_t pad = (size + 4095) & ~size_t(4095);
    t->bufs.emplace_back(xrt::ext::bo(*g_dev, pad));
    auto *m = t->bufs.back().map<uint8_t *>();
    std::memset(m, 0, pad);
    t->bufs.back().sync(XCL_BO_SYNC_BO_TO_DEVICE);
    return m;

  } catch (const std::exception &e) {
    last_error = e.what();
    return nullptr;
  }
}

TDN_EXPORT void tdn_sync(void *h, int i, int to_device, size_t size, size_t off) {
  last_error.clear();
  try {
    auto *t = static_cast<Td *>(h);
    t->bufs.at(i).sync(to_device ? XCL_BO_SYNC_BO_TO_DEVICE : XCL_BO_SYNC_BO_FROM_DEVICE, size, off);

  } catch (const std::exception &e) {
    last_error = e.what();
    return;
  }
}

// start + wait; wall ms, or -state on failure
TDN_EXPORT double tdn_run(void *h) {
  last_error.clear();
  try {
    auto *t = static_cast<Td *>(h);
    if (!t->run) {
    t->run = std::make_unique<xrt::run>(t->kern);
    t->run->set_arg(0, 3);
    t->run->set_arg(1, t->instr);
    t->run->set_arg(2, t->nwords);
    for (size_t i = 0; i < t->bufs.size(); ++i) t->run->set_arg(static_cast<int>(3 + i), t->bufs[i]);
    }
    auto t0 = std::chrono::steady_clock::now();
    t->run->start();
    auto st = t->run->wait(std::chrono::milliseconds(20000));
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return st == ERT_CMD_STATE_COMPLETED ? ms : -static_cast<double>(st);

  } catch (const std::exception &e) {
    last_error = e.what();
    return -1.;
  }
}

TDN_EXPORT void tdn_close(void *h) { delete static_cast<Td *>(h); }
}
