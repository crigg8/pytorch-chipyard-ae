#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include "mlir/ExecutionEngine/RunnerUtils.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <cerrno>
#include <vector>

#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>

static inline float bitcast_u32_to_f32(uint32_t bits) {
  float out;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

static inline uint32_t bitcast_f32_to_u32(float v) {
  uint32_t out;
  std::memcpy(&out, &v, sizeof(out));
  return out;
}

// Some riscv toolchains in this setup do not provide these libgcc helpers.
extern "C" float __extendhfsf2(uint16_t h) {
  const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
  uint32_t exp = (h >> 10) & 0x1fu;
  uint32_t frac = h & 0x03ffu;
  uint32_t out_bits = 0;

  if (exp == 0) {
    if (frac == 0) {
      out_bits = sign;
    } else {
      // Normalize half subnormal.
      int32_t e = -14;
      while ((frac & 0x0400u) == 0) {
        frac <<= 1;
        --e;
      }
      frac &= 0x03ffu;
      const uint32_t exp32 = static_cast<uint32_t>(e + 127);
      out_bits = sign | (exp32 << 23) | (frac << 13);
    }
  } else if (exp == 0x1fu) {
    out_bits = sign | 0x7f800000u | (frac << 13);
  } else {
    const uint32_t exp32 = exp + static_cast<uint32_t>(127 - 15);
    out_bits = sign | (exp32 << 23) | (frac << 13);
  }

  return bitcast_u32_to_f32(out_bits);
}

extern "C" uint16_t __truncsfhf2(float f) {
  const uint32_t bits = bitcast_f32_to_u32(f);
  const uint16_t sign = static_cast<uint16_t>((bits >> 16) & 0x8000u);
  const uint32_t exp = (bits >> 23) & 0xffu;
  const uint32_t frac = bits & 0x7fffffu;

  // NaN / Inf.
  if (exp == 0xffu) {
    if (frac == 0) {
      return static_cast<uint16_t>(sign | 0x7c00u);
    }
    uint16_t payload = static_cast<uint16_t>(frac >> 13);
    if (payload == 0) payload = 1;
    return static_cast<uint16_t>(sign | 0x7c00u | payload);
  }

  const int32_t exp_unbiased = static_cast<int32_t>(exp) - 127;
  int32_t half_exp = exp_unbiased + 15;

  // Overflow -> Inf.
  if (half_exp >= 31) {
    return static_cast<uint16_t>(sign | 0x7c00u);
  }

  // Underflow -> subnormal/zero.
  if (half_exp <= 0) {
    if (half_exp < -10) {
      return sign;
    }
    uint32_t mant = frac | 0x00800000u;
    const int32_t shift = 14 - half_exp;
    uint32_t out_mant = mant >> shift;
    const uint32_t rem_mask = (1u << shift) - 1u;
    const uint32_t rem = mant & rem_mask;
    const uint32_t halfway = 1u << (shift - 1);
    if (rem > halfway || (rem == halfway && (out_mant & 1u))) {
      ++out_mant;
    }
    return static_cast<uint16_t>(sign | static_cast<uint16_t>(out_mant));
  }

  uint32_t half_frac = frac >> 13;
  const uint32_t rem = frac & 0x1fffu;
  if (rem > 0x1000u || (rem == 0x1000u && (half_frac & 1u))) {
    ++half_frac;
    if (half_frac == 0x0400u) {
      half_frac = 0;
      ++half_exp;
      if (half_exp >= 31) {
        return static_cast<uint16_t>(sign | 0x7c00u);
      }
    }
  }

  return static_cast<uint16_t>(
      sign | (static_cast<uint16_t>(half_exp) << 10) |
      static_cast<uint16_t>(half_frac));
}

// Keep these structs identical to host.cpp declarations
static constexpr uint32_t PACKET_VERSION = 1;
static constexpr int MAX_DIMS = 5;

static constexpr uint32_t KIND_SCALAR = 0;
static constexpr uint32_t KIND_TENSOR = 1;

static constexpr uint32_t SCALAR_INT64   = 1;
static constexpr uint32_t SCALAR_FLOAT64 = 2;
static constexpr uint32_t SCALAR_BOOL    = 3;
static constexpr uint32_t SCALAR_BYTES = 4;


struct PacketHeader {
  char     magic[8];
  uint32_t version;
  uint32_t n_args;
  uint32_t entry_size;
  uint32_t reserved;
  uint64_t payload_offset;
};

struct PacketEntry {
  uint32_t kind;
  uint32_t scalar_type;
  uint32_t tensor_dtype;
  uint32_t ndim;
  uint32_t flags;
  uint32_t reserved0;

  uint64_t sizes[MAX_DIMS];
  int64_t  strides[MAX_DIMS];

  uint64_t nbytes;
  uint64_t data_offset;
  uint64_t reserved1;
};

struct PacketLoaded {
  PacketHeader hdr{};
  std::vector<PacketEntry> entries;
  std::vector<void*> buffers;

  ~PacketLoaded();
};

PacketLoaded::~PacketLoaded() {
  for (void* p : buffers) std::free(p);
  buffers.clear();
}

static void read_exact(int fd, void* buf, size_t n) {
  uint8_t* p = reinterpret_cast<uint8_t*>(buf);
  size_t got = 0;
  while (got < n) {
    ssize_t r = ::read(fd, p + got, n - got);
    if (r <= 0) {
      std::fprintf(
          stderr,
          "[chipyard-driver] read_exact failed fd=%d got=%zu need=%zu r=%zd errno=%d (%s)\n",
          fd, got, n, r, errno, std::strerror(errno));
      std::abort();
    }
    got += static_cast<size_t>(r);
  }
}

static void pread_exact(int fd, void* buf, size_t n, off_t off) {
  uint8_t* p = reinterpret_cast<uint8_t*>(buf);
  size_t got = 0;
  while (got < n) {
    ssize_t r = ::pread(fd, p + got, n - got, off + static_cast<off_t>(got));
    if (r <= 0) {
      std::fprintf(
          stderr,
          "[chipyard-driver] pread_exact failed fd=%d off=%lld got=%zu need=%zu r=%zd errno=%d (%s)\n",
          fd, static_cast<long long>(off), got, n, r, errno, std::strerror(errno));
      std::abort();
    }
    got += static_cast<size_t>(r);
  }
}

PacketLoaded load_packet(const char* packet_path) {
  int fd = ::open(packet_path, O_RDONLY);
  if (fd < 0) {
    std::fprintf(
        stderr,
        "[chipyard-driver] load_packet failed path=%s errno=%d (%s)\n",
        packet_path, errno, std::strerror(errno));
    std::abort();
  }

  PacketLoaded out;

  // header
  read_exact(fd, &out.hdr, sizeof(PacketHeader));

  // validate
  const char expected_magic[8] = {'T','T','C','I','P','K','T','1'};
  if (std::memcmp(out.hdr.magic, expected_magic, sizeof(expected_magic)) != 0) {
    std::fprintf(stderr, "[chipyard-driver] bad packet magic\n");
    std::abort();
  }

  // entries
  out.entries.resize(out.hdr.n_args);
  read_exact(fd, out.entries.data(), out.hdr.n_args * sizeof(PacketEntry));

  // payloads
  out.buffers.resize(out.hdr.n_args, nullptr);

  for (uint32_t i = 0; i < out.hdr.n_args; ++i) {
    const PacketEntry& e = out.entries[i];

    if (e.nbytes == 0) {
      out.buffers[i] = nullptr;
      continue;
    }

    void* buf = std::malloc(static_cast<size_t>(e.nbytes));
    if (buf == nullptr) {
      std::fprintf(
          stderr,
          "[chipyard-driver] malloc failed for entry[%u] nbytes=%llu\n",
          i,
          static_cast<unsigned long long>(e.nbytes));
      std::abort();
    }

    pread_exact(fd, buf, static_cast<size_t>(e.nbytes), static_cast<off_t>(e.data_offset));
    out.buffers[i] = buf;

  }

  ::close(fd);
  return out;
}

static void write_exact(int fd, const void* buf, size_t n) {
  const uint8_t* p = reinterpret_cast<const uint8_t*>(buf);
  size_t left = n;
  while (left > 0) {
    ssize_t w = ::write(fd, p, left);
    p += static_cast<size_t>(w);
    left -= static_cast<size_t>(w);
  }
}

void writeback_packet(const char* packet_path, const PacketLoaded& pkt) {
  int fd = ::open(packet_path, O_RDWR);  // O_WRONLY 말고 O_RDWR 권장

  // Write back updated entry metadata (e.g., cycles in reserved1).
  if (!pkt.entries.empty()) {
    if (::lseek(fd, static_cast<off_t>(sizeof(PacketHeader)), SEEK_SET) != (off_t)-1) {
      write_exact(fd, pkt.entries.data(), pkt.hdr.n_args * sizeof(PacketEntry));
    }
  }

  for (uint32_t i = 0; i < pkt.hdr.n_args; ++i) {
    const PacketEntry& e = pkt.entries[i];
    void* buf = pkt.buffers[i];

    if (e.nbytes == 0 || !buf) continue;

    if (::lseek(fd, static_cast<off_t>(e.data_offset), SEEK_SET) != (off_t)-1) {
      write_exact(fd, buf, static_cast<size_t>(e.nbytes));
    }
  }

  ::close(fd);
}

/*-------------scalar--------------*/
int64_t packet_get_i64(const PacketEntry& e, void* payload) {
  switch (e.scalar_type) {
    case SCALAR_INT64:
      return *reinterpret_cast<int64_t*>(payload);
    case SCALAR_FLOAT64:
      return static_cast<int64_t>(*reinterpret_cast<double*>(payload));
    case SCALAR_BOOL:
      return *reinterpret_cast<uint64_t*>(payload) != 0 ? 1 : 0;
    default:
      return 0;
  }
}

int32_t packet_get_i32(const PacketEntry& e, void* payload) {
  switch (e.scalar_type) {
    case SCALAR_INT64:
      return static_cast<int32_t>(*reinterpret_cast<int64_t*>(payload));
    case SCALAR_FLOAT64:
      return static_cast<int32_t>(*reinterpret_cast<double*>(payload));
    case SCALAR_BOOL:
      return *reinterpret_cast<uint64_t*>(payload) != 0 ? 1 : 0;
    default:
      return 0;
  }
}

double packet_get_f64(const PacketEntry& e, void* payload) {
  switch (e.scalar_type) {
    case SCALAR_FLOAT64:
      return *reinterpret_cast<double*>(payload);
    case SCALAR_INT64:
      return static_cast<double>(*reinterpret_cast<int64_t*>(payload));
    case SCALAR_BOOL:
      return *reinterpret_cast<uint64_t*>(payload) != 0 ? 1.0 : 0.0;
    default:
      return 0.0;
  }
}

float packet_get_f32(const PacketEntry& e, void* payload) {
  switch (e.scalar_type) {
    case SCALAR_FLOAT64:
      return static_cast<float>(*reinterpret_cast<double*>(payload));
    case SCALAR_INT64:
      return static_cast<float>(*reinterpret_cast<int64_t*>(payload));
    case SCALAR_BOOL:
      return *reinterpret_cast<uint64_t*>(payload) != 0 ? 1.0f : 0.0f;
    default:
      return 0.0f;
  }
}

bool packet_get_bool(const PacketEntry& e, void* payload) {
  uint64_t v = *reinterpret_cast<uint64_t*>(payload);
  return v != 0;
}
