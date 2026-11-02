// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
//
// Built on top of Xllama by Gianluca Mazza (https://github.com/gianlucamazza/xllama)
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace xllama::experimental {

// Positioned file reads, app-portable for UWP/AppContainer. O_DIRECT-equivalent
// reads are requested where supported; the caller must still verify alignment
// and that the destination is memory-resident when it matters.
class MoeFile {
  public:
    MoeFile();
    ~MoeFile();
    MoeFile(const MoeFile&) = delete;
    MoeFile& operator=(const MoeFile&) = delete;

    // Open for reading. `prefer_unbuffered` asks to bypass the OS page cache;
    // `unbuffered` reports whether the requested mode is actually in effect.
    bool open(const std::string& path, bool prefer_unbuffered, bool& unbuffered);
    std::uint64_t size() const;
    // Reads [offset, offset+bytes). Returns bytes transferred, may be short at
    // end of file. Caller-owned destination must be aligned if unbuffered.
    std::size_t read(std::uint64_t offset, void* destination, std::size_t bytes);
    bool valid() const noexcept {
        return valid_;
    }

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    bool valid_ = false;
};

} // namespace xllama::experimental
