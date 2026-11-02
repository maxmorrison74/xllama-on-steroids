// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
//
// Built on top of Xllama by Gianluca Mazza (https://github.com/gianlucamazza/xllama)
#include "moe_file.h"

#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <sys/stat.h>

#ifdef _WIN32
    #include <windows.h>

namespace {
std::wstring widen_path(const std::string& path) {
    const int n = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, nullptr, 0);
    std::wstring w(static_cast<std::size_t>(n > 0 ? n - 1 : 0), L'\0');
    if (n > 1)
        MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, w.data(), n);
    return w;
}
} // namespace
#else
    #include <fcntl.h>
    #include <unistd.h>
#endif

namespace xllama::experimental {
namespace {
void* aligned_alloc(std::size_t bytes) {
#ifdef _WIN32
    return _aligned_malloc(bytes, 4096);
#else
    void* p = nullptr;
    if (posix_memalign(&p, 4096, bytes) != 0)
        return nullptr;
    return p;
#endif
}
void aligned_free(void* p) {
#ifdef _WIN32
    _aligned_free(p);
#else
    ::free(p);
#endif
}
} // namespace

struct MoeFile::Impl {
#ifdef _WIN32
    HANDLE handle = INVALID_HANDLE_VALUE;
#else
    int fd = -1;
#endif
    bool unbuffered = false;
    std::uint64_t file_size = 0;

    ~Impl() {
        close();
    }
    void close() {
#ifdef _WIN32
        if (handle != INVALID_HANDLE_VALUE) {
            CloseHandle(handle);
            handle = INVALID_HANDLE_VALUE;
        }
#else
        if (fd >= 0) {
            ::close(fd);
            fd = -1;
        }
#endif
    }
};

MoeFile::MoeFile() = default;
MoeFile::~MoeFile() = default;

bool MoeFile::open(const std::string& path, bool prefer_unbuffered, bool& unbuffered) {
    impl_.reset(new Impl());
#ifdef _WIN32
    DWORD flags = FILE_FLAG_SEQUENTIAL_SCAN | FILE_ATTRIBUTE_NORMAL;
    if (prefer_unbuffered)
        flags |= FILE_FLAG_NO_BUFFERING;

    CREATEFILE2_EXTENDED_PARAMETERS params{};
    params.dwSize = sizeof(params);
    params.dwFileAttributes = FILE_ATTRIBUTE_NORMAL;
    params.dwFileFlags = flags;
    params.lpSecurityAttributes = nullptr;
    params.hTemplateFile = nullptr;

    impl_->handle = CreateFile2(widen_path(path).c_str(), GENERIC_READ, FILE_SHARE_READ,
                                OPEN_EXISTING, &params);
    if (impl_->handle == INVALID_HANDLE_VALUE && prefer_unbuffered) {
        params.dwFileFlags = FILE_FLAG_SEQUENTIAL_SCAN | FILE_ATTRIBUTE_NORMAL;
        impl_->handle = CreateFile2(widen_path(path).c_str(), GENERIC_READ, FILE_SHARE_READ,
                                    OPEN_EXISTING, &params);
        if (impl_->handle != INVALID_HANDLE_VALUE) {
            impl_->unbuffered = false;
            unbuffered = false;
        } else {
            impl_.reset();
            return false;
        }
    } else if (impl_->handle == INVALID_HANDLE_VALUE) {
        impl_.reset();
        return false;
    } else {
        impl_->unbuffered = prefer_unbuffered;
        unbuffered = prefer_unbuffered;
    }
    LARGE_INTEGER li{};
    if (GetFileSizeEx(impl_->handle, &li))
        impl_->file_size = static_cast<std::uint64_t>(li.QuadPart);
#else
    int flags = O_RDONLY;
    #ifdef O_DIRECT
    if (prefer_unbuffered)
        flags |= O_DIRECT;
    #endif
    impl_->fd = ::open(path.c_str(), flags);
    if (impl_->fd < 0) {
        if (prefer_unbuffered) {
            impl_->fd = ::open(path.c_str(), O_RDONLY);
            if (impl_->fd >= 0) {
                impl_->unbuffered = false;
                unbuffered = false;
            } else {
                impl_.reset();
                return false;
            }
        } else {
            impl_.reset();
            return false;
        }
    } else {
        impl_->unbuffered = prefer_unbuffered;
        unbuffered = prefer_unbuffered;
    }
    struct stat st{};
    if (::fstat(impl_->fd, &st) == 0)
        impl_->file_size = static_cast<std::uint64_t>(st.st_size);
#endif
    valid_ = true;
    return true;
}

std::uint64_t MoeFile::size() const {
    if (!impl_)
        return 0;
    return impl_->file_size;
}

std::size_t MoeFile::read(std::uint64_t offset, void* destination, std::size_t bytes) {
    if (!impl_)
        throw std::runtime_error("MoeFile read on closed handle");
    if (bytes == 0)
        return 0;
#ifdef _WIN32
    OVERLAPPED ov{};
    ov.Offset = static_cast<DWORD>(offset & 0xffffffffu);
    ov.OffsetHigh = static_cast<DWORD>(offset >> 32);
    DWORD got = 0;
    if (!ReadFile(impl_->handle, destination, static_cast<DWORD>(bytes), &got, &ov))
        throw std::runtime_error("MoeFile ReadFile failed");
    return static_cast<std::size_t>(got);
#else
    std::size_t done = 0;
    while (done < bytes) {
        const ssize_t n = ::pread(impl_->fd, static_cast<char*>(destination) + done, bytes - done,
                                  static_cast<off_t>(offset + done));
        if (n <= 0)
            throw std::runtime_error("MoeFile pread failed");
        done += static_cast<std::size_t>(n);
    }
    return done;
#endif
}

} // namespace xllama::experimental
