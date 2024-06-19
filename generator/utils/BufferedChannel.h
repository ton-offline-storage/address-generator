#pragma once

#include <utility>
#include <optional>
#include <queue>
#include <mutex>
#include <condition_variable>

template <class T>
class BufferedChannel {
    bool cl_ = false;
    size_t n_;
    std::queue<T> q_;
    std::mutex m_;
    std::condition_variable cs_;
    std::condition_variable cr_;

public:
    explicit BufferedChannel(int size) : n_(size) {
    }

    void Send(const T& value) {
        std::unique_lock<std::mutex> lock(m_);
        while (q_.size() >= n_ && !cl_) {
            cs_.wait(lock);
        }
        if (cl_) {
            throw std::runtime_error("Send to closed");
        }
        q_.push(value);
        cr_.notify_one();
    }

    std::optional<T> Recv() {
        std::unique_lock<std::mutex> lock(m_);
        while (q_.empty() && !cl_) {
            cr_.wait(lock);
        }
        if (q_.empty()) {
            return std::nullopt;
        } else {
            T res = q_.front();
            q_.pop();
            cs_.notify_one();
            return res;
        }
    }

    void Close() {
        std::unique_lock<std::mutex> lock(m_);
        cl_ = true;
        cs_.notify_all();
        cr_.notify_all();
    }
};