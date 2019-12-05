//
// Created by Liu Guanfu on 12/5/19.
//

#ifndef GBDT_CONCURRENCY_H
#define GBDT_CONCURRENCY_H

#include <climits>
#include <condition_variable>
#include <mutex>
#include "util.hpp"
#include <cassert>


namespace gbdt {

/**
 * Single writer, multiple readers
 * Writer has higher priority, so further readers will block
 */
  class ReaderWriterLatch {
    using mutex_t = std::mutex;
    using cond_t = std::condition_variable;

  public:
    ReaderWriterLatch() = default;
    ~ReaderWriterLatch() { std::lock_guard<mutex_t> guard(mutex_); }

    DISALLOW_COPY_AND_ASSIGN(ReaderWriterLatch);

    /**
     * Acquire a write latch.
     */
    void WLock() {
      std::unique_lock<mutex_t> latch(mutex_);
      writer_entered_ = true;
      while (reader_count_ > 0) {
        writer_.wait(latch);
      }
    }

    /**
     * Release a write latch.
     */
    void WUnlock() {
      std::lock_guard<mutex_t> guard(mutex_);
      writer_entered_ = false;
      reader_.notify_all();
    }

    /**
     * Acquire a read latch.
     */
    void RLock() {
      std::unique_lock<mutex_t> latch(mutex_);
      while (writer_entered_) {
        reader_.wait(latch);
      }
      reader_count_++;
    }

    /**
     * Release a read latch.
     */
    void RUnlock() {
      std::lock_guard<mutex_t> guard(mutex_);
      reader_count_--;
      if (writer_entered_) {
        if (reader_count_ == 0) {
          writer_.notify_one();
        }
      }
    }

  private:
    mutex_t mutex_;
    cond_t writer_;
    cond_t reader_;
    uint32_t reader_count_{0};
    bool writer_entered_{false};
  };




}  // namespace bustub


#endif //GBDT_CONCURRENCY_H
