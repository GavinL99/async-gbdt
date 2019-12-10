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
#include <atomic>


namespace gbdt {

/**
 * Single writer, multiple readers
 * Writer has higher priority, so further readers will block
 */
  using mutex_t = std::mutex;
  using cond_t = std::condition_variable;
  using uniq_lock = std::unique_lock<std::mutex>;

  class ReaderWriterLatch {

  public:
    ReaderWriterLatch() = default;

    ~ReaderWriterLatch() {
      uniq_lock latch(mutex_);
    }

    /**
     * Acquire a write latch.
     */
    void WLock() {
      uniq_lock latch(mutex_);
      while (writer_entered_) {
        reader_.wait(latch);
      }
      writer_entered_ = true;
      while (reader_count_ > 0) {
        writer_.wait(latch);
      }
    }

    /**
     * Release a write latch.
     */
    void WUnlock() {
      uniq_lock latch(mutex_);
      writer_entered_ = false;
      reader_.notify_all();
    }

    /**
     * Acquire a read latch.
     */
    void RLock() {
      uniq_lock latch(mutex_);
      while (writer_entered_) {
        reader_.wait(latch);
      }
      reader_count_++;
    }

    /**
     * Release a read latch.
     */
    void RUnlock() {
      uniq_lock latch(mutex_);
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


  /*
   * The good thing about a vector is we can maintain the forest in place and don't need to change prediction
   * function much
   */
  template<typename T>
  class ConcurrentVector {
  public:
    ConcurrentVector() : processed_(0) {};

    void push_and_notify(T *data) {
      uniq_lock latch(latch_);
      list_.push_back(data);
      if (list_.size() == processed_ + 1) {
        latch.unlock();
        cv_.notify_one();
      }
    }

    T* wait_and_consume() {
      uniq_lock latch(latch_);
      while (list_.size() == processed_) {
        cv_.wait(latch);
      }
      return list_[processed_++];
    }

    int get_processed() { return processed_; };

    size_t get_total_num() { return list_.size(); };

    T* get_elem(int idx) {
      return list_[idx];
    }

    void destroy_all() {
      for (T* t: list_) {
        delete t;
      }
    }

  private:
    std::vector<T*> list_;
    mutex_t latch_;
    cond_t cv_;
    std::atomic<int> processed_;
  };


}  // namespace bustub


#endif //GBDT_CONCURRENCY_H
