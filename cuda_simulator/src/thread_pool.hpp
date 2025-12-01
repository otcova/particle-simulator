#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <mutex>
#include <queue>
#include <shared_mutex>
#include <thread>
#include <vector>

// Barrier implementation from:
// https://stackoverflow.com/questions/68181218/how-can-i-cleanly-mimic-c20-barrier-behavior-in-c14
struct Barrier {
    std::mutex m;
    std::condition_variable cv;
    std::size_t size;
    std::ptrdiff_t remaining;
    std::ptrdiff_t phase = 0;
    std::function<void()> completion;

    Barrier(std::size_t s, std::function<void()> f)
        : size(s), remaining(s), completion(std::move(f)) {}

    void arrive_and_wait() {
        std::unique_lock lk(m);
        --remaining;
        if (remaining != 0) {
            auto myphase = phase + 1;
            cv.wait(lk, [&] { return myphase - phase <= 0; });
        } else {
            completion();
            remaining = size;
            ++phase;
            cv.notify_all();
        }
    }
};

class ThreadPool {
   public:
    ThreadPool(size_t num_threads = std::thread::hardware_concurrency())
        : shutdown(false), finished_threads(0), barrier(num_threads, [this]() { on_task_done(); }) {
        workers.reserve(num_threads);
        for (size_t i = 0; i < num_threads; ++i) {
            workers.emplace_back([this, i]() { worker_thread(i); });
        }
    }

    ~ThreadPool() {
        {
            std::unique_lock lk(mtx);
            shutdown = true;
        }
        cv.notify_all();
        for (auto& t : workers) t.join();
    }

    void run(size_t n, std::function<void(size_t)> f) {
        size_t divisions = workers.size() * 8;
        size_t block_size = (n + divisions + 1) / divisions;
        if (block_size == 0) block_size = 1;

        run(n, block_size, f);
    }

    void run(size_t n, size_t block_size, std::function<void(size_t)> f) {
        {
            std::unique_lock lk(mtx);
            tasks.push({n, block_size, std::move(f)});
            if (tasks.size() == 1) {
                task_index.store(workers.size() * block_size, std::memory_order_relaxed);
                cv.notify_all();
            }
        }
    }

    void sync() {
        std::unique_lock lk(mtx);
        cv.wait(lk, [this] { return tasks.empty(); });
    }

   private:
    struct Task {
        size_t n;
        size_t block_size;
        std::function<void(size_t)> f;
    };

    std::vector<std::thread> workers;
    std::queue<Task> tasks;
    std::shared_mutex mtx;
    std::condition_variable_any cv;
    std::atomic<size_t> finished_threads;
    std::atomic<size_t> task_index;
    bool shutdown;
    Barrier barrier;

    void worker_thread(size_t tid) {
        Task task;
        while (true) {
            {
                std::shared_lock lk(mtx);
                cv.wait(lk, [this] { return !tasks.empty() || shutdown; });
                if (shutdown) break;
                task = tasks.front();
            }

            size_t start = tid * task.block_size;

            // Execute portion of the task
            while (start < task.n) {
                size_t end = std::min(start + task.block_size, task.n);

                for (size_t i = start; i < end; ++i) {
                    task.f(i);
                }

                // Take next block to execute
                start = task_index.fetch_add(task.block_size, std::memory_order_relaxed);
            }

            barrier.arrive_and_wait();
        }
    }

    void on_task_done() {
        bool is_empty;

        {
            std::unique_lock lk(mtx);
            tasks.pop();
            is_empty = tasks.empty();
            if (!is_empty) {
                Task task = tasks.front();
                task_index.store(workers.size() * task.block_size, std::memory_order_relaxed);
            }
        }

        if (is_empty) cv.notify_all();
    }
};
