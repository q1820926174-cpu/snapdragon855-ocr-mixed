// SPDX-License-Identifier: Apache-2.0
// Persistent binary-pipe inference worker. Stdout is reserved for JSON responses.
#define SCV41_GPU_LIBRARY
#include "gpu.cpp"
#include <unistd.h>

struct Request { uint64_t id; ncnn::Mat input; };

int main(int argc, char** argv) {
    if (argc == 2 && std::string(argv[1]) == "--version") {
        std::cout << "pnnx-gpu-worker 0.3.0 protocol=1 nhwc-f32=48x640x3" << std::endl;
        return 0;
    }
    try {
        const std::string base = env("SCV41_PIPE_BASE", "/models");
        const int workers = integer("OCR_GPU_WORKERS", 2);
        const int capacity = integer("OCR_MAX_INFLIGHT", 10);
        check(workers >= 1 && workers <= 4 && capacity >= workers && capacity <= 32, "worker limits");
        check(ncnn::get_gpu_count() == 1, "Exactly one Vulkan GPU is required; CPU fallback is disabled");
        const auto& gpu = ncnn::get_gpu_info(0);
        check(gpu.type() != 3, "A hardware GPU is required; software Vulkan devices are rejected");
        std::vector<std::string> keys;
        std::ifstream dictionary(base + "/keys.txt");
        std::string line;
        while (std::getline(dictionary, line)) keys.push_back(line);
        check(keys.size() == 18710 && keys[0].empty(), "CTC dictionary");
        std::vector<std::unique_ptr<ncnn::Net>> cnns, tails;
        std::vector<std::unique_ptr<FusedGpuCtc>> reducers;
        const double started = now();
        for (int i = 0; i < workers; ++i) {
            std::unique_ptr<ncnn::Net> tail(new ncnn::Net), cnn(new ncnn::Net);
            auto& opt = tail->opt;
            opt.num_threads = 1;
            opt.use_vulkan_compute = true;
            opt.use_fp16_packed = opt.use_fp16_storage = opt.use_fp16_arithmetic = opt.use_fp16_uniform = false;
            opt.use_subgroup_ops = opt.use_cooperative_matrix = false;
            tail->set_vulkan_device(0);
            cnn->opt = opt;
            cnn->opt.use_fp16_packed = cnn->opt.use_fp16_arithmetic = true;
            cnn->set_vulkan_device(0);
            check(!cnn->load_param((base + "/cnn/model.param").c_str()) && !cnn->load_model((base + "/cnn/model.bin").c_str()), "CNN load");
            check(!tail->load_param((base + "/tail/model.param").c_str()) && !tail->load_model((base + "/tail/model.bin").c_str()), "tail load");
            reducers.emplace_back(new FusedGpuCtc(opt));
            cnns.push_back(std::move(cnn)); tails.push_back(std::move(tail));
        }
        std::mutex mutex, output_mutex;
        std::condition_variable cv;
        std::deque<Request> queue;
        bool done = false;
        std::vector<std::thread> threads;
        for (int i = 0; i < workers; ++i) threads.emplace_back([&, i] {
            try {
                while (true) {
                    Request request;
                    {
                        std::unique_lock<std::mutex> lock(mutex);
                        cv.wait(lock, [&] { return done || !queue.empty(); });
                        if (queue.empty() && done) return;
                        request = std::move(queue.front()); queue.pop_front(); cv.notify_all();
                    }
                    const double start = now();
                    auto cnn = cnns[i]->create_extractor();
                    auto tail = tails[i]->create_extractor();
                    check(!cnn.input("t_0", request.input), "GPU input");
                    auto result = reducers[i]->execute(tail, keys, true, &cnn);
                    std::lock_guard<std::mutex> lock(output_mutex);
                    std::cout << std::setprecision(14) << "{\"type\":\"result\",\"id\":" << request.id
                        << ",\"text\":" << quote(result.text) << ",\"confidence\":" << result.confidence
                        << ",\"hash\":\"" << std::hex << result.hash << std::dec
                        << "\",\"gpu_ms\":" << ms(start) << ",\"worker\":" << i << "}" << std::endl;
                    check(bool(std::cout), "response pipe");
                }
            } catch (const std::exception& e) {
                std::cerr << "GPU_WORKER_FATAL " << e.what() << std::endl;
                std::_Exit(70);
            }
        });
        std::cout << "{\"type\":\"ready\",\"protocol\":1,\"backend\":\"vulkan\",\"model_initializations\":1,\"workers\":"
            << workers << ",\"device_name\":" << quote(gpu.device_name()) << ",\"device_type\":" << gpu.type() << ",\"init_ms\":" << ms(started)
            << ",\"feature_host_readbacks\":0,\"ctc_readback_bytes\":640}" << std::endl;
        while (true) {
            uint64_t id = 0;
            std::cin.read(reinterpret_cast<char*>(&id), sizeof(id));
            if (std::cin.eof() && std::cin.gcount() == 0) break;
            if (!std::cin || !id) std::_Exit(65);
            std::vector<float> raw(92160);
            std::cin.read(reinterpret_cast<char*>(raw.data()), raw.size() * sizeof(float));
            if (!std::cin) std::_Exit(65);
            Request request; request.id = id; request.input = ncnn::Mat(640, 48, 3);
            for (int c = 0; c < 3; ++c) {
                float* p = request.input.channel(c);
                for (int s = 0; s < 48 * 640; ++s) {
                    const float value = raw[s * 3 + c];
                    if (!std::isfinite(value)) std::_Exit(65);
                    p[s] = value;
                }
            }
            std::unique_lock<std::mutex> lock(mutex);
            cv.wait(lock, [&] { return queue.size() < size_t(capacity); });
            queue.push_back(std::move(request)); cv.notify_all();
        }
        { std::lock_guard<std::mutex> lock(mutex); done = true; cv.notify_all(); }
        for (auto& thread : threads) thread.join();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "GPU_WORKER_STARTUP_FAILED " << e.what() << std::endl;
        return 70;
    }
}
