/**
 * NOTE: Convert model with --fp16 may lead to incorrect results
 */
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <fstream>

#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>

#define CUDA_CHECK(call)                                                      \
    do                                                                        \
    {                                                                         \
        const cudaError_t error_code = call;                                  \
        if (error_code != cudaSuccess)                                        \
        {                                                                     \
            printf("CUDA_CHECK Error:\n");                                    \
            printf("    File:       %s\n", __FILE__);                         \
            printf("    Line:       %d\n", __LINE__);                         \
            printf("    Error code: %d\n", error_code);                       \
            printf("    Error text: %s\n", cudaGetErrorString(error_code));   \
            exit(1);                                                          \
        }                                                                     \
    } while (0)

struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};

const std::vector<std::string> labels = {
    "Person", "Bicycle", "Car", "Motorcycle", "Airplane", "Bus", "Train",
    "Truck", "Boat", "Traffic light", "Fire hydrant", "Stop sign", "Parking meter",
    "Bench", "Bird", "Cat", "Dog", "Horse", "Sheep", "Cow", "Elephant", "Bear",
    "Zebra", "Giraffe", "Backpack", "Umbrella", "Handbag", "Tie", "Suitcase",
    "Frisbee", "Skis", "Snowboard", "Sports ball", "Kite", "Baseball bat",
    "Baseball glove", "Skateboard", "Surfboard", "Tennis racket", "Bottle",
    "Wine glass", "Cup", "Fork", "Knife", "Spoon", "Bowl", "Banana", "Apple",
    "Sandwich", "Orange", "Broccoli", "Carrot", "Hot dog", "Pizza", "Donut",
    "Cake", "Chair", "Couch", "Potted plant", "Bed", "Dining table", "Toilet",
    "Tv", "Laptop", "Mouse", "Remote", "Keyboard", "Cell phone", "Microwave",
    "Oven", "Toaster", "Sink", "Refrigerator", "Book", "Clock", "Vase", "Scissors",
    "Teddy bear", "Hair drier", "Toothbrush"
};

class Logger : public nvinfer1::ILogger
{
public:
    void log(Severity severity, const char *msg) noexcept override
    {
        if (severity <= nvinfer1::ILogger::Severity::kWARNING)
        {
            std::cerr << "[TensorRT] ";
            switch (severity)
            {
                case nvinfer1::ILogger::Severity::kINTERNAL_ERROR:  std::cerr << "INTERNAL_ERROR: ";    break;
                case nvinfer1::ILogger::Severity::kERROR:           std::cerr << "ERROR: ";             break;
                case nvinfer1::ILogger::Severity::kWARNING:         std::cerr << "WARNING: ";           break;
                case nvinfer1::ILogger::Severity::kINFO:            std::cerr << "INFO: ";              break;
                case nvinfer1::ILogger::Severity::kVERBOSE:         std::cerr << "VERBOSE: ";           break;
            }
            std::cerr << msg << "\n";
        }
    }
};

static Logger logger;

bool DrawObjects(cv::Mat &image, const std::vector<Object> &objects,
    const std::vector<std::string> &labels, bool isSilent)
{
    for (auto obj : objects)
    {
        if (obj.label >= static_cast<int>(labels.size()))
            return false;

        if (isSilent != true)
            std::printf("%s = %.2f%% at (%.1f, %.1f)  %.1f x %.1f\n", labels[obj.label].c_str(), obj.prob * 100.0f,
                obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

        char text[256];
        snprintf(text, sizeof(text), "%s %.1f%%", labels[obj.label].c_str(), obj.prob * 100.0f);

        auto scalar = cv::Scalar(255, 255, 255);
        cv::rectangle(image, obj.rect, scalar, 2);

        int baseLine = 5;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.75, 1, &baseLine);

        int x = obj.rect.x - 1;
        int y = obj.rect.y - label_size.height - baseLine;
        y = std::max(0, y);
        if (x + label_size.width > image.cols)
            x = image.cols - label_size.width;

        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
            scalar, -1);

        cv::putText(image, text, cv::Point(x, y + label_size.height + baseLine / 2),
            cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 0, 0), 2);
    }

    return true;
}

size_t CountElement(const nvinfer1::Dims &dims)
{
    int64_t total = 1;
    for (int32_t i = 0; i < dims.nbDims; ++i)
        total *= dims.d[i];
    return static_cast<size_t>(total);
}

template <typename T>
T Clamp(T val, T min, T max)
{
    return val > min ? (val < max ? val : max) : min;
}

void GetLetterboxDimensions(
    const int img_rows, const int img_cols,
    const int target_size,
    int &resize_rows, int &resize_cols, int &pad_rows, int &pad_cols, float &scale
)
{
    scale = static_cast<float>(target_size) / std::max(img_rows, img_cols);
    resize_rows = static_cast<int>(std::round(img_rows * scale));
    resize_cols = static_cast<int>(std::round(img_cols * scale));
    pad_rows = target_size - resize_rows;
    pad_cols = target_size - resize_cols;
}

int main(int argc, char *argv[])
{
    // --- Settings ---
    if (argc < 3)
    {
        std::printf("Usage: %s model image [conf] [target size]\n", argv[0]);
        return 0;
    }
    const std::string model_path = std::string(argv[1]);
    float conf_thres = 0.25f;
    int target_size = 640;
    if (argc >= 4 && std::stof(argv[3]) > 0.0f)
        conf_thres = std::stof(argv[3]);
    if (argc >= 5 && std::stoi(argv[4]) > 0 &&
        std::stoi(argv[4]) % 32 == 0 && std::stoi(argv[4]) > 32)
        target_size = std::stoi(argv[4]);

    std::cout << "Model: " << model_path << "\n";
    std::cout << "Input: " << argv[2] << "\n";
    std::cout << "Conf: " << conf_thres << "\n";
    std::cout << "Target size: " << target_size << "\n";

    // --- Init TRT ---
    // load model data
    std::ifstream engine_file(model_path, std::ios::binary);
    if (!engine_file)
    {
        std::cerr << "Failed to open engine file\n";
        return -1;
    }
    engine_file.seekg(0, engine_file.end);
    std::streamsize engine_size = engine_file.tellg();
    engine_file.seekg(0, engine_file.beg);
    std::unique_ptr<char[]> engine_data{std::make_unique<char[]>(engine_size)};
    if (!engine_file.read(engine_data.get(), engine_size))
    {
        std::cerr << "Failed to read engine file\n";
        return -1;
    }
    engine_file.close();

    // create runtime, engine, context, and stream
    std::unique_ptr<nvinfer1::IRuntime> runtime{nvinfer1::createInferRuntime(logger)};
    if (!runtime)
    {
        std::cerr << "Failed to create runtime\n";
        return -1;
    }
    std::unique_ptr<nvinfer1::ICudaEngine> engine{runtime->deserializeCudaEngine(engine_data.get(), engine_size)};
    if (!engine)
    {
        std::cerr << "Failed to deserialize engine\n";
        return -1;
    }
    std::unique_ptr<nvinfer1::IExecutionContext> context{engine->createExecutionContext()};
    if (!context)
    {
        std::cerr << "Failed to create contexts\n";
        return -1;
    }
    std::unique_ptr<cudaStream_t> stream = std::make_unique<cudaStream_t>();
    CUDA_CHECK(cudaStreamCreate(stream.get()));

    // get model info
    std::vector<std::pair<int, std::string>> in_tensor_info, out_tensor_info;
    for (int i = 0; i < engine->getNbIOTensors(); ++i)
    {
        const char *tensor_name = engine->getIOTensorName(i);
        nvinfer1::TensorIOMode io_mode = engine->getTensorIOMode(tensor_name);
        if (io_mode == nvinfer1::TensorIOMode::kINPUT)
            in_tensor_info.push_back({i, std::string(tensor_name)});
        else if (io_mode == nvinfer1::TensorIOMode::kOUTPUT)
            out_tensor_info.push_back({i, std::string(tensor_name)});
    }

    // create pinned host memory
    nvinfer1::Dims in0_dims = context->getTensorShape(in_tensor_info[0].second.c_str());
    nvinfer1::Dims out2_dims = context->getTensorShape(out_tensor_info[2].second.c_str());
    size_t max_in0_size_byte = CountElement(in0_dims) * sizeof(float);
    size_t max_in1_size_byte = CountElement(context->getTensorShape(in_tensor_info[1].second.c_str())) * sizeof(int64_t);
    size_t max_out0_size_byte = CountElement(context->getTensorShape(out_tensor_info[0].second.c_str())) * sizeof(int64_t);
    size_t max_out1_size_byte = CountElement(context->getTensorShape(out_tensor_info[1].second.c_str())) * sizeof(float);
    size_t max_out2_size_byte = CountElement(out2_dims) * sizeof(float);
    std::vector<void *> host_ins, host_outs;
    host_ins.resize(in_tensor_info.size());
    host_outs.resize(out_tensor_info.size());
    CUDA_CHECK(cudaMallocHost(&host_ins[0], max_in0_size_byte));
    CUDA_CHECK(cudaMallocHost(&host_ins[1], max_in1_size_byte));
    CUDA_CHECK(cudaMallocHost(&host_outs[0], max_out0_size_byte));
    CUDA_CHECK(cudaMallocHost(&host_outs[1], max_out1_size_byte));
    CUDA_CHECK(cudaMallocHost(&host_outs[2], max_out2_size_byte));
    // create cuda memory
    std::vector<void *> buffers{};
    buffers.resize(engine->getNbIOTensors());
    CUDA_CHECK(cudaMalloc(&buffers[in_tensor_info[0].first], max_in0_size_byte));
    CUDA_CHECK(cudaMalloc(&buffers[in_tensor_info[1].first], max_in1_size_byte));
    CUDA_CHECK(cudaMalloc(&buffers[out_tensor_info[0].first], max_out0_size_byte));
    CUDA_CHECK(cudaMalloc(&buffers[out_tensor_info[1].first], max_out1_size_byte));
    CUDA_CHECK(cudaMalloc(&buffers[out_tensor_info[2].first], max_out2_size_byte));

    // set in/out tensor address
    context->setInputTensorAddress(in_tensor_info[0].second.c_str(), buffers[in_tensor_info[0].first]);
    context->setInputTensorAddress(in_tensor_info[1].second.c_str(), buffers[in_tensor_info[1].first]);
    context->setOutputTensorAddress(out_tensor_info[0].second.c_str(), buffers[out_tensor_info[0].first]);
    context->setOutputTensorAddress(out_tensor_info[1].second.c_str(), buffers[out_tensor_info[1].first]);
    context->setOutputTensorAddress(out_tensor_info[2].second.c_str(), buffers[out_tensor_info[2].first]);

    // --- Detect ---
    cv::Mat image = cv::imread(argv[2]);
    if (image.empty())
    {
        std::cout << "Failed to read image\n";
        return -1;
    }

    // preprocessing
    int img_rows = image.rows;
    int img_cols = image.cols;
    float scale;
    int resize_rows, resize_cols, pad_rows, pad_cols;
    GetLetterboxDimensions(
        img_rows, img_cols, target_size,
        resize_rows, resize_cols, pad_rows, pad_cols, scale
    );
    cv::Mat letterbox;
    cv::resize(image, letterbox, cv::Size(resize_cols, resize_rows), 0, 0, cv::INTER_AREA);
    cv::copyMakeBorder(
        letterbox, letterbox,
        pad_rows / 2, pad_rows - pad_rows / 2,
        pad_cols / 2, pad_cols - pad_cols / 2,
        cv::BORDER_CONSTANT, cv::Scalar(114.0, 114.0, 114.0)
    );

    const int blob_dims[4] = {1, 3, static_cast<int>(in0_dims.d[2]), static_cast<int>(in0_dims.d[3])};
    cv::Mat blob(4, blob_dims, CV_32F, host_ins[0]);
    // no normalization
    cv::dnn::blobFromImage(letterbox, blob, 1.0f / 255.0f, cv::Size(letterbox.cols, letterbox.rows), cv::Scalar(0, 0, 0), true, false, CV_32F);

    nvinfer1::Dims trt_in0_dims{}, trt_in1_dims{};
    trt_in0_dims.nbDims = 4;
    trt_in0_dims.d[0] = 1;
    trt_in0_dims.d[1] = 3;
    trt_in0_dims.d[2] = letterbox.rows;
    trt_in0_dims.d[3] = letterbox.cols;
    context->setInputShape(in_tensor_info[0].second.c_str(), trt_in0_dims);

    int64_t *orig_size = static_cast<int64_t *>(host_ins[1]);
    orig_size[0] = letterbox.cols;
    orig_size[1] = letterbox.rows;
    trt_in1_dims.nbDims = 2;
    trt_in1_dims.d[0] = 1;
    trt_in1_dims.d[1] = 2;
    context->setInputShape(in_tensor_info[1].second.c_str(), trt_in1_dims);

    // execute
    CUDA_CHECK(cudaMemcpyAsync(buffers[0], blob.data, max_in0_size_byte, cudaMemcpyHostToDevice, *stream));
    CUDA_CHECK(cudaMemcpyAsync(buffers[1], orig_size, max_in1_size_byte, cudaMemcpyHostToDevice, *stream));

    context->enqueueV3(*stream);

    CUDA_CHECK(cudaMemcpyAsync(host_outs[0], buffers[2], max_out0_size_byte, cudaMemcpyDeviceToHost, *stream));
    CUDA_CHECK(cudaMemcpyAsync(host_outs[1], buffers[3], max_out1_size_byte, cudaMemcpyDeviceToHost, *stream));
    CUDA_CHECK(cudaMemcpyAsync(host_outs[2], buffers[4], max_out2_size_byte, cudaMemcpyDeviceToHost, *stream));
    CUDA_CHECK(cudaStreamSynchronize(*stream));

    const int64_t *labels_ptr = reinterpret_cast<const int64_t *>(host_outs[0]);
    const float *boxes_ptr = reinterpret_cast<const float *>(host_outs[1]);
    const float *scores_ptr = reinterpret_cast<const float *>(host_outs[2]);

    size_t num_box = static_cast<size_t>(out2_dims.d[1]);
    size_t walk = 4;
    float dw = pad_cols / 2, dh = pad_rows / 2;
    std::vector<Object> objects;
    for (size_t i = 0; i < num_box; ++i)
    {
        if (scores_ptr[i] < conf_thres)
            continue;

        float x0 = boxes_ptr[i * walk];
        float y0 = boxes_ptr[i * walk + 1];
        float x1 = boxes_ptr[i * walk + 2];
        float y1 = boxes_ptr[i * walk + 3];

        x0 = (x0 - dw) / scale;
        y0 = (y0 - dh) / scale;
        x1 = (x1 - dw) / scale;
        y1 = (y1 - dh) / scale;

        x0 = Clamp(x0, 0.0f, static_cast<float>(img_cols));
        y0 = Clamp(y0, 0.0f, static_cast<float>(img_rows));
        x1 = Clamp(x1, x0, static_cast<float>(img_cols));
        y1 = Clamp(y1, y0, static_cast<float>(img_rows));

        Object object;
        object.rect.x = x0;
        object.rect.y = y0;
        object.rect.width = x1 - x0;
        object.rect.height = y1 - y0;
        object.prob = scores_ptr[i];
        object.label = static_cast<int>(labels_ptr[i]);
        objects.emplace_back(object);
    }

    // save results
    if (DrawObjects(image, objects, labels, false))
        cv::imwrite("./result.jpg", image);
    else
        std::cout << "Failed to draw objects\n";

    // --- Release resources ---
    for (const auto &buffer : buffers)
        if (buffer)
            CUDA_CHECK(cudaFree(buffer));
    for (const auto &host_in : host_ins)
        if (host_in)
            CUDA_CHECK(cudaFreeHost(host_in));
    for (const auto &host_out : host_outs)
        if (host_out)
            CUDA_CHECK(cudaFreeHost(host_out));
    if (stream && *stream)
        CUDA_CHECK(cudaStreamDestroy(*stream));

    return 0;
}
