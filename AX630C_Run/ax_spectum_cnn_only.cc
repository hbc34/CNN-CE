#include <cstring>
#include <fstream>
#include <sstream>
#include <vector>
#include <numeric>
#include <iostream>

#include "base/common.hpp"
#include "middleware/io.hpp"

#include "utilities/args.hpp"
#include "utilities/cmdline.hpp"
#include "utilities/file.hpp"
#include "utilities/timer.hpp"

#include <ax_sys_api.h>
#include <ax_engine_api.h>

const int DEFAULT_LOOP_COUNT = 1;

const float PEAK_THRESHOLD = 0.23f; // Set peak threshold
const int MIN_DISTANCE = 1;         // Set minimum distance between peaks

// Function: Parse CSV file and load spectrum data
std::vector<float> load_spectrum_data(const std::string& file_path) {
    std::vector<float> intensity;
    std::ifstream file(file_path);
    std::string line;

    // Skip the header line of CSV file
    std::getline(file, line);

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string point_index, intensity_value;

        std::getline(ss, point_index, ',');
        std::getline(ss, intensity_value, ',');
        intensity.push_back(std::stof(intensity_value));
    }

    return intensity;
}

#include <algorithm> // for std::partial_sort

// Peak detection post-processing function using only 1D-CNN
std::vector<int> post_process_peaks_cnn_only(const std::vector<float>& predictions, int min_distance, int max_peaks = 10, float min_threshold = 0.03) {
    std::vector<int> peak_indices;

    // Step 1: Detect all peaks
    for (size_t i = 1; i < predictions.size() - 1; ++i) {
        if (predictions[i] > predictions[i - 1] && predictions[i] > predictions[i + 1]) {
            peak_indices.push_back(i);
        }
    }

    // Step 2: Filter peaks, merge peaks within close range
    std::vector<int> filtered_peaks;
    for (size_t i = 0; i < peak_indices.size(); ++i) {
        if (filtered_peaks.empty() || peak_indices[i] - filtered_peaks.back() > min_distance) {
            filtered_peaks.push_back(peak_indices[i]);
        } else {
            // Merge: select the peak with maximum value within the range
            if (predictions[peak_indices[i]] > predictions[filtered_peaks.back()]) {
                filtered_peaks.back() = peak_indices[i];
            }
        }
    }

    // Step 3: Filter peaks with prediction values greater than or equal to min_threshold
    std::vector<int> thresholded_peaks;
    for (int peak : filtered_peaks) {
        if (predictions[peak] >= min_threshold) {
            thresholded_peaks.push_back(peak);
        }
    }

    // Step 4: Select at most max_peaks peaks from those meeting the threshold
    std::partial_sort(thresholded_peaks.begin(),
                      thresholded_peaks.begin() + std::min(max_peaks, static_cast<int>(thresholded_peaks.size())),
                      thresholded_peaks.end(),
                      [&predictions](int a, int b) { return predictions[a] > predictions[b]; });

    thresholded_peaks.resize(std::min(max_peaks, static_cast<int>(thresholded_peaks.size())));

    return thresholded_peaks;
}

// Add wavelength mapping function
float map_index_to_wavelength(int index, int total_points = 2001, float start_wavelength = 1525.0f, float end_wavelength = 1565.0f) {
    return start_wavelength + (end_wavelength - start_wavelength) * index / (total_points - 1);
}

// Model inference and post-processing function (using only 1D-CNN)
bool run_model_cnn_only(const std::string& model, const std::vector<float>& data, const int& repeat)
{
    // 1. init engine
    AX_ENGINE_NPU_ATTR_T npu_attr;
    memset(&npu_attr, 0, sizeof(npu_attr));
    npu_attr.eHardMode = AX_ENGINE_VIRTUAL_NPU_DISABLE;
    auto ret = AX_ENGINE_Init(&npu_attr);

    if (0 != ret)
    {
        return ret;
    }

    // 2. load model
    std::vector<char> model_buffer;
    if (!utilities::read_file(model, model_buffer))
    {
        fprintf(stderr, "Read Run-Joint model(%s) file failed.\n", model.c_str());
        return false;
    }

    // 3. create handle
    AX_ENGINE_HANDLE handle;
    ret = AX_ENGINE_CreateHandle(&handle, model_buffer.data(), model_buffer.size());
    SAMPLE_AX_ENGINE_DEAL_HANDLE
    fprintf(stdout, "Engine creating handle is done.\n");

    // 4. create context
    ret = AX_ENGINE_CreateContext(handle);
    SAMPLE_AX_ENGINE_DEAL_HANDLE
    fprintf(stdout, "Engine creating context is done.\n");

    // 5. set io
    AX_ENGINE_IO_INFO_T* io_info;
    ret = AX_ENGINE_GetIOInfo(handle, &io_info);
    SAMPLE_AX_ENGINE_DEAL_HANDLE
    fprintf(stdout, "Engine get io info is done. \n");

    // 6. alloc io
    AX_ENGINE_IO_T io_data;
    ret = middleware::prepare_io(io_info, &io_data, std::make_pair(AX_ENGINE_ABST_DEFAULT, AX_ENGINE_ABST_CACHED));
    SAMPLE_AX_ENGINE_DEAL_HANDLE
    fprintf(stdout, "Engine alloc io is done. \n");

    // 7. insert input
    // Ensure input data shape is (1, 2001, 1)
    std::vector<uint8_t> input_data(8004,0);
    if (data.size() != 2001) {
        fprintf(stderr, "Input data size mismatch. Expected 2001, got %zu\n", data.size());
        return false;
    }
    std::memcpy(input_data.data(), data.data(), data.size() * sizeof(float));
    
    // Print parsed data
    std::cout << "Intensity Data: ";
    for (const auto& value : input_data) {
        fprintf(stdout, "0x%02X ",value);
    }
    std::cout << std::endl;
    
    // 7. insert input
    ret = middleware::push_input(input_data, &io_data, io_info);
    SAMPLE_AX_ENGINE_DEAL_HANDLE_IO

    fprintf(stdout, "Engine push input is done. \n");
    fprintf(stdout, "--------------------------------------\n");

    // 8. warm up
    for (int i = 0; i < 5; ++i)
    {
        AX_ENGINE_RunSync(handle, &io_data);
    }

    // 9. run model
    std::vector<float> time_costs(repeat, 0);
    for (int i = 0; i < repeat; ++i)
    {
        timer tick;
        ret = AX_ENGINE_RunSync(handle, &io_data);
        time_costs[i] = tick.cost();
        SAMPLE_AX_ENGINE_DEAL_HANDLE_IO
    }

    // 10. get result
    std::vector<float> predictions(io_data.pOutputs[0].nSize / sizeof(float));
    std::memcpy(predictions.data(), io_data.pOutputs[0].pVirAddr, io_data.pOutputs[0].nSize);
    
    // Output prediction values for 2001 points
    std::cout << "Predictions: ";
    for (const auto& value : predictions) {
        std::cout << value << " ";
    }
    std::cout << std::endl;
    
    // Use only 1D-CNN for peak detection
    auto peak_positions = post_process_peaks_cnn_only(predictions, 20, 16 , 0.03);
    std::cout << "CNN-Only Predicted Peaks: ";
    for (auto& peak : peak_positions) {
        std::cout << peak << " ";
    }
    std::cout << std::endl;
    
    // Output wavelengths corresponding to peaks
    std::cout << "CNN-Only Peak Wavelengths (nm): ";
    for (auto& peak : peak_positions) {
        float wavelength = map_index_to_wavelength(peak);
        fprintf(stdout, "%.4f ", wavelength);
    }
    std::cout << std::endl;
    
    fprintf(stdout, "--------------------------------------\n");
    auto total_time = std::accumulate(time_costs.begin(), time_costs.end(), 0.f);
    auto min_max_time = std::minmax_element(time_costs.begin(), time_costs.end());
    
    fprintf(stdout,
            "Repeat %d times, avg time %.2f ms, max_time %.2f ms, min_time %.2f ms\n",
            (int)time_costs.size(),
            total_time / (float)time_costs.size(),
            *min_max_time.second,
            *min_max_time.first);

    middleware::free_io(&io_data);
    return AX_ENGINE_DestroyHandle(handle);
}

std::vector<float> apply_gaussian_filter(const std::vector<float>& intensity, float sigma) {
    // Convert 1D vector to OpenCV Mat format (assumed to be 1xN)
    cv::Mat input = cv::Mat(intensity).reshape(1, 1);

    // Storage for Gaussian filter results
    cv::Mat filtered;
    int kernel_size = static_cast<int>(6 * sigma + 1) | 1; // Ensure kernel size is odd
    cv::GaussianBlur(input, filtered, cv::Size(kernel_size, 1), sigma);

    // Return result from Mat to std::vector
    std::vector<float> filtered_vector;
    filtered_vector.assign((float*)filtered.data, (float*)filtered.data + filtered.total());
    return filtered_vector;
}

int main(int argc, char* argv[])
{
    cmdline::parser cmd;
    cmd.add<std::string>("model", 'm', "joint file(a.k.a. joint model)", true, "");
    cmd.add<std::string>("spectrum", 's', "CSV file with spectrum data", true, "");
    cmd.add<int>("repeat", 'r', "repeat count", false, DEFAULT_LOOP_COUNT);
    cmd.parse_check(argc, argv);

    // 0. get app args
    auto model_file = cmd.get<std::string>("model");
    auto spectrum_file = cmd.get<std::string>("spectrum");

    auto model_file_flag = utilities::file_exist(model_file);
    auto spectrum_file_flag = utilities::file_exist(spectrum_file);

    if (!model_file_flag || !spectrum_file_flag)
    {
        auto show_error = [](const std::string& kind, const std::string& value) {
            fprintf(stderr, "Input file %s(%s) is not exist, please check it.\n", kind.c_str(), value.c_str());
        };

        if (!model_file_flag) { show_error("model", model_file); }
        if (!spectrum_file_flag) { show_error("spectrum", spectrum_file); }

        return -1;
    }

    auto repeat = cmd.get<int>("repeat");

    // 1. print args
    fprintf(stdout, "--------------------------------------\n");
    fprintf(stdout, "model file : %s\n", model_file.c_str());
    fprintf(stdout, "spectrum file : %s\n", spectrum_file.c_str());
    fprintf(stdout, "--------------------------------------\n");

    // 2. load spectrum data from CSV file
    auto spectrum_data = load_spectrum_data(spectrum_file);
    auto spectrum_data_filtered = apply_gaussian_filter(spectrum_data,10.0);
    
    // 3. sys_init
    AX_SYS_Init();

    // 4. -  engine model  -  can only use AX_ENGINE** inside
    {
        run_model_cnn_only(model_file, spectrum_data_filtered, repeat);

        // 4.3 engine de init
        AX_ENGINE_Deinit();
    }
    // 4. -  engine model  -

    AX_SYS_Deinit();
    return 0;
}