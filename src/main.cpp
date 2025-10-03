#include <opencv2/opencv.hpp>
#include <ceres/ceres.h>
#include <cmath>
#include <iostream>
#include <vector>
#include <string>
#include <cstddef>
#include <chrono>  // 添加时间测量头文件

// ------------------ 数据结构 ------------------
struct SamplePoint {
    double t;         // 秒
    cv::Point2d px;   // 物理坐标（像素)
};

// ------------------ 轨迹模型残差（Ceres） ------------------
struct TrajModelCost {
    TrajModelCost(double t_i, double x_i, double y_i, double x0_i, double y0_i)
        : t(t_i), xi(x_i), yi(y_i), x0(x0_i), y0(y0_i) {}

    template <typename T>
    bool operator()(const T* const params, T* residuals) const {
        const T vx0 = params[0];
        const T vy0 = params[1];
        const T g   = params[2];
        const T k   = params[3];

        const T expkt = ceres::exp(-k * T(t));
        const T x = T(x0) + (vx0 / k) * (T(1.0) - expkt);
        const T y = T(y0) + ((vy0 + g / k) / k) * (T(1.0) - expkt) - (g / k) * T(t);

        residuals[0] = x - T(xi);
        residuals[1] = y - T(yi);
        return true;
    }

    double t, xi, yi;
    double x0, y0;
};

// ------------------ 颜色检测（HSV） ------------------
cv::Point2d detectBlueProjectile(const cv::Mat& frame) {
    cv::Mat hsv;
    cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);

    // H:95-100, S:150-255, V:30-255
    const cv::Scalar lower_hsv(95, 150, 30);
    const cv::Scalar upper_hsv(100, 255, 255);

    cv::Mat mask;
    cv::inRange(hsv, lower_hsv, upper_hsv, mask);

    // 去噪：开操作
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2, 2));
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);

    // 找轮廓
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    if (contours.empty()) return {-1, -1};

    // 取最大有效轮廓
    double max_area = 0.0;
    int max_idx = -1;
    for (size_t i = 0; i < contours.size(); ++i) {
        double area = cv::contourArea(contours[i]);
        if (area > max_area && area > 3.0) {
            max_area = area;
            max_idx = static_cast<int>(i);
        }
    }
    if (max_idx == -1) return {-1, -1};

    // 计算质心
    cv::Moments m = cv::moments(contours[max_idx]);
    if (m.m00 == 0) return {-1, -1};

    double cx = m.m10 / m.m00;
    double cy = m.m01 / m.m00;
    return {cx, cy};
}

// ------------------ 主函数 ------------------
int main(int, char**) {
    // ====== 1) 记录程序开始时间 ======
    auto programStartTime = std::chrono::high_resolution_clock::now(); // 记录程序开始时间

    // ====== 1) 读取视频并检测点 ======
    cv::VideoCapture cap("/home/pl/1/video.mp4");
    if (!cap.isOpened()) {
        std::cerr << "无法打开视频：/home/pl/1/video.mp4\n";
        return 1;
    }

    const double fps = 60.0;          
    const int maxFrames = 100;        // 最多处理前 180 帧
    std::vector<SamplePoint> samples;

    cv::Mat frame, firstFrame;
    int frameIdx = 0;
    int frameH = 0, frameW = 0;

    while (cap.read(frame) && frameIdx < maxFrames) {
        if (frameIdx == 0) firstFrame = frame.clone();
        if (frameH == 0) {
            frameH = frame.rows;
            frameW = frame.cols;
        }

        cv::Point2d proj = detectBlueProjectile(frame);
        if (proj.x >= 0 && proj.y >= 0) {
            // OpenCV 像素坐标（原点左上，Y 向下）→ 物理坐标（原点左下，Y 向上）
            double t = frameIdx / fps;
            double x_phys = proj.x;
            double y_phys = frameH - proj.y;
            samples.push_back({t, {x_phys, y_phys}});
        }
        ++frameIdx;
    }
    cap.release();

    if (samples.empty()) {
        std::cerr << "未检测到任何轨迹点，结束。\n";
        return 0;
    }

    // ====== 2) Ceres 拟合 ======
    const double x0 = samples.front().px.x;
    const double y0 = samples.front().px.y;

    ceres::Problem problem;
    double params[4] = { 300.0, 200.0, 500.0, 0.1 };

    for (size_t i = 0; i < samples.size(); ++i) {
        const auto& s = samples[i];
        auto* cost = new ceres::AutoDiffCostFunction<TrajModelCost, 2, 4>(
            new TrajModelCost(s.t, s.px.x, s.px.y, x0, y0));
        problem.AddResidualBlock(cost, new ceres::HuberLoss(3.0), params);
    }

    problem.SetParameterLowerBound(params, 2, 100.0);   // g
    problem.SetParameterUpperBound(params, 2, 1000.0);
    problem.SetParameterLowerBound(params, 3, 0.01);    // k
    problem.SetParameterUpperBound(params, 3, 1.0);

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = false;  

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // 只输出 vx0, vy0, g, k
    std::cout << "vx0: " << params[0]
              << " vy0: " << params[1]
              << " g: "  << params[2]
              << " k: "  << params[3] << std::endl;

    // ====== 3) 计算程序总耗时 ======
    auto programEndTime = std::chrono::high_resolution_clock::now(); // 记录程序结束时间
    auto programTotalDuration = std::chrono::duration_cast<std::chrono::milliseconds>(programEndTime - programStartTime); // 计算总耗时
    std::cout << "程序总耗时: " << programTotalDuration.count() << " 毫秒" << std::endl;

    return 0;
}
