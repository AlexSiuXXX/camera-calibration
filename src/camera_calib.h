#ifndef CAMERA_CALIB_H
#define CAMERA_CALIB_H

#include <iostream>
#include <vector>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>



struct monoChessboardCalibrationParams {

    cv::Mat cameraMatrix;
    cv::Mat distCoeffs;

    std::vector<cv::Mat> rvecs;
    std::vector<cv::Mat> tvecs;

    std::vector<std::vector<cv::Point2f>> imagePoints;
    std::vector<std::vector<cv::Point3f>> objectPoints;

    std::vector<float> reprojErrs;
    double totErrs = 0.0;

};

struct stereoCalibrationParams {
    // Extrinsic matrices (rotation, translation, essential, fundamental)
    cv::Mat R, t, E, F;
    // Rectification parameters
    cv::Mat R1, R2, P1, P2, Q;
    // Rectangle within the rectified image that contains all valid points
    cv::Rect valid_roi[2];
};

class config {

public:
    enum Pattern { CHESSBOARD, NOT_EXISTING };
    enum Mode { INTRINSIC, INVALID };

public:
    config(const std::string &config_path);
    ~config() {};

    void read_config(const cv::FileNode &node);
    void write_config(cv::FileStorage &fsw, monoChessboardCalibrationParams *monoCalibParams) const;
    void read_img_list(const std::string &image_path);

public:
    std::string conig_path;

    std::string mode_str;
    std::string calib_pattern_str;

    cv::Size boardSize;
    float squareSize;

    std::vector<std::string> imageList;   // Image list to run calibration

    int nImages;        
    cv::Size imageSize;     

private:
    std::string mode_var;
    std::string calib_pattern_var;

};

class monoCalib {

public:
    monoCalib() {};
    ~monoCalib() {};

    bool collectImages();

    void computeChessboardCorners(config *cfg, std::vector<cv::Point3f> &objectPointsBuffer);
    void chessboardDetection(cv::Mat &image, config *cfg, monoChessboardCalibrationParams *monoCalibParams);

    void show(cv::Mat &image, const std::string &win_name);

    bool runCalibration(config *cfg, monoChessboardCalibrationParams *monoCalibParams);
    double computeReprojErrs(monoChessboardCalibrationParams *monoCalibParams);

    
};

#endif