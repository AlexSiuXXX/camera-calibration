#include "camera_calib.h"

#include <ctime>

config::config(const std::string &config_path) {
    this->conig_path = config_path;
}

void config::read_config(const cv::FileNode &node) {

    node["Mode"] >> mode_var;
    node["Calibration_Pattern"] >> calib_pattern_var;

    node["ChessboardSize_Width" ] >> boardSize.width;
    node["ChessboardSize_Height"] >> boardSize.height;
    node["SquareSize"]  >> squareSize;

    std::cout << "[Mode]  " << mode_var << "\n";
    std::cout << "[Calibration_Pattern]  " << calib_pattern_var << "\n";
    std::cout << "[ChessboardSize_Width]  " << boardSize.width << "\n";
    std::cout << "[ChessboardSize_Height]  " << boardSize.height << "\n";
    std::cout << "[SquareSize]  " << squareSize << "\n";

}

void config::write_config(cv::FileStorage &fsw, monoChessboardCalibrationParams *monoCalibParams) const {

    time_t now = time(0);
    char *dt = ctime(&now);
   

    fsw << "Curent Time " << dt
        << "Camera matrix " << monoCalibParams->cameraMatrix
        << "distort coeff " << monoCalibParams->distCoeffs
        << "reprojetion error " << monoCalibParams->totErrs;

    fsw.release();

}

void config::read_img_list(const std::string &image_path) {
    cv::glob(image_path, imageList);
    nImages = imageList.size();

    for (int i = 0; i < nImages; i++) {
        std::cout << imageList[i] << "\n";
    }
}

bool monoCalib::collectImages() {

    bool done_collection = false;

    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cout << "Cannot open camera !!\n";
        return false;
    }

    cv::Mat frame;
    int cnt = 0;

    while (true) {

        bool ret = cap.read(frame);
        cnt++;

        if (!ret) {
            std::cout << "can't retrive frame\n";
            break;
        }

        std::string n = "/home/alexxxayjvm/Documents/rss_lib/images/intrChessboard/" + std::to_string(cnt) + ".jpg";

        cv::imshow("live", frame);

        if (cv::waitKey(1) == 'q') {
            return false;
            break;
        }

        if (cv::waitKey(1) == 's') {
            cv::imwrite(n, frame);
        }

    }

}

void monoCalib::show(cv::Mat &image, const std::string &win_name) {

    cv::imshow(win_name, image);
    cv::waitKey(1);

}

void monoCalib::computeChessboardCorners(config *cfg, std::vector<cv::Point3f> &objectPointsBuffer) {

    for (int i = 0; i < cfg->boardSize.height; i++) {
        for (int j = 0; j < cfg->boardSize.width; j++) {
            objectPointsBuffer.push_back(cv::Point3f(float(j*cfg->squareSize), float(i*cfg->squareSize), 0));
        }
    }
}

void monoCalib::chessboardDetection(cv::Mat &image, config *cfg, monoChessboardCalibrationParams *monoCalibParams) {

    cv::Mat img_gray;
    cv::cvtColor(image, img_gray, cv::COLOR_BGR2GRAY);

    std::vector<cv::Point2f> imagePointsBuffer;
    std::vector<cv::Point3f> objectPointsBuffer;

    bool foundCorners = cv::findChessboardCorners(image, 
                                                  cfg->boardSize, 
                                                  imagePointsBuffer, 
                                                  cv::CALIB_CB_ADAPTIVE_THRESH | 
                                                  cv::CALIB_CB_FILTER_QUADS |
                                                  cv::CALIB_CB_FAST_CHECK |
                                                  cv::CALIB_CB_NORMALIZE_IMAGE);

    if (foundCorners) {

        cv::cornerSubPix(img_gray,
                         imagePointsBuffer,
                         cv::Size(11, 11),
                         cv::Size(-1, -1),
                         cv::TermCriteria(cv::TermCriteria::EPS+cv::TermCriteria::MAX_ITER, 30, 0.1));
        
        monoCalibParams->imagePoints.push_back(imagePointsBuffer);
        computeChessboardCorners(cfg, objectPointsBuffer);

        monoCalibParams->objectPoints.push_back(objectPointsBuffer);

        cv::drawChessboardCorners(image, cfg->boardSize, cv::Mat(imagePointsBuffer), foundCorners);
        show(image, "image corner");

        // cv::destroyAllWindows();

    }

}


bool monoCalib::runCalibration(config *cfg, monoChessboardCalibrationParams *monoCalibParams) {

    monoCalibParams->cameraMatrix = cv::Mat::eye(3, 3, CV_64F);
    monoCalibParams->distCoeffs = cv::Mat::zeros(5, 1, CV_64F);

    int flags = cv::CALIB_FIX_ASPECT_RATIO+cv::CALIB_FIX_K3+cv::CALIB_ZERO_TANGENT_DIST+cv::CALIB_FIX_PRINCIPAL_POINT;

    double errr = cv::calibrateCamera(monoCalibParams->objectPoints, 
                                      monoCalibParams->imagePoints, 
                                      cfg->imageSize, 
                                      monoCalibParams->cameraMatrix,
                                      monoCalibParams->distCoeffs,
                                      monoCalibParams->rvecs,
                                      monoCalibParams->tvecs,
                                      flags);

    bool ok = cv::checkRange(monoCalibParams->cameraMatrix) && cv::checkRange(monoCalibParams->distCoeffs);

    std::cout << "\n\n[camera matrix] \n" << monoCalibParams->cameraMatrix << "\n";
    std::cout << "\n\n[dist coeff] \n" << monoCalibParams->distCoeffs << "\n\n";
   
    for (int i = 0; i < monoCalibParams->rvecs.size(); i++) {
        std::cout << " [rvec] \n" << monoCalibParams->rvecs[i] << "\n\n\n";
    }


    for (int i = 0; i < monoCalibParams->tvecs.size(); i++) {
        std::cout << "\n\n [tvec " << i << "]\n" << monoCalibParams->tvecs[i] << "\n\n\n";
    }

    double err = computeReprojErrs(monoCalibParams);
    std::cout << "[reproj err] " << err << "  " << errr << "\n";

    return ok;
}

double monoCalib::computeReprojErrs(monoChessboardCalibrationParams *monoCalibParams) {

    std::vector<cv::Point2f> imagePt2;

    int totPts = 0;

    double totErrs = 0;
    double curErr = 0;

    monoCalibParams->reprojErrs.resize(monoCalibParams->objectPoints.size());

    for (int i = 0; i < (int)monoCalibParams->objectPoints.size(); i++) {
        cv::projectPoints(cv::Mat(monoCalibParams->objectPoints[i]),
                          monoCalibParams->rvecs[i],
                          monoCalibParams->tvecs[i],
                          monoCalibParams->cameraMatrix,
                          monoCalibParams->distCoeffs,
                          imagePt2);

        curErr = cv::norm(cv::Mat(monoCalibParams->imagePoints[i]), cv::Mat(imagePt2), cv::NORM_L2);
        int n = (int)monoCalibParams->objectPoints[i].size();

        monoCalibParams->reprojErrs[i] = (float)sqrt(curErr*curErr/n);

        totErrs += curErr*curErr;
        totPts += n;

    }

    monoCalibParams->totErrs = sqrt(totErrs/totPts);

    return sqrt(totErrs/totPts);

}

