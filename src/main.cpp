#include "camera_calib.h"

int main() {

    std::string pth = "/home/alexxxayjvm/Documents/rss_lib/config/pinhole_chessboard_params.yml";
    std::string pth_out = "/home/alexxxayjvm/Documents/rss_lib/config/pinhole_chessboard_params_out.yml";
    std::string img_pth = "/home/alexxxayjvm/Documents/rss_lib/images/intrChessboard";

    monoCalib mc;

    config *cfg = new config(pth);
    monoChessboardCalibrationParams *monoP = new monoChessboardCalibrationParams;

    // if (mc.collectImages()) {
        
    // } else {
    //     std::cout << "failed to collect image\n";

    // }    

    std::cout << "collect image succ\n";
       
    cv::FileStorage fsr(pth, cv::FileStorage::READ);
    cv::FileStorage fsw(pth_out, cv::FileStorage::WRITE);
    
    cv::FileNode n = fsr["Setting"];
    cfg->read_config(n);
    std::cout << "\n\n";
    cfg->read_img_list(img_pth);


    for (int i  = 0; i < cfg->nImages; i++) {
        cv::Mat img = cv::imread(cfg->imageList[i]);

        cfg->imageSize = cv::Size(img.rows, img.cols);
        std::cout << img.rows << "  " << img.cols << "\n";
        mc.chessboardDetection(img, cfg, monoP);
    }

    if (mc.runCalibration(cfg, monoP)) {
        std::cout << "sd\n";
    } else {
        std::cout << "l\n";
    }

    cfg->write_config(fsw, monoP);


    delete cfg;
    delete monoP;
    
    return 0;

}
