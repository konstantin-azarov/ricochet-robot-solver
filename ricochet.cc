#include <cmath>
#include <opencv2/opencv.hpp>

struct Line {
  float angle;
  int x;
};

float rad(float deg) {
  return deg*M_PI/180.0;
}

Line sweepLine(
    const cv::Mat_<cv::Vec3b>& img,
    int min_x, int max_x,
    float min_angle, float max_angle, float angle_step,
    int bucket_width, 
    int threshold) {
  int width = img.cols;
  int height = img.rows;

  int offset[height];

  float best_angle = -1;
  int best_x = 0;
  int best_score = 0;

  for (float angle = min_angle; angle <= max_angle; angle += angle_step) {
    for (int y = 0; y < height; ++y) {
      offset[y] = round(-tanf(angle) * y);
    }

    for (int x = min_x; x <= max_x; ++x) {
      int score = 0;
      if (x + offset[0] - bucket_width + 1 >= 0 && 
          x + offset[0] + bucket_width < width &&
          x + offset[height-1] - bucket_width + 1 >= 0 && 
          x + offset[height-1] + bucket_width < width) {
        for (int y = 0; y < height; ++y) {
          int x0 = x + offset[y];
          for (int c = 0; c < 3; ++c) {
            int left_sum = 0;
            for (int xx = x0 - bucket_width + 1; xx <= x0; ++ xx) {
              left_sum += img[y][xx][c];
            }

            int right_sum = 0;
            for (int xx = x0 + 1; xx <= x0 + bucket_width; ++ xx) {
              right_sum += img[y][xx][c];
            }

            if (abs(((float)(right_sum - left_sum) / bucket_width)) > threshold) {
              score++;
            }
          }
        }
        if (score > best_score) {
          best_score = score;
          best_x = x;
          best_angle = angle;
        }
      }
    }
  }

  return Line { best_angle, best_x };
}

const int kWidth = 1664;
const int kHeight = 936;

void drawLine(cv::Mat& img, const Line& l) {
  cv::line(
      img,
      cv::Point2i(l.x, 0),
      cv::Point2i(l.x - tan(l.angle) * kHeight, kHeight),
      cv::Scalar(0, 0, 255));
}

void drawLineH(cv::Mat& img, const Line& l) {
  cv::line(
      img,
      cv::Point2i(0, l.x),
      cv::Point2i(kWidth, l.x - tan(l.angle) * kWidth),
      cv::Scalar(0, 0, 255));
}

Line findLine(const cv::Mat_<cv::Vec3d>& img,
              int min_x, int max_x, int refine_width,
              float min_angle, float max_angle, float angle_step, 
              float refine_step,               
              int bucket_width,
              int threshold) {
  auto l = sweepLine(
      img,
      min_x, max_x,
      min_angle, max_angle, angle_step,
      bucket_width,
      threshold);

  return sweepLine(
      img,
      l.x - refine_width, l.x + refine_width,
      l.angle - angle_step, l.angle + angle_step, refine_step,
      bucket_width,
      threshold);
}

std::vector<cv::Vec2i> findRectangle(cv::Mat_<cv::Vec3b>& img) {
  auto l = findLine(
      img, 
      0, kWidth / 2, 20,
      rad(0), rad(20), rad(2), rad(0.05),
      5,
      60);

  auto r = findLine(
      img, 
      kWidth / 2, kWidth, 20,
      rad(-10), rad(0), rad(2), rad(0.05),
      5,
      60);

  cv::Mat_<cv::Vec3b> img_t;
  cv::transpose(img, img_t);


  auto t = findLine(
      img_t,
      0, kHeight/2, 20,
      rad(-10), rad(10), rad(2), rad(0.05),
      5,
      60);

  auto b = findLine(
      img_t,
      kHeight/2, kHeight, 20,
      rad(-10), rad(10), rad(2), rad(0.05),
      5,
      60);


  drawLine(img, l);
  drawLine(img, r);
  drawLineH(img, t);
  drawLineH(img, b);
  
  cv::imshow("debug", img);
  cv::waitKey(-1);

  return std::vector<cv::Vec2i>{};
}

int main(int argc, char** argv) {
  auto img_raw = cv::imread(argv[1]);

  cv::Mat_<cv::Vec3b> img_bgr, img_hsv;
  cv::resize(img_raw, img_bgr, cv::Size(kWidth, kHeight)); 
  cv::cvtColor(img_bgr, img_hsv, cv::COLOR_BGR2HSV);

  findRectangle(img_bgr);


  return 0;
}
