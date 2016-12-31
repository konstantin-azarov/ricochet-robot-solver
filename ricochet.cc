#include <cmath>
#include <opencv2/opencv.hpp>

const int kWidth = 1664;
const int kHeight = 936;

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
  int offset[kHeight];

  float best_angle = -1;
  int best_x = 0;
  int best_score = 0;

  for (float angle = min_angle; angle <= max_angle; angle += angle_step) {
    for (int y = 0; y < kHeight; ++y) {
      offset[y] = round(-tanf(angle) * y);
    }

    for (int x = min_x; x <= max_x; ++x) {
      int score = 0;
      for (int y = 0; y < kHeight; ++y) {
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

  return Line { best_angle, best_x };
}

void drawLine(cv::Mat& img, const Line& l) {
  cv::line(
      img,
      cv::Point2i(l.x, 0),
      cv::Point2i(l.x - tan(l.angle) * kHeight, kHeight),
      cv::Scalar(0, 0, 255));
}

int main(int argc, char** argv) {
  auto img_raw = cv::imread(argv[1]);

  cv::Mat_<cv::Vec3b> img_bgr, img_hsv;
  cv::resize(img_raw, img_bgr, cv::Size(kWidth, kHeight)); 
  cv::cvtColor(img_bgr, img_hsv, cv::COLOR_BGR2HSV);

  auto l = sweepLine(
      img_bgr, 
      100, kWidth / 2, 
      rad(0), rad(10), rad(0.1),
      5,
      60);

  auto r = sweepLine(
      img_bgr, 
      kWidth / 2, kWidth - 100,
      rad(-10), rad(0), rad(0.1),
      5,
      60);

  drawLine(img_bgr, l);
  drawLine(img_bgr, r);


  cv::imshow("debug", img_bgr);
  cv::waitKey(-1);

  return 0;
}
