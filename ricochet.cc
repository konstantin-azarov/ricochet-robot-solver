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
    const cv::Mat_<cv::Vec3i>& line_img,
    int min_x, int max_x,
    float min_angle, float max_angle, float angle_step,
    int bucket_width, 
    int threshold) {
  int width = line_img.cols - 1;
  int height = line_img.rows;

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
            int left_sum = 
              line_img[y][x0 + 1][c] - line_img[y][x0 - bucket_width + 1][c];
            int right_sum = 
              line_img[y][x0 + bucket_width + 1][c] - line_img[y][x0+1][c];

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

cv::Mat_<cv::Vec3i> lineSum(const cv::Mat_<cv::Vec3b>& img) {
  cv::Mat_<cv::Vec3i> res(img.rows, img.cols+1);
  for (int i=0; i < img.rows; ++i) {
    res[i][0] = cv::Vec3i(0, 0, 0);
    for (int j=0; j < img.cols; ++j) {
      res[i][j+1][0] = res[i][j][0] + img[i][j][0];
      res[i][j+1][1] = res[i][j][1] + img[i][j][1];
      res[i][j+1][2] = res[i][j][2] + img[i][j][2];
    }
  }

  return res;
}

cv::Point2i intersectLines(Line v, Line h) {
  double kv = -tan(v.angle);
  double kh = -tan(h.angle);

  double y = (h.x + kh * v.x) / (1 - kv*kh);
  double x = v.x + kv * y;

  return cv::Point2i(round(x), round(y));
}

std::vector<cv::Point2i> findRectangle(
    cv::Mat_<cv::Vec3b>& img,
    int width, int threshold) {
  auto line_sum = lineSum(img);

  auto l = findLine(
      line_sum, 
      0, kWidth / 2, 20,
      rad(0), rad(20), rad(2), rad(0.05),
      width, threshold);

  auto r = findLine(
      line_sum, 
      kWidth / 2, kWidth, 20,
      rad(-10), rad(0), rad(2), rad(0.05),
      width, threshold);

  cv::Mat_<cv::Vec3b> img_t;
  cv::transpose(img, img_t);

  auto line_sum_t = lineSum(img_t);

  auto t = findLine(
      line_sum_t,
      0, kHeight/2, 20,
      rad(-10), rad(10), rad(2), rad(0.05),
      width, threshold);

  auto b = findLine(
      line_sum_t,
      kHeight/2, kHeight, 20,
      rad(-10), rad(10), rad(2), rad(0.05),
      width, threshold);


  auto tl = intersectLines(l, t);
  auto tr = intersectLines(r, t);
  auto bl = intersectLines(l, b);
  auto br = intersectLines(r, b);

  return std::vector<cv::Point2i>{ tl, tr, br, bl };
}

cv::Mat_<double> computeRectification(
    const std::vector<cv::Point2i>& p,
    const std::vector<cv::Point2i>& t) {
  cv::Mat_<double> a(8, 8, 0.0);
  cv::Mat_<double> b(8, 1, 0.0);

  for (int i=0; i < p.size(); ++i) {
    a(2*i,     0) = p[i].x;
    a(2*i,     1) = p[i].y;
    a(2*i,     2) = 1;
    a(2*i,     6) = -t[i].x * p[i].x;
    a(2*i,     7) = -t[i].x * p[i].y;
    b(2*i) = t[i].x;

    a(2*i + 1, 3) = p[i].x;
    a(2*i + 1, 4) = p[i].y;
    a(2*i + 1, 5) = 1;
    a(2*i + 1, 6) = -t[i].y * p[i].x;
    a(2*i + 1, 7) = -t[i].y * p[i].y;
    b(2*i + 1) = t[i].y;
  }

  cv::Mat_<double> res;

  cv::vconcat(
      cv::Mat_<double>(a.inv() * b),
      cv::Mat_<double>(1, 1, 1.0),
      res);

  return res.reshape(1, 3);
}

cv::Mat_<cv::Vec2f> computeMap(cv::Mat_<double> m, int w, int h) {
  cv::Mat_<cv::Vec2f> res(h, w); 

  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      double u = x * m(0, 0) + y * m(0, 1) + m(0, 2);
      double v = x * m(1, 0) + y * m(1, 1) + m(1, 2);
      double w = x * m(2, 0) + y * m(2, 1) + m(2, 2);

      res(y, x) = cv::Vec2d(u / w, v / w);
    }
  }

  return res;
}

struct Cell {
  bool located = false;
  int cx = -1, cy = -1, w, h;

  bool right_wall = false, left_wall = false;
};

const int kMapCells = 16;

struct Cells {
  Cell c[kMapCells][kMapCells];
};

Cells extractCells(
    const cv::Mat_<uint8_t> img,
    int nominal_size, int cell_min_size, int cell_max_size, 
    float aspect_threshold, 
    float min_fill,
    float min_overlap) {
  cv::Mat_<int> labels, stats;
  cv::Mat centroids;

  cv::connectedComponentsWithStats(255-img, labels, stats, centroids);

  Cells cells;

  for (int i=0; i < stats.rows; ++i) {
    int x0 = stats(i, cv::CC_STAT_LEFT);
    int y0 = stats(i, cv::CC_STAT_TOP);
    int w = stats(i, cv::CC_STAT_WIDTH);
    int h = stats(i, cv::CC_STAT_HEIGHT);
    int cx = x0 + w/2;
    int cy = y0 + h/2;

    if (w >= cell_min_size && w <= cell_max_size &&
        h >= cell_min_size && h <= cell_max_size &&
        abs(1.0 - (float)w/h) < aspect_threshold) {

      int color = -1;
      int fill = 0;
      for (int y=y0; y < y0 + h && color <=0; ++y) {
        for (int x=x0; x < x0 + w && color <=0; ++x) {
          if (labels(y, x) == i) {
            color = img(y, x);
            fill++;
          }
        }
      }

      if (color == 0 && (float)fill / (w*h) > min_fill) {
        int ix = cx / nominal_size;
        int iy = cy / nominal_size;

        int nx0 = ix * nominal_size;
        int ny0 = iy * nominal_size;

        int tx0 = std::max(nx0, x0);
        int tx1 = std::min(nx0 + nominal_size, x0 + w);
        int ty0 = std::max(ny0, y0);
        int ty1 = std::min(ny0 + nominal_size, y0 + h);

        float overlap = 
          (float)(tx1 - tx0)*(ty1 - ty0) / (nominal_size * nominal_size);

        if (overlap > min_overlap) {
          auto& c = cells.c[iy][ix];
          c.located = true;
          c.cx = x0 + w/2;
          c.cy = y0 + h/2;
          c.w = w;
          c.h = h;
        }
      }
    }
  }


  // propagate
  int cx, cy, h, w;

  auto propagate_h = [&cy, &h](Cell& c) {
      if (c.located) {
        cy = c.cy;
        h = c.h;
      } else if (cy > -1) {
        c.cy = cy;
        c.h = h;
      }
  };

  auto propagate_v = [&cx, &w](Cell& c) {
      if (c.located) {
        cx = c.cx;
        w = c.w;
      } else if (cx > -1) {
        c.cx = cx;
        c.w = w;
      }
  };

  for (int i=0; i < kMapCells; ++i) {
    cy = -1; h = -1;
    for (int j=0; j < kMapCells; ++j) {
      propagate_h(cells.c[i][j]);
    }

    cy = -1; h = -1;
    for (int j = kMapCells-1; j >= 0; --j) {
      propagate_h(cells.c[i][j]);
    }
  }

  for (int j=0; j < kMapCells; ++j) {
    cx = -1; w = -1;
    for (int i=0; i < kMapCells; ++i) {
      propagate_v(cells.c[i][j]);
    }

    cx = -1; w = -1;
    for (int i = kMapCells-1; i >= 0; --i) {
      propagate_v(cells.c[i][j]);
    }
  }

  return cells;
}

struct ColorStat {
  int mean[3];
  int var[3];
};

void detectWalls(cv::Mat_<cv::Vec3b>& img_rgb, Cells& cells) {
  cv::Mat blurred; 
  cv::blur(img_rgb, blurred, cv::Size(5, 5));
 
  cv::Mat_<uint8_t> mask(img_rgb.rows, img_rgb.cols, (uint8_t)0);

  for (int i = 0; i < kMapCells; ++i) {
    for (int j = 0; j < kMapCells; ++j) {
      auto& c = cells.c[i][j];
      if (c.located) {
        int x0 = c.cx - c.w*0.3;
        int x1 = c.cx + c.w*0.3;
        int y0 = c.cy - c.h*0.3;
        int y1 = c.cy + c.h*0.3;

        mask(cv::Range(y0, y1), cv::Range(x0, x1)) = 255;
      }
    }
  }

  /* cv::imshow("mask", mask); */
  /* cv::waitKey(-1); */

  std::vector<double> baseline_mean, baseline_std_dev; 
  cv::meanStdDev(img_rgb, baseline_mean, baseline_std_dev, mask);

  std::vector<double> mean, std_dev;

  for (int i = 0; i < kMapCells - 1; ++i) {
    for (int j = 0; j < kMapCells - 1; ++j) {
      auto& c1 = cells.c[i][j];
      auto& c2 = cells.c[i][j+1];
      // right
      int x0 = (c1.cx + c2.cx)/2 - 5;
      int x1 = x0 + 5;
      int y0 = std::max(c1.cy - c1.h/2, c2.cy - c2.h/2);
      int y1 = std::min(c1.cy + c1.h/2, c2.cy + c2.h/2);

      cv::meanStdDev(
          img_rgb(cv::Range(y0, y1), cv::Range(x0, x1)),
          mean, std_dev);

      bool good = false;
      for (int c = 0; c < 3 && !good; c++) {
        double d = abs(mean[c] - baseline_mean[c]);
        if (d > std_dev[c]*2 && d > baseline_std_dev[c]*2) {
          good = true;
        }
      }

      c1.right_wall = good;
      c2.left_wall = good;
    }
  }
  /* std::cout << mean << std::endl; */
  /* std::cout << std_dev << std::endl; */
}

struct Robot {
  int cx, cy, r;
};

Robot findRobot(
    cv::Mat img_hsv, 
    uint8_t hue, uint8_t hue_delta, uint8_t min_sat, uint8_t min_value,
    int expected_size) {
  cv::Mat_<uint8_t> mask;

  std::vector<uint8_t> lower_bound = 
      {(uint8_t)(hue - hue_delta), min_sat, min_value};
  std::vector<uint8_t> upper_bound = 
      {(uint8_t)(hue + hue_delta), 255, 255};
  cv::inRange(img_hsv, lower_bound, upper_bound, mask);
  cv::dilate(
      mask, mask, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));
  cv::morphologyEx(
      mask, mask, cv::MORPH_CLOSE, 
      getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)),
      cv::Point(-1, -1),
      2);

  cv::Mat_<int> labels, stats;
  cv::Mat centroids;

  cv::connectedComponentsWithStats(mask, labels, stats, centroids);

  int best_size = 0;
  int best_i = -1;
  for (int i=1; i < stats.rows; ++i) {
    int a = stats(i, cv::CC_STAT_AREA);
    if (a > best_size) {
      best_size = a;
      best_i = i;
    }
  }

  Robot res { -1, -1, -1 };

  if (best_i != -1) {
    int x0 = stats(best_i, cv::CC_STAT_LEFT);
    int y0 = stats(best_i, cv::CC_STAT_TOP);
    int w = stats(best_i, cv::CC_STAT_WIDTH);
    int h = stats(best_i, cv::CC_STAT_HEIGHT);

    res.cx = x0 + w/2;
    res.cy = y0 + h/2;
    res.r = std::max(w/2, h/2);
  }
  
  /* cv::imshow("hsv", img_hsv); */
  /* cv::imshow("robot", mask); */
  /* cv::waitKey(-1); */

  return res;
}

void drawCells(cv::Mat& img, const Cells& cells) {
  for (int i = 0; i < kMapCells; ++i) {
    for (int j = 0; j < kMapCells; ++j) {
      auto& c = cells.c[i][j];

      if (c.cx >= 0 && c.cy >=0) {
        cv::rectangle(
            img,
            cv::Point2i(c.cx - c.w/2, c.cy - c.h/2),
            cv::Point2i(c.cx + c.w/2, c.cy + c.h/2),
            c.located ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255),
            1);

        if (c.right_wall) {
          cv::line(
              img,
              cv::Point2i(c.cx + c.w/2, c.cy - c.h/2),
              cv::Point2i(c.cx + c.w/2, c.cy + c.h/2),
              cv::Scalar(0, 0, 0),
              2);
        }
      }
    }
  }
}

void drawRobot(cv::Mat img, const Robot& robot, cv::Scalar color) {
  cv::circle(img, cv::Point2i(robot.cx, robot.cy), robot.r, color, 2);
}

const int kMapSize = 512;

int main(int argc, char** argv) {
  auto img_raw = cv::imread(argv[1]);

  cv::Mat_<cv::Vec3b> img_bgr, img_hsv;
  cv::resize(img_raw, img_bgr, cv::Size(kWidth, kHeight)); 
  cv::cvtColor(img_bgr, img_hsv, cv::COLOR_BGR2HSV);

  auto rect = findRectangle(img_bgr, 10, 60);

  cv::line(img_bgr, rect[0], rect[1], cv::Scalar(0, 255, 0));
  cv::line(img_bgr, rect[1], rect[2], cv::Scalar(0, 255, 0));
  cv::line(img_bgr, rect[2], rect[3], cv::Scalar(0, 255, 0));
  cv::line(img_bgr, rect[3], rect[0], cv::Scalar(0, 255, 0));
  
  /* cv::imshow("debug", img_bgr); */ 

  std::vector<cv::Point2i> target_rect {
    cv::Point2i(0, 0),
    cv::Point2i(kMapSize, 0),
    cv::Point2i(kMapSize, kMapSize),
    cv::Point2i(0, kMapSize)
  };

  auto transform = computeRectification(target_rect, rect);
  auto map = computeMap(transform, kMapSize, kMapSize);

  cv::Mat_<cv::Vec3b> img_r, img_r_bgr;
  
  cv::remap(img_hsv, img_r, map, cv::noArray(), cv::INTER_LINEAR);
  cv::remap(img_bgr, img_r_bgr, map, cv::noArray(), cv::INTER_LINEAR);

  /* cv::imshow("rectified", img_r); */

  cv::Mat_<uint8_t> img_g, edges;
  cv::extractChannel(img_r, img_g, 2);
  cv::blur(img_g, img_g, cv::Size(2, 2));

  int edge_threshold_low = 50;
  int edge_threshold_high = 3*edge_threshold_low;

  cv::Canny(img_g, edges, edge_threshold_low, edge_threshold_high);
  cv::dilate(
      edges, edges, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));
  
  float nominal_size = kMapSize / kMapCells;
  auto cells = extractCells(
      edges, 
      nominal_size, nominal_size*0.5, nominal_size*1.2, 
      0.2, 0.9, 0.5);

  // Walls
  detectWalls(img_r_bgr, cells);

  cv::Mat dbg_img;
  img_r_bgr.copyTo(dbg_img);
  //cv::cvtColor(edges, dbg_img, cv::COLOR_GRAY2BGR); 

  drawCells(dbg_img, cells);

  // Robots
  int min_robot_value = 90;
  int robot_size = kMapSize / kMapCells;

  auto yellow = findRobot(img_r, 20, 20, 150, min_robot_value, robot_size);
  auto green = findRobot(img_r, 60, 20, 150, min_robot_value, robot_size);
  auto red = findRobot(img_r, 180, 20, 150, min_robot_value, robot_size);
  auto blue = findRobot(img_r, 110, 20, 150, min_robot_value, robot_size);

  drawRobot(dbg_img, yellow, cv::Scalar(0, 255, 255));
  drawRobot(dbg_img, green, cv::Scalar(0, 255, 0));
  drawRobot(dbg_img, red, cv::Scalar(0, 0, 255));
  drawRobot(dbg_img, blue, cv::Scalar(255, 0, 0));

  

  cv::imshow("edges", dbg_img);
  cv::waitKey(-1);

  return 0;
}
