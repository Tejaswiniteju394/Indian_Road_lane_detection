# Indian Road Lane Detection (Image-Based)

This project uses OpenCV and NumPy to detect **left, right, and center** lanes from **images of Indian roads**. It works well on single-frame image uploads and draws colored lines:

- **Green** for left lane
- **Red** for right lane
- **Yellow** for center lane (midpoint between left and right)


## 🚀 Features

- Edge detection with Canny filter
- Region of Interest masking
- Hough Line Transform for lane detection
- Averaging multiple lines for smooth detection
- Lane overlay on original image

## 🛠 Requirements

```bash
pip install opencv-python numpy matplotlib
