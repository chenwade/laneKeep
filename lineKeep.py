import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os
import imageio
import skimage

class Line_Keep_Image(object):
    def __init__(self, frame):
        '''
        :param filename: the name of input image
        '''
        self.image = frame
        self.image_height = self.image.shape[0]
        self.image_weight = self.image.shape[1]
        self.image_channel = self.image.shape[2]
        self.result = np.zeros_like(self.image)

    def process_image(self, gaussian_param, canny_param, vertices, hough_param):

        '''
        :param gussian_param:
        :param canny_param:
        :param vertices:
        :param hough_param:
        :return:
        '''

        '''
        gray
        '''
        gray = self.grayscale(self.image)

        '''
        gaussian blur
        '''
        kernel_size = gaussian_param[0]
        blur_gray = self.gaussian_blur(gray, kernel_size)

        '''
        canny process
        '''
        low_threshold = canny_param[0]
        high_threshold = canny_param[1]
        edges = self.canny(blur_gray, low_threshold, high_threshold)

        '''
        ROI (region of interest )
        '''
        masked_edges = self.region_of_interest(edges, vertices)

        '''
        Hough line transform
        '''
        rho = hough_param[0]
        theta = hough_param[1]
        threshold = hough_param[2]
        min_line_len = hough_param[3]
        max_line_gap = hough_param[4]
        lines = self.hough_lines(masked_edges, rho, theta, threshold, min_line_len, max_line_gap)

        '''
        weighted sumprocess_image
        '''

        self.result = self.weighted_img(self.image, lines, 0.8, 1)
        return self.result

    def grayscale(self, img):
        """Applies the Grayscale transform
        This will return an image with only one color channel
        but NOTE: to see the returned image as grayscale
        (assuming your grayscaled image is called 'gray')
        you should call plt.imshow(gray, cmap='gray')"""
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Or use BGR2GRAY if you read an image with cv2.imread()
        # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def canny(self, img, low_threshold, high_threshold):
        """Applies the Canny transform"""
        return cv2.Canny(img, low_threshold, high_threshold)

    def gaussian_blur(self, img, kernel_size):
        """Applies a Gaussian Noise kernel"""
        return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

    def region_of_interest(self, img, vertices):
        """
        Applies an image mask.

        Only keeps the region of the image defined by the polygon
        formed from `vertices`. The rest of the image is set to black.
        """
        # defining a blank mask to start with
        mask = np.zeros_like(img)

        # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
        if len(img.shape) > 2:
            channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255

        # filling pixels inside the polygon defined by "vertices" with the fill color
        cv2.fillPoly(mask, vertices, ignore_mask_color)

        # returning the image only where mask pixels are nonzero
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image

    def draw_lines(self, img, lanes, color=[255, 0, 0], thickness=5):
        """
        NOTE: this is the function you might want to use as a starting point once you want to
        average/extrapolate the line segments you detect to map out the full
        extent of the lane (going from the result shown in raw-lines-example.mp4
        to that shown in P1_example.mp4).

        Think about things like separating line segments by their
        slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
        line vs. the right line.  Then, you can average the position of each of
        the lines and extrapolate to the top and bottom of the lane.

        This function draws `lines` with `color` and `thickness`.
        Lines are drawn on the image inplace (mutates the image).
        If you want to make the lines semi-transparent, think about combining
        this function with the weighted_img() function below
        """
        left_lane = lanes[0]
        right_lane = lanes[1]
        cv2.line(img, left_lane[0], left_lane[1], color, thickness)
        cv2.line(img, right_lane[0], right_lane[1], color, thickness)



    def filter_bad_lines(self, lines, threshold):
        """
        ROI中图像经过霍夫线变换之后,会识别出很多直线,对于正确的车道线,其斜率应该近似.
        该函数会过滤掉一些干扰的直线
        """
        slope = [(y2 - y1) / (x2 - x1) for line in lines for x1, y1, x2, y2 in line]
        while len(lines) > 0:
            mean = np.mean(slope)
            diff = [abs(s - mean) for s in slope]
            idx = np.argmax(diff)
            if diff[idx] > threshold:
                slope.pop(idx)
                lines.pop(idx)
            else:
                break


    def calc_lane_vertice(self, points, ymin, ymax):
        x = [p[0] for p in points]
        y = [p[1] for p in points]
        fit = np.polyfit(y, x, 1)
        fit_fn = np.poly1d(fit)

        xmin = int(fit_fn(ymin))
        xmax = int(fit_fn(ymax))

        return [(xmin, ymin), (xmax, ymax)]


    def hough_lines_2_lane_lines(self, lines):
        '''
        这里主要想将霍尔线变化识别出的多条直线构造成两条车道线。
        改进方法可以考虑通过直线拟合polyfit()来构造直线
        :param lines: 图像做霍尔线变换后的线
        :return: 处理好后的两个车道
        '''
        '''
        seperate the lines into left group and right group,
        '''
        left_lines = []
        right_lines = []
        for line in lines:
            for x1, y1, x2, y2 in line:
                fit = np.polyfit((x1, x2), (y1, y2), 1)
                if fit[0] < -0.5:
                    left_lines.append(line)
                elif fit[0] > 0.5:
                    right_lines.append(line)

        if(len(left_lines) <= 0 or len(right_lines) <= 0):
            return None

        """ filtrate the bad lines """
        self.filter_bad_lines(left_lines, 0.1)
        self.filter_bad_lines(right_lines, 0.1)

        left_points = [(x1, y1) for line in left_lines for x1, y1, x2, y2 in line]
        left_points = left_points + [(x2, y2) for line in left_lines for x1, y1, x2, y2 in line]
        right_points = [(x1, y1) for line in right_lines for x1, y1, x2, y2 in line]
        right_points = right_points + [(x2, y2) for line in right_lines for x1, y1, x2, y2 in line]

        left_lane_vtx = self.calc_lane_vertice(left_points, 320, 540)
        right_lane_vtx = self.calc_lane_vertice(right_points, 320, 540)

        return [left_lane_vtx, right_lane_vtx]



    def hough_lines(self, img, rho, theta, threshold, min_line_len, max_line_gap):
        """
        `img` should be the output of a Canny transform.

        Returns an image with hough lines drawn.
        """
        lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                                maxLineGap=max_line_gap)

        line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        lanes = self.hough_lines_2_lane_lines(lines)
        if lanes:
            self.draw_lines(line_img, lanes)
        return line_img


    # Python 3 has support for cool math symbols.
    def weighted_img(self, img, initial_img, α=0.8, β=1., γ=0.):
        """
        `img` is the output of the hough_lines(), An image with lines drawn on it.
        Should be a blank image (all black) with lines drawn on it.

        `initial_img` should be the image before any processing.

        The result image is computed as follows:

        initial_img * α + img * β + γ
        NOTE: initial_img and img must be the same shape!
        """
        return cv2.addWeighted(initial_img, α, img, β, γ)


if __name__ == "__main__":

    '''
    parameter setting
    '''
    kernel_size = 5

    low_threshold = 50
    high_threshold = 150

    rho = 2
    theta = np.pi / 180
    threshold = 15
    min_line_len = 40
    max_line_gap = 20

    gaussian_param = [kernel_size]
    canny_param = [low_threshold, high_threshold]
    vertices = np.array([[(100, 540), (900, 540), (540, 320), (460, 320)]], dtype=np.int32)
    hough_param = [rho, theta, threshold, min_line_len, max_line_gap]

    '''change to dir and process the images in the dir'''
    """
    image_dir = '/home/wade/adas/CarND-LaneLines-P1/test_images'
    os.chdir(image_dir)
    filenames = os.listdir('.')
    for filename in filenames:
        frame = mpimg.imread(filename)
        image = Line_Keep_Image(frame)
        '''process the image and show the lane lines'''
        line_keep_image = image.process_image(gaussian_param, canny_param, vertices, hough_param)
              
        '''show the processed image'''
        plt.imshow(line_keep_image)
        plt.show()
    """
    #raw_image = imageio.get_reader('CarND-LaneLines-P1/test_videos/challenge.avi')
    raw_image = imageio.get_reader('CarND-LaneLines-P1/test_videos/solidYellowLeft.avi')
    #raw_image = imageio.get_reader('CarND-LaneLines-P1/test_videos/solidWhiteRight.avi')
    for i, img in enumerate(raw_image):
        frame = skimage.img_as_ubyte(img, True)

#        cv2.imshow('frame', rgb_frame)
        image = Line_Keep_Image(frame)
        line_keep_image = image.process_image(gaussian_param, canny_param, vertices, hough_param)
        b, g, r = cv2.split(line_keep_image)
        line_keep_image_rgb = cv2.merge([r, g, b])

        cv2.imshow('frame', line_keep_image_rgb)

        cv2.waitKey(0)





