import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os
import imageio
import skimage
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML


'''
 parameter setting
'''

gussian_kernel_size = 5   # Gaussian blur kernel size

canny_low_threshold = 50  # Canny edge detection low threshold
canny_high_threshold = 150  # Canny edge detection high threshold

# Hough transform parameters
rho = 2
theta = np.pi / 180
threshold = 15
min_line_len = 40
max_line_gap = 20


gaussian_param = [gussian_kernel_size]
canny_param = [canny_low_threshold, canny_high_threshold]
hough_param = [rho, theta, threshold, min_line_len, max_line_gap]


def process_image(image):
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
    gray = grayscale(image)
    '''
    gaussian blur
    '''
    kernel_size = gaussian_param[0]
    blur_gray = gaussian_blur(gray, kernel_size)
    '''
    canny process
    '''
    low_threshold = canny_param[0]
    high_threshold = canny_param[1]
    edges = canny(blur_gray, low_threshold, high_threshold)

    '''
    ROI (region of interest )
    '''
    """estimate the region of interest of image based on the size of it"""
    image_height = image.shape[0]
    image_width = image.shape[1]
    bottom_left = (image_width * 0.1, image_height)
    bottom_right = (image_width * 0.9, image_height)
    top_left = (image_width * 0.46875, image_height * 0.6)
    top_right = (image_width * 0.5625, image_height * 0.6)
    vertices = np.array(
        [[bottom_left, bottom_right, top_right, top_left]],
        dtype=np.int32)
    masked_edges = region_of_interest(edges, vertices)

    '''
    Hough line transform
    '''
    rho = hough_param[0]
    theta = hough_param[1]
    threshold = hough_param[2]
    min_line_len = hough_param[3]
    max_line_gap = hough_param[4]
    h_lines = hough_lines(masked_edges, rho, theta, threshold, min_line_len, max_line_gap)

    '''
    Get the lane image from the hough lines
    '''
    lane_img = get_lane_img(image, h_lines)
    '''
    weighted sumprocess_image
    '''
    result = weighted_img(image, lane_img, 0.8, 1)
    return result


def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    #return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
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


def draw_lines(img, lanes, color=[255, 0, 0], thickness=8):
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


def filter_bad_lines(lines, threshold):
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


def calc_lane_vertice(points, ymin, ymax):
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    fit = np.polyfit(y, x, 1)
    fit_fn = np.poly1d(fit)
    xmin = int(fit_fn(ymin))
    xmax = int(fit_fn(ymax))
    return [(xmin, ymin), (xmax, ymax)]


def hough_lines_2_lane_lines(lines, bottom_height, top_height):
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
    filter_bad_lines(left_lines, 0.1)
    filter_bad_lines(right_lines, 0.1)
    left_points = [(x1, y1) for line in left_lines for x1, y1, x2, y2 in line]
    left_points = left_points + [(x2, y2) for line in left_lines for x1, y1, x2, y2 in line]
    right_points = [(x1, y1) for line in right_lines for x1, y1, x2, y2 in line]
    right_points = right_points + [(x2, y2) for line in right_lines for x1, y1, x2, y2 in line]
    left_lane_vtx = calc_lane_vertice(left_points, bottom_height, top_height)
    right_lane_vtx = calc_lane_vertice(right_points, bottom_height, top_height)
    return [left_lane_vtx, right_lane_vtx]


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    return lines


def get_lane_img(img, hough_lines):
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    img_height = img.shape[0]
    lanes = hough_lines_2_lane_lines(hough_lines, int(img_height), int(img_height * 0.6))
    if lanes:
        draw_lines(line_img, lanes)
    return line_img


def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    `initial_img` should be the image before any processing.
    The result image is computed as follows:
    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)


def image_test(input_dir, output_dir):

    if not os.path.exists(input_dir):
        print("input dir not found")
        exit(1)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    file_names = os.listdir(input_dir)
    for file_name in file_names:

        image = mpimg.imread(input_dir + file_name)
        processed_img = process_image(image)

        mpimg.imsave(output_dir + file_name, processed_img, format='jpg')

        """
        image = cv2.imread(input_dir + file_name)
        processed_img = process_image(image, gaussian_param, canny_param, hough_param)
        cv2.imwrite(output_dir + file_name, processed_img)
        """


def video_test(input_file, output_file):

    clip1 = VideoFileClip(input_file)
    white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
    white_clip.write_videofile(output_file, audio=False)



if __name__ == "__main__":

    image_test("CarND-LaneLines-P1/test_images/", 'CarND-LaneLines-P1/test_images_output/')

    video_test("CarND-LaneLines-P1/test_videos/solidWhiteRight.mp4",
               'CarND-LaneLines-P1/test_videos_output/solidWhiteRight.mp4')

    video_test("CarND-LaneLines-P1/test_videos/solidYellowLeft.mp4",
               'CarND-LaneLines-P1/test_videos_output/solidYellowLeft.mp4')

    video_test("CarND-LaneLines-P1/test_videos/challenge.mp4",
               'CarND-LaneLines-P1/test_videos_output/challenge.mp4')


    """
    #raw_image = imageio.get_reader('CarND-LaneLines-P1/test_videos/challenge.avi')
    raw_image = imageio.get_reader('CarND-LaneLines-P1/test_videos/solidYellowLeft.avi')
    #raw_image = imageio.get_reader('CarND-LaneLines-P1/test_videos/solidWhiteRight.avi')
 

    for i, img in enumerate(raw_image):
        frame = skimage.img_as_ubyte(img, True)
        line_keep_image = process_image(frame)


        # because the color of cv image follow as b,g, r, so we should translate the b,g,r to r,g,b
        b, g, r = cv2.split(line_keep_image)
        line_keep_image_rgb = cv2.merge([r, g, b])
        cv2.imshow('frame', line_keep_image_rgb)
        cv2.waitKey(0)
    """





