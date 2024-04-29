import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
import random


class Image:
    def __init__(self, image_path):
        self.image_path = image_path
        self.img = cv.imread(image_path)
        self.gray_img = cv.cvtColor(self.img, cv.COLOR_RGB2GRAY)
        self.rect_points = []
        self.rectangle_created = False

    def plot(self):
        # Plot the image on an xy plane
        plt.imshow(self.gray_img, cmap='gray')
        plt.axis('on')  # Show axis
        plt.xlabel('X')  # X-axis label
        plt.ylabel('Y')  # Y-axis label
        plt.title('Your Image Title')  # Title of the plot

        # Connect the onclick method to the 'button_press_event' event
        plt.gcf().canvas.mpl_connect('button_press_event', self.onclick)
        plt.gcf().canvas.mpl_connect('key_press_event', self.onkeypress)
        plt.show()

    def onclick(self, event):
        if self.rectangle_created:
            return  # Don't add more points if the rectangle is already created

        if event.xdata is not None and event.ydata is not None:
            x = event.xdata
            y = event.ydata
            self.rect_points.append((x, y))

            # Plot the selected point
            plt.plot(x, y, 'ro')

            # Draw rectangle if two points are selected
            if len(self.rect_points) == 2:
                x1, y1 = self.rect_points[0]
                x2, y2 = self.rect_points[1]
                xmin = min(x1, x2)
                xmax = max(x1, x2)
                ymin = min(y1, y2)
                ymax = max(y1, y2)
                plt.plot([xmin, xmax], [ymin, ymin], color='red')
                plt.plot([xmin, xmax], [ymax, ymax], color='red')
                plt.plot([xmin, xmin], [ymin, ymax], color='red')
                plt.plot([xmax, xmax], [ymin, ymax], color='red')
                self.rectangle_created = True

            # Refresh the plot
            plt.draw()

    def onkeypress(self, event):
        if event.key == ' ':  # Spacebar pressed
            if self.rectangle_created:
                min_x, min_y = self.rect_points[0]
                max_x, max_y = self.rect_points[1]
                for i in range(int(min_y), int(max_y) + 1):
                    for j in range(int(min_x), int(max_x) + 1):
                        self.gray_img[i, j] = random.randint(0, 255)
                plt.imshow(self.gray_img, cmap='gray')
                plt.draw()

