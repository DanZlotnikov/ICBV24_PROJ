import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
import random


class Image:
    def __init__(self, image_path):
        self.image_path = image_path
        self.img = cv.imread(image_path)
        self.gray_img = cv.cvtColor(self.img, cv.COLOR_BGR2GRAY)
        self.clicked_points = []
        self.polygon_closed = False

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
        if self.polygon_closed:
            return  # Don't add more points if the polygon is closed

        if event.xdata is not None and event.ydata is not None:
            x = event.xdata
            y = event.ydata
            if event.button == 1:  # Left click
                print(f'You clicked at x = {x}, y = {y}')
                self.clicked_points.append((x, y))

                # Plot a line from the last selected point to the new one
                if len(self.clicked_points) > 1:
                    last_x, last_y = self.clicked_points[-2]
                    plt.plot([last_x, x], [last_y, y], color='red')

            elif event.button == 3:  # Right click
                # Close the polygon
                if len(self.clicked_points) > 2:
                    last_x, last_y = self.clicked_points[-1]
                    first_x, first_y = self.clicked_points[0]
                    plt.plot([last_x, first_x], [last_y, first_y], color='red')
                    self.polygon_closed = True

            # Refresh the plot
            plt.draw()

    def onkeypress(self, event):
        if event.key == ' ':  # Spacebar pressed
            if self.polygon_closed:
                polygon_points = np.array(self.clicked_points)
                min_x, min_y = polygon_points.min(axis=0)
                max_x, max_y = polygon_points.max(axis=0)
                for i in range(int(min_y), int(max_y) + 1):
                    for j in range(int(min_x), int(max_x) + 1):
                        if self.point_inside_polygon(j, i, polygon_points):
                            self.gray_img[i, j] = random.randint(0, 255)
                plt.imshow(self.gray_img, cmap='gray')
                plt.draw()

    def point_inside_polygon(self, x, y, poly):
        n = len(poly)
        inside = False
        p1x, p1y = poly[0]
        for i in range(1, n + 1):
            p2x, p2y = poly[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside
