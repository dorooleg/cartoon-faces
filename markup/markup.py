#!/usr/bin/env python3
import cv2
import copy
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import os


class Context:
    def __init__(self):
        Tk().withdraw()
        self.filename = askopenfilename()
        self.img = cv2.imread(self.filename, -1)
        self.new_img = copy.deepcopy(self.img)
        self.down = False
        self.points = []
        self.update_points = []


class Drawer:
    def draw_points(self):
        global context
        context.new_img = copy.deepcopy(context.img)
        cv2.imshow('image', context.new_img)
        for (i, (x, y)) in enumerate(context.points):
            cv2.circle(context.new_img, (x, y), 5, (255, 0, 0), -1)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(context.new_img, str(i), (x, y), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    def update_points_(self, x, y):
        global context
        for (ux, uy) in context.update_points:
            for i in range(len(context.points)):
                context.points[i] = (x, y) if context.points[i][0] == ux and context.points[i][1] == uy else \
                    context.points[i]
        context.update_points = [(x, y)]
        self.draw_points()

    def draw_circle(self, event, x, y, flags, param):
        global context
        if event == cv2.EVENT_LBUTTONDOWN:
            context.down = True
            context.update_points = [v for v in context.points if in_circle(x, v[0], y, v[1], 5)]
        elif event == cv2.EVENT_LBUTTONUP:
            context.update_points = []
            context.down = False
        elif event == cv2.EVENT_MOUSEMOVE:
            if context.down:
                self.update_points_(x, y)
        elif event == cv2.EVENT_LBUTTONDBLCLK:
            context.points.append((x, y))
            self.draw_points()
        elif event == cv2.EVENT_RBUTTONDBLCLK:
            context.points = list(filter(lambda v: out_circle(x, v[0], y, v[1], 5), context.points))
            self.draw_points()


def in_circle(x1, x2, y1, y2, r):
    return pow(x1 - x2, 2) + pow(y1 - y2, 2) <= pow(r, 2)

def out_circle(x1, x2, y1, y2, r):
    return not in_circle(x1, x2, y1, y2, r)

def save_markup(path):
    with open(path, 'w') as file:
        mylist_x = [str(x) for (x, _) in context.points]
        mylist_y = [str(x) for (_, x) in context.points]
        file.write(", ".join(mylist_x))
        file.write("\n")
        file.write(", ".join(mylist_y))


def get_csv_path(path):
    return os.path.splitext(path)[0] + '.csv'


context = Context()


def main_loop():
    while True:
        cv2.imshow('image', context.new_img)
        key = cv2.waitKey(20) & 0xFF
        if key == ord('q'):
            break
        if key == ord('a'):
            path = get_csv_path(context.filename)
            save_markup(path)


def main():
    drawer = Drawer()
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', drawer.draw_circle)
    main_loop()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()