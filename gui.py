#!/usr/bin/env python3
from tkinter import Tk, LEFT, SUNKEN, X, Label
from tkinter.ttk import Frame, Button
from PIL import Image, ImageTk
import detector
import cv2

import effects.mask as mask


class gui(Tk):
    def __init__(self, *args, **kwargs):
        Tk.__init__(self, *args, **kwargs)
        self.geometry("600x600")
        self.global_name = ""

        self.masks, self.image_names = mask.Loader().load()
        self.vs = detector.get_video_stream()
        self.pipeline = None

        separator = Frame(self, height=200, relief=SUNKEN)
        image = Image.open("./data/images/mermaid.png")
        photo = ImageTk.PhotoImage(image)
        self.label = Label(separator, image=photo)
        self.label.photo = photo
        self.label.pack()
        separator.pack(fill=X, padx=10)
        self.flag = True
        self.add_button()

    def set_mask_effect(self, mask_name):
        self.pipeline = detector.create_effect_pipeline(mask_name, self.masks)

    def add_button(self):
        for name, path in self.image_names.items():
            img = cv2.imread(path,  cv2.IMREAD_UNCHANGED)
            img = cv2.resize(img, (50, 50))
            cv2_im = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
            image = Image.fromarray(cv2_im).copy()
            photo = ImageTk.PhotoImage(image)
            b = Button(image=photo, text=name, command=self.call)
            b.photo = photo
            print("[INFO] mask found: {}".format(name))
            b.bind("<ButtonPress-1>", self.call)
            b.pack(side=LEFT, expand=1)

    def callback(self):
        if self.pipeline is None:
            return
        frame = detector.create(self.vs, self.pipeline)

        cv2_im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(cv2_im)
        img2 = ImageTk.PhotoImage(img)
        self.label.configure(image=img2)
        self.label.image = img2
        self.update()
        self.after(500, self.callback)

    def call(self, event=None):
        if event is None:
            if self.flag:
                self.callback()
            return
        if hasattr(event, 'widget'):
            name = event.widget.cget("text")
        else:
            name = event

        print("[LOG] replace mask from {} to {}".format(self.global_name, name))
        if self.global_name != name:
            self.set_mask_effect(name)
        self.global_name = name
        if self.flag:
            self.flag = False
            self.callback()


def main():
    app = gui()
    app.mainloop()


if __name__ == '__main__':
    main()
