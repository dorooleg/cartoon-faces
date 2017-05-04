from tkinter import Tk, LEFT, SUNKEN, X, Label, Scrollbar, Y, RIGHT, FALSE, HORIZONTAL, Toplevel
from tkinter.ttk import Frame, Button, Style
from PIL import Image, ImageTk
import detector
import cv2
import random


def init_camera(mask_name='mermaid'):
    return detector.init(mask_name)


class SampleApp(Tk):
    def __init__(self, *args, **kwargs):
        Tk.__init__(self, *args, **kwargs)
        self.geometry("600x600")
        self.global_name = 'mermaid'
        self.vs, self.pipeline, self.fd, self.image_names = init_camera(self.global_name)

        separator = Frame(self, height=200, relief=SUNKEN)
        image = Image.open("./data/images/mermaid.png")
        photo = ImageTk.PhotoImage(image)
        self.label = Label(separator, image=photo)
        self.label.photo = photo
        self.label.pack()
        separator.pack(fill=X, padx=10)
        self.add_button()

    def add_button(self):
        for name, path in self.image_names.items():
            img = cv2.imread(path)
            # print(img.shape)
            img = cv2.resize(img, (50, 50))
            cv2_im = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(cv2_im).copy()
            photo = ImageTk.PhotoImage(image)
            b = Button(self.call(name), image=photo, command=self.call(name))
            b.photo = photo
            b.pack(side=LEFT, expand=1)

    def call(self, name):
        print(self.global_name, name)
        if self.global_name != name:
            self.pipline = detector.replace_faces(name, self.fd)
        self.global_name = name
        self.callback()

    def callback(self):
        # global global_name
        # if global_name != name:
        #     self.pipline = detector.replace_faces(name, self.fd)
        # global_name = name
        frame = detector.create(self.vs, self.pipline)
        cv2_im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(cv2_im)
        img2 = ImageTk.PhotoImage(img)
        self.label.configure(image=img2)
        self.label.image = img2
        self.update()
        # self.after(1000, self.callback)


def main():
    # root = Tk()
    # root.geometry("600x600")

    app = SampleApp()
    app.mainloop()
    # separator = Frame(root, height=200, relief=SUNKEN)
    # image = Image.open("./data/images/mermaid.png")
    # photo = ImageTk.PhotoImage(image)
    # label = Label(separator, image=photo)
    # label.photo = photo
    # label.pack()
    # separator.pack(fill=X, padx=10)
    #
    # s = Style()
    # s.configure("Visible.TButton", foreground="red", background="pink")
    #
    # button = {}
    # for name, path in image_names.items():
    #     # path = image_names[name]
    #     print(name, path)
    #     img = cv2.imread(path)
    #     img = cv2.resize(img, (50, 50))
    #     cv2_im = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #     image = Image.fromarray(cv2_im).copy()
    #     photo = ImageTk.PhotoImage(image)
    #
    #     tkinter.Button(root, image=photo, cursor="dot",
    #                    command=lambda: callback(name, vs, pipeline, fd, label, root)).pack(side=LEFT, expand=1)
    #     root.update()
    # #
    # image1 = Image.open("../m_bg.png")
    # photo1 = ImageTk.PhotoImage(image1)
    # name_two = 'pakahontas'
    # b1 = Button(root, image=photo1, style="Visible.TButton", cursor="dot",
    #             command=lambda: callback(name_two, vs, pipeline, fd, label, root))
    # b1.pack(side=LEFT, expand=1)
    #
    # image2 = Image.open("../mermaid_1.jpg")
    # name_tree = 'shrek'
    # photo2 = ImageTk.PhotoImage(image2)
    # b2 = Button(root, image=photo2, style="Visible.TButton", cursor="dot",
    #             command=lambda: callback(name_tree, vs, pipeline, fd, label, root))
    # b2.pack(side=LEFT, expand=1)

    # scrollbar = Scrollbar(root, orient=HORIZONTAL)
    # scrollbar.pack(fill=X, side=LEFT, expand=FALSE)

    # root.mainloop()


if __name__ == '__main__':
    main()
