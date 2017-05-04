from tkinter import Tk, LEFT, SUNKEN, X, Label
from tkinter.ttk import Frame, Button
from PIL import Image, ImageTk
import detector
import cv2

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
        self.flag = True
        self.add_button()

    def add_button(self):
        for name, path in self.image_names.items():
            img = cv2.imread(path)
            img = cv2.resize(img, (50, 50))
            cv2_im = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(cv2_im).copy()
            photo = ImageTk.PhotoImage(image)
            b = Button(image=photo, text=name, command=self.call)
            b.photo = photo
            print(name)
            b.bind("<ButtonPress-1>", self.call)
            b.pack(side=LEFT, expand=1)

    def callback(self):
        frame = detector.create(self.vs, self.pipline)
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
        print(self.global_name, name)
        if self.global_name != name:
            self.pipline = detector.replace_faces(name, self.fd)
        self.global_name = name
        if self.flag:
            self.flag = False
            self.callback()


def main():
    app = SampleApp()
    app.mainloop()

if __name__ == '__main__':
    main()
