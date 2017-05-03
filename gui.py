from tkinter import Tk, LEFT, SUNKEN, X, Label, Scrollbar, Y, RIGHT, FALSE, HORIZONTAL
from tkinter.ttk import Frame, Button, Style
from PIL import Image, ImageTk
import detector
import cv2


def init():
    return detector.init()


def callback(vs, pipline, label, root):
    frame = detector.create(vs, pipline)
    cv2_im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(cv2_im)
    img2 = ImageTk.PhotoImage(img)
    label.configure(image=img2)
    label.image = img2
    root.update()
    root.after(1000, callback(vs, pipline, label, root))


def main():
    vs, pipeline = init()

    root = Tk()
    root.geometry("600x600")

    import os
    print(os.getcwd())
    separator = Frame(root, height=200, relief=SUNKEN)
    image = Image.open("./data/images/mermaid.png")
    photo = ImageTk.PhotoImage(image)
    label = Label(separator, image=photo)
    label.photo = photo
    label.pack()
    separator.pack(fill=X, padx=10)

    s = Style()
    s.configure("Visible.TButton", foreground="red", background="pink")

    image = Image.open("../faces.jpeg")
    photo = ImageTk.PhotoImage(image)
    b = Button(root, image=photo, style="Visible.TButton", cursor="dot")
    b.pack(side=LEFT, expand=1)

    image1 = Image.open("../m_bg.png")
    photo1 = ImageTk.PhotoImage(image1)
    b1 = Button(root, image=photo1, style="Visible.TButton", cursor="dot",
                command=lambda: callback(vs, pipeline, label, root))
    b1.pack(side=LEFT, expand=1)

    image2 = Image.open("../mermaid_1.jpg")
    photo2 = ImageTk.PhotoImage(image2)
    b2 = Button(root, image=photo2, style="Visible.TButton", cursor="dot")
    b2.pack(side=LEFT, expand=1)

    scrollbar = Scrollbar(root, orient=HORIZONTAL)
    scrollbar.pack(fill=X, side=LEFT, expand=FALSE)

    root.mainloop()


if __name__ == '__main__':
    main()
