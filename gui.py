from tkinter import Tk, LEFT, SUNKEN, X
from tkinter.ttk import Frame, Button, Style
from PIL import Image, ImageTk


def main():
    root = Tk()
    root.geometry("300x300")

    separator = Frame(root, height=200, relief=SUNKEN)
    separator.pack(fill=X, padx=10)

    s = Style()
    s.configure("Visible.TButton", foreground="red", background="pink")

    frame = Frame(root)
    frame.pack_propagate(0)
    image = Image.open("faces.jpeg")
    photo = ImageTk.PhotoImage(image)
    b = Button(root, image=photo, style="Visible.TButton", cursor="dot")
    b.pack(side=LEFT, expand=1)

    image1 = Image.open("m_bg.png")
    photo1 = ImageTk.PhotoImage(image1)
    b1 = Button(root, image=photo1, style="Visible.TButton", cursor="dot")
    b1.pack(side=LEFT, expand=1)

    image2 = Image.open("mermaid_1.jpg")
    photo2 = ImageTk.PhotoImage(image2)
    b2 = Button(root, image=photo2, style="Visible.TButton", cursor="dot")
    b2.pack(side=LEFT, expand=1)
    frame.pack(fill=X)
    root.mainloop()


if __name__ == '__main__':
    main()
