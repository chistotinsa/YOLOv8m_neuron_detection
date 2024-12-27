import os
import tkinter as tk
from tkinter import filedialog, messagebox
import threading
from neuron_count import run
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, '..', 'models', 'YOLOv8m_brain_cell_v3_maP50_0.742.pt')


class NeuronCounterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Neuron Counter")
        self.root.geometry("400x300")

        self.photo_path = ""
        self.results_path = ""

        self.create_widgets()

    def create_widgets(self):
        label = tk.Label(self.root, text="Neuron Counting in Hippocampus", font=("Arial", 14))
        label.pack(pady=10)

        self.select_photo_button = tk.Button(self.root, text="Select Photos Folder", command=self.select_photo_folder)
        self.select_photo_button.pack(pady=5)

        self.select_results_button = tk.Button(self.root, text="Select Results File", command=self.select_results_file)
        self.select_results_button.pack(pady=5)

        self.start_button = tk.Button(self.root, text="Start Counting", command=self.start_counting)
        self.start_button.pack(pady=20)

        self.status_label = tk.Label(self.root, text="", font=("Arial", 10))
        self.status_label.pack(pady=10)

    def select_photo_folder(self):
        self.photo_path = filedialog.askdirectory(title="Select Folder with Photos")
        if self.photo_path:
            self.status_label.config(text=f"Photos folder selected: {self.photo_path}")

    def select_results_file(self):
        self.results_path = filedialog.asksaveasfilename(defaultextension=".xlsx",
                                                         filetypes=[("Excel Files", "*.xlsx")],
                                                         title="Select Result File")
        if self.results_path:
            self.status_label.config(text=f"Results file selected: {self.results_path}")

    def start_counting(self):
        if not self.photo_path or not self.results_path:
            messagebox.showerror("Error", "Please select both photos folder and result file!")
            return

        self.status_label.config(text="Counting neurons... Please wait.")
        self.start_button.config(state=tk.DISABLED)

        threading.Thread(target=self.run_analysis).start()

    def run_analysis(self):
        try:
            run(self.photo_path, self.results_path, save=True, model_path=model_path)
            self.status_label.config(text="Counting completed successfully!")
            messagebox.showinfo("Success", "Neuron counting completed successfully!")
        except Exception as e:
            self.status_label.config(text="Error occurred!")
            messagebox.showerror("Error", f"An error occurred: {e}")
        finally:
            self.start_button.config(state=tk.NORMAL)


root = tk.Tk()
app = NeuronCounterApp(root)
root.mainloop()
