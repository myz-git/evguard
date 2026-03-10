import customtkinter as ctk
import tkinter as tk

# Create main window
root = ctk.CTk()
root.title("EVGuard Control Panel")

# Set window size and style (Windows 9x theme)
root.geometry("800x600")
root.tk_setPalette(background="#C0C0C0")  # Classic Windows 9x-style background color

# Frame A layout
frame_A = ctk.CTkFrame(root, width=400, height=600)
frame_A.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

# Function list section
def create_function_frame(parent, label):
    frame = ctk.CTkFrame(parent)
    label = ctk.CTkLabel(frame, text=label)
    label.pack(side="left", padx=5)
    start_button = ctk.CTkButton(frame, text="Start")
    start_button.pack(side="left", padx=5)
    stop_button = ctk.CTkButton(frame, text="Stop")
    stop_button.pack(side="left", padx=5)
    status_label = ctk.CTkLabel(frame, text="Status: Stopped")
    status_label.pack(side="right", padx=5)
    frame.pack(fill="x", pady=5)

# Create five function frames
create_function_frame(frame_A, "Fsd0")
create_function_frame(frame_A, "Fsd1")
create_function_frame(frame_A, "GuardA")
create_function_frame(frame_A, "GuardB")

# Bottom frame for status refresh and quit
bottom_frame = ctk.CTkFrame(frame_A)
status_refresh_button = ctk.CTkButton(bottom_frame, text="Status Refresh")
status_refresh_button.pack(side="left", padx=5)
quit_button = ctk.CTkButton(bottom_frame, text="Exit All")
quit_button.pack(side="left", padx=5)
bottom_frame.pack(fill="x", pady=10)

# Frame B layout
frame_B = ctk.CTkFrame(root, width=400, height=600)
frame_B.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

# Console section
console_frame = ctk.CTkFrame(frame_B)
console_label = ctk.CTkLabel(console_frame, text="Console Output")
console_label.pack(pady=5)
console_text = ctk.CTkTextbox(console_frame, width=350, height=300)
console_text.pack(pady=5)
auto_scroll_checkbox = ctk.CTkCheckBox(console_frame, text="Auto Scroll")
auto_scroll_checkbox.pack(side="left", padx=5)
clear_log_button = ctk.CTkButton(console_frame, text="Clear Log")
clear_log_button.pack(side="left", padx=5)
detailed_log_button = ctk.CTkButton(console_frame, text="View Detailed Log")
detailed_log_button.pack(side="left", padx=5)
console_frame.pack(fill="both", expand=True, pady=10)

# Start the GUI
root.mainloop()