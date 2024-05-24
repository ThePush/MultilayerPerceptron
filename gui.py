import customtkinter as ctk
from tkinter import messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from MultilayerPerceptron import MultilayerPerceptron


ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class MLPApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Multilayer Perceptron GUI")
        self.attributes("-zoomed", True)
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.custom_font = ("Ubuntu", 12)

        self.mlp = MultilayerPerceptron()

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=3)

        self.setup_widgets()

    def setup_widgets(self):
        # Main Frame
        main_frame = ctk.CTkFrame(self)
        main_frame.grid(
            row=0,
            column=0,
            sticky="nsew",
            padx=10,
            pady=10,
        )
        main_frame.grid_rowconfigure(0, weight=1)
        main_frame.grid_columnconfigure(1, weight=1)

        # Parameters Frame (left side)
        parameters_frame = ctk.CTkFrame(main_frame)
        parameters_frame.grid(
            row=0,
            column=0,
            sticky="ns",
            padx=10,
            pady=10,
        )

        # Plot Frame (right side)
        self.plot_frame = ctk.CTkFrame(main_frame)
        self.plot_frame.grid(
            row=0,
            column=1,
            sticky="nsew",
            padx=10,
            pady=10,
        )

        # Labels and Entry widgets with grid layout inside parameters_frame
        labels = [
            "Path to dataset",
            "Layers (e.g., 24 24 24)",
            "Epochs",
            "Batch size",
            "Learning rate",
            "Initializer",
            "Activation function",
            "Optimizer",
            "Momentum",
            "Lambda regularization",
            "Patience",
        ]
        for i, text in enumerate(labels):
            ctk.CTkLabel(parameters_frame, text=text).grid(
                row=i,
                column=0,
                padx=5,
                pady=5,
                sticky="w",
            )

        self.entry_dataset_path = ctk.CTkEntry(
            parameters_frame,
            width=200,
            corner_radius=10,
            font=self.custom_font,
        )
        self.entry_dataset_path.grid(
            row=0,
            column=1,
            padx=5,
            pady=5,
            sticky="ew",
        )

        self.entry_layers = ctk.CTkEntry(parameters_frame, width=200)
        self.entry_layers.grid(
            row=1,
            column=1,
            padx=5,
            pady=5,
            sticky="ew",
        )

        self.entry_epochs = ctk.CTkEntry(parameters_frame, width=200)
        self.entry_epochs.grid(
            row=2,
            column=1,
            padx=5,
            pady=5,
            sticky="ew",
        )

        self.entry_batch_size = ctk.CTkEntry(parameters_frame, width=200)
        self.entry_batch_size.grid(
            row=3,
            column=1,
            padx=5,
            pady=5,
            sticky="ew",
        )

        self.entry_learning_rate = ctk.CTkEntry(parameters_frame, width=200)
        self.entry_learning_rate.grid(
            row=4,
            column=1,
            padx=5,
            pady=5,
            sticky="ew",
        )

        # ComboBox for initializer, activation function, and optimizer
        self.combobox_initializer = ctk.CTkComboBox(
            parameters_frame,
            values=["he_uniform", "he_normal", "xavier_uniform", "xavier_normal"],
        )
        self.combobox_initializer.grid(
            row=5,
            column=1,
            padx=5,
            pady=5,
            sticky="ew",
        )

        self.combobox_activation = ctk.CTkComboBox(
            parameters_frame, values=["sigmoid", "relu"]
        )
        self.combobox_activation.grid(
            row=6,
            column=1,
            padx=5,
            pady=5,
            sticky="ew",
        )

        self.combobox_optimizer = ctk.CTkComboBox(
            parameters_frame,
            values=["gradient_descent", "adam", "nesterov"],
        )
        self.combobox_optimizer.grid(
            row=7,
            column=1,
            padx=5,
            pady=5,
            sticky="ew",
        )

        self.entry_momentum = ctk.CTkEntry(parameters_frame, width=200)
        self.entry_momentum.grid(
            row=8,
            column=1,
            padx=5,
            pady=5,
            sticky="ew",
        )

        self.entry_lambda_reg = ctk.CTkEntry(parameters_frame, width=200)
        self.entry_lambda_reg.grid(
            row=9,
            column=1,
            padx=5,
            pady=5,
            sticky="ew",
        )

        self.entry_patience = ctk.CTkEntry(parameters_frame, width=200)
        self.entry_patience.grid(
            row=10,
            column=1,
            padx=5,
            pady=5,
            sticky="ew",
        )

        self.train_button = ctk.CTkButton(
            parameters_frame,
            text="Train Model",
            command=self.train_model,
        )
        self.train_button.grid(
            row=11,
            column=0,
            columnspan=2,
            pady=20,
        )

        # Label and Entry for the dataset path specifically for splitting
        ctk.CTkLabel(parameters_frame, text="Dataset to Split").grid(
            row=12,
            column=0,
            padx=5,
            pady=5,
            sticky="w",
        )
        self.entry_dataset_split_path = ctk.CTkEntry(
            parameters_frame,
            width=200,
            placeholder_text="e.g., data/data.csv",
        )
        self.entry_dataset_split_path.grid(
            row=12,
            column=1,
            padx=5,
            pady=5,
            sticky="ew",
        )

        # Label and Entry for the split ratio
        ctk.CTkLabel(parameters_frame, text="Split Ratio").grid(
            row=13,
            column=0,
            padx=5,
            pady=5,
            sticky="w",
        )
        self.entry_split_ratio = ctk.CTkEntry(
            parameters_frame, width=200, placeholder_text="e.g., 0.25"
        )
        self.entry_split_ratio.grid(
            row=13,
            column=1,
            padx=5,
            pady=5,
            sticky="ew",
        )

        # Button for splitting the dataset
        self.split_button = ctk.CTkButton(
            parameters_frame,
            text="Split Dataset",
            command=self.split_dataset,
        )
        self.split_button.grid(
            row=14,
            column=0,
            columnspan=2,
            pady=20,
        )

        # Add a new row for the prediction dataset path
        ctk.CTkLabel(parameters_frame, text="Dataset for Prediction").grid(
            row=15,
            column=0,
            padx=5,
            pady=5,
            sticky="w",
        )
        self.entry_predict_dataset_path = ctk.CTkEntry(
            parameters_frame,
            width=200,
            placeholder_text="e.g., data/test.csv",
        )
        self.entry_predict_dataset_path.grid(
            row=15,
            column=1,
            padx=5,
            pady=5,
            sticky="ew",
        )

        # Add a new row for the prediction button
        self.predict_button = ctk.CTkButton(
            parameters_frame, text="Predict", command=self.predict
        )
        self.predict_button.grid(
            row=16,
            column=0,
            columnspan=2,
            pady=20,
        )

        self.set_default_values()

    def set_default_values(self):
        self.entry_dataset_path.insert(0, "data/train.csv")
        self.entry_layers.insert(0, "8 8")
        self.entry_epochs.insert(0, "200")
        self.entry_batch_size.insert(0, "")
        self.entry_learning_rate.insert(0, "0.00314")
        self.combobox_initializer.set("xavier_uniform")
        self.combobox_activation.set("sigmoid")
        self.combobox_optimizer.set("nesterov")
        self.entry_momentum.insert(0, "0.9")
        self.entry_lambda_reg.insert(0, "0.0")
        self.entry_patience.insert(0, "20")
        self.entry_dataset_split_path.insert(0, "data/data.csv")
        self.entry_split_ratio.insert(0, "0.25")
        self.entry_predict_dataset_path.insert(0, "data/test.csv")

    def on_closing(self):
        """Clean up any background tasks before closing the window."""
        # Destroy the matplotlib plot properly
        if hasattr(self, "plot_canvas"):
            self.plot_canvas.get_tk_widget().destroy()
            plt.close("all")

        self.quit()
        self.destroy()

    def train_model(self):
        try:
            # Retrieve all values from the entries
            batch_size_value = self.entry_batch_size.get()
            batch_size = int(batch_size_value) if batch_size_value else None

            params = {
                "dataset_path": self.entry_dataset_path.get(),
                "layer": list(map(int, self.entry_layers.get().split())),
                "epochs": int(self.entry_epochs.get()),
                "batch_size": batch_size,
                "learning_rate": float(self.entry_learning_rate.get()),
                "initializer": self.combobox_initializer.get(),
                "activation": self.combobox_activation.get(),
                "optimizer": self.combobox_optimizer.get(),
                "momentum": float(self.entry_momentum.get()),
                "lambda_reg": float(self.entry_lambda_reg.get()),
                "patience": int(self.entry_patience.get()),
                "random_state": 42,
                "plot_results": False,  # We will handle plotting in the GUI
            }

            logger.info(f"Training model with parameters: {params}")

            model = self.mlp.mlp_train(**params)
            logger.success("Model trained successfully.")
            self.plot_results(model)
        except Exception as e:
            logger.error(f"An error occurred: {e}")
            messagebox.showerror("Training Error", f"An error occurred: {str(e)}")

    def plot_results(self, model):
        if not hasattr(self, "figure"):
            self.figure, self.ax = plt.subplots(figsize=(5, 4))
            self.plot_canvas = FigureCanvasTkAgg(self.figure, self.plot_frame)
            self.plot_canvas.get_tk_widget().grid(
                row=0,
                column=0,
                sticky="nsew",
            )
        plt.figure(figsize=(18, 10))

        metrics = ["train", "valid"]
        titles = ["Loss", "Accuracy", "Precision", "Recall", "F1 Score"]
        data_keys = [
            "log_loss_history",
            "accuracy_history",
            "precision_history",
            "recall_history",
            "f1_score_history",
        ]

        for i, key in enumerate(data_keys, 1):
            plt.subplot(2, 3, i)
            for metric in metrics:
                plt.plot(
                    getattr(model, key)[metric],
                    label=f"{metric.capitalize()} {titles[i-1]}",
                )

            if titles[i - 1] in ["Loss", "Accuracy"]:
                # Adding markers for best values
                if titles[i - 1] == "Loss":
                    epoch_of_best = np.argmin(getattr(model, key)["valid"])
                    best_value = getattr(model, key)["valid"][epoch_of_best]
                    marker_label = "Best Validation Loss"
                    ylabel = "Binary Cross-Entropy Loss"
                else:
                    epoch_of_best = np.argmax(getattr(model, key)["valid"])
                    best_value = getattr(model, key)["valid"][epoch_of_best]
                    marker_label = "Best Validation Accuracy"
                    ylabel = "Accuracy"

                plt.scatter(
                    epoch_of_best,
                    best_value,
                    color="red",
                    label=marker_label,
                    zorder=5,
                )
                plt.annotate(
                    (
                        f"{best_value:.4f}"
                        if titles[i - 1] == "Loss"
                        else f"{best_value:.2%}"
                    ),
                    (float(epoch_of_best), float(best_value)),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha="center",
                )

            plt.title(titles[i - 1])
            plt.xlabel("Epoch")
            plt.ylabel(ylabel)
            plt.legend()

        # Clear previous plots in the plot frame
        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        # Embed the plot in the GUI
        canvas = FigureCanvasTkAgg(plt.gcf(), master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def split_dataset(self):
        dataset_path = self.entry_dataset_split_path.get()
        try:
            split_ratio = float(self.entry_split_ratio.get())
            if not 0 < split_ratio < 1:
                logger.error("Split ratio must be between 0 and 1")
                raise ValueError("Split ratio must be between 0 and 1")
        except ValueError as e:
            logger.error(f"Invalid split ratio: {e}")
            messagebox.showerror("Error", f"Invalid split ratio: {e}")
            return

        try:
            self.mlp.mlp_split(dataset_path, split_ratio)
            logger.info("Dataset successfully split.")
            # messagebox.showinfo("Success", "Dataset successfully split.")
        except Exception as e:
            logger.error(f"Failed to split dataset: {e}")
            messagebox.showerror("Error", f"Failed to split dataset: {e}")

    def predict(self):
        dataset_path = self.entry_predict_dataset_path.get()
        try:
            # Trigger the prediction method of your MLP class
            (
                accuracy,
                precision,
                recall,
                f1,
                loss,
            ) = self.mlp.mlp_predict(dataset_path)
            messagebox.showinfo(
                "Prediction Results",
                f"Accuracy: {accuracy:.2f}\n"
                f"Precision: {precision:.2f}\n"
                f"Recall: {recall:.2f}\n"
                f"F1 Score: {f1:.2f}\n"
                f"Loss: {loss:.2f}",
            )
        except Exception as e:
            logger.error(f"Failed to perform prediction: {e}")
            messagebox.showerror("Error", f"Failed to perform prediction: {e}")


if __name__ == "__main__":
    try:
        app = MLPApp()
        app.mainloop()
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        messagebox.showerror("Error", f"An error occurred: {e}")
