import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import joblib

# Tải các công cụ NLP
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Hàm tiền xử lý văn bản
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

# Load mô hình và vectorizer
rf_model = joblib.load('random_forest_model.pkl')
nb_model = joblib.load('naive_bayes_model.pkl')
dt_model = joblib.load('decision_tree_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Hàm dự đoán email
def predict_email(email_content, algorithm):
    # Tiền xử lý email
    cleaned_email = clean_text(email_content)
    vectorized_test_email = vectorizer.transform([cleaned_email])

    # Lựa chọn mô hình dựa trên thuật toán
    selected_model = None
    if algorithm == 'Random Forest':
        selected_model = rf_model
    elif algorithm == 'Naive Bayes':
        selected_model = nb_model
    else:  # Decision Tree
        selected_model = dt_model

    # Thực hiện dự đoán
    prediction = selected_model.predict(vectorized_test_email)[0]
    probability = selected_model.predict_proba(vectorized_test_email)[0][1]
    return prediction, probability

# Hàm xử lý khi nhấn nút "Dự đoán Email"
def on_predict_click():
    email_content = email_input.get("1.0", "end-1c").strip()  # Lấy nội dung từ Text widget
    if not email_content:
        messagebox.showwarning("Cảnh báo", "Vui lòng nhập nội dung email để kiểm tra!")
        return

    algorithm = algorithm_var.get()  # Lấy thuật toán từ dropdown
    prediction, probability = predict_email(email_content, algorithm)

    # Hiển thị kết quả
    result_text = f"📩 **Kết quả dự đoán ({algorithm}):**\n"
    if prediction == 1:
        result_text += f"🚨 **SPAM!** (Xác suất: {probability:.2%})"
    else:
        result_text += f"✅ **Email an toàn.** (Xác suất spam: {probability:.2%})"
    result_label.config(text=result_text)

# Hàm xử lý khi nhấn nút "Tải lên file"
def on_upload_click():
    file_path = filedialog.askopenfilename(filetypes=(("Text files", "*.txt"), ("All files", "*.*")))
    if file_path:
        with open(file_path, 'r', encoding='utf-8') as file:
            email_content = file.read()
            email_input.delete("1.0", "end")  # Xóa nội dung cũ
            email_input.insert("1.0", email_content)  # Chèn nội dung file

# Hàm xử lý khi nhấn nút "Hủy file tải lên"
def on_cancel_click():
    email_input.delete("1.0", "end")  # Xóa nội dung trong Text widget
    result_label.config(text="")  # Xóa kết quả

# Tạo cửa sổ chính
root = tk.Tk()
root.title("Dự đoán Email Spam")

# Tạo dropdown để chọn thuật toán
algorithm_var = tk.StringVar(value="Random Forest")
algorithm_dropdown = tk.OptionMenu(root, algorithm_var, "Random Forest", "Naive Bayes", "Decision Tree")
algorithm_dropdown.pack(pady=10)

# Tạo nút để tải lên file
upload_button = tk.Button(root, text="Tải lên file", command=on_upload_click)
upload_button.pack(pady=10)

# Tạo hộp nhập liệu cho email
email_input = tk.Text(root, height=10, width=50)
email_input.pack(pady=10)

# Tạo nút để dự đoán
predict_button = tk.Button(root, text="Dự đoán Email", command=on_predict_click)
predict_button.pack(pady=10)

# Tạo nút hủy file tải lên
cancel_button = tk.Button(root, text="Hủy file tải lên", command=on_cancel_click)
cancel_button.pack(pady=10)

# Tạo nhãn để hiển thị kết quả
result_label = tk.Label(root, text="", font=("Arial", 12))
result_label.pack(pady=10)

# Chạy ứng dụng
root.mainloop()