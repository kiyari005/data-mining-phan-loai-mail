import os
import pandas as pd
import numpy as np
import re
import string
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB  # Naive Bayes phù hợp cho dữ liệu văn bản
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Đường dẫn đến thư mục chứa dữ liệu
base_path = 'D:/z_all download/Khai-thac-du-lieu-main/dataset mail spam or non spam'

# Hàm đọc dữ liệu từ các thư mục con
def load_data(base_path):
    data = []
    labels = []
    for folder in os.listdir(base_path):
        if folder.startswith('enron'):
            spam_path = os.path.join(base_path, folder, 'spam')
            ham_path = os.path.join(base_path, folder, 'ham')

            # Đọc các tệp trong thư mục spam
            for filename in os.listdir(spam_path):
                with open(os.path.join(spam_path, filename), 'r', encoding='latin-1') as f:
                    data.append(f.read())
                    labels.append(1)  # 1 là spam

            # Đọc các tệp trong thư mục ham
            for filename in os.listdir(ham_path):
                with open(os.path.join(ham_path, filename), 'r', encoding='latin-1') as f:
                    data.append(f.read())
                    labels.append(0)  # 0 là ham
    return data, labels

# Đọc dữ liệu
emails, labels = load_data(base_path)

# Tạo DataFrame từ dữ liệu đọc được
df = pd.DataFrame({'email': emails, 'label': labels})

# Kiểm tra số lượng email spam và không phải spam
print(df['label'].value_counts())

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

# Áp dụng tiền xử lý
df['clean_email'] = df['email'].apply(clean_text)

# Chia dữ liệu thành tập huấn luyện (80%) và kiểm tra (20%)
X_train, X_test, y_train, y_test = train_test_split(df['clean_email'], df['label'], test_size=0.2, random_state=42)

# Chuyển đổi văn bản thành vector số bằng TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Khởi tạo mô hình Naive Bayes
nb_model = MultinomialNB()  # Sử dụng MultinomialNB cho dữ liệu văn bản
nb_model.fit(X_train_tfidf, y_train)

# Dự đoán trên tập kiểm tra
y_pred = nb_model.predict(X_test_tfidf)
y_prob = nb_model.predict_proba(X_test_tfidf)[:, 1]

# Đánh giá mô hình
accuracy = accuracy_score(y_test, y_pred)
print(f'Độ chính xác của mô hình: {accuracy:.4f}')
print('Báo cáo phân loại:')
print(classification_report(y_test, y_pred))

# Vẽ ma trận nhầm lẫn
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.xlabel('Dự đoán')
plt.ylabel('Thực tế')
plt.title('Ma trận nhầm lẫn')
plt.show()