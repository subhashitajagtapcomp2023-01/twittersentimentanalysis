import tkinter as tk
from tkinter import messagebox
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ---------------- LOAD MODEL ----------------
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# ---------------- PREPROCESS FUNCTION ----------------
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|@\S+|#\S+|[^a-z\s]", "", text)
    text = " ".join(
        lemmatizer.lemmatize(word)
        for word in text.split()
        if word not in stop_words
    )
    return text

# ---------------- PREDICT FUNCTION ----------------
def predict_sentiment():
    user_text = text_input.get("1.0", tk.END).strip()
    if user_text == "":
        messagebox.showwarning("Input Error", "Please enter some text!")
        return

    clean_text = preprocess_text(user_text)
    vector = vectorizer.transform([clean_text])
    prediction = model.predict(vector)[0]

    if prediction == "positive":
        result_label.config(
            text="😊 Positive Sentiment",
            fg="#1DA1F2"
        )
    elif prediction == "negative":
        result_label.config(
            text="😠 Negative Sentiment",
            fg="#E0245E"
        )
    else:
        result_label.config(
            text="😐 Neutral Sentiment",
            fg="#657786"
        )

# ---------------- GUI WINDOW ----------------
root = tk.Tk()
root.title("Twitter Sentiment Analyzer")
root.geometry("520x420")
root.configure(bg="#E8F5FD")  # Twitter-like background

# ---------------- HEADER ----------------
header = tk.Label(
    root,
    text="🐦 Twitter Sentiment Analyzer",
    font=("Helvetica", 18, "bold"),
    bg="#1DA1F2",
    fg="white",
    pady=12
)
header.pack(fill="x")

# ---------------- INPUT LABEL ----------------
input_label = tk.Label(
    root,
    text="Enter a Tweet or Text:",
    font=("Helvetica", 12),
    bg="#E8F5FD"
)
input_label.pack(pady=(20, 5))

# ---------------- TEXT BOX ----------------
text_input = tk.Text(
    root,
    height=6,
    width=50,
    font=("Helvetica", 11),
    wrap="word",
    bd=2,
    relief="groove"
)
text_input.pack(pady=5)

# ---------------- BUTTON ----------------
predict_button = tk.Button(
    root,
    text="Check Sentiment",
    font=("Helvetica", 12, "bold"),
    bg="#1DA1F2",
    fg="black",
    activebackground="#0d8ddb",
    padx=20,
    pady=8,
    command=predict_sentiment
)
predict_button.pack(pady=20)

# ---------------- RESULT ----------------
result_label = tk.Label(
    root,
    text="",
    font=("Helvetica", 15, "bold"),
    bg="#E8F5FD"
)
result_label.pack(pady=10)

# ---------------- FOOTER ----------------
footer = tk.Label(
    root,
    text="ML-based Sentiment Analysis | Python • NLP",
    font=("Helvetica", 9),
    bg="#E8F5FD",
    fg="#657786"
)
footer.pack(side="bottom", pady=10)

root.mainloop()
