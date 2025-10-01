import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# --- loadin the data ---
# had to use this encoding thing b/c of an error
df = pd.read_csv('spam.csv', encoding='latin1')

# dont need the empty columns
df = df[['v1', 'v2']]
df.columns = ['Category', 'Message'] # better names

# spam is 1, ham is 0
df['Category'] = df['Category'].map({'spam': 1, 'ham': 0})

X = df['Message']
y = df['Category']

# split it up
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# ====== model stuff ======

# this pipeline thing is cool, does two steps in one
# Tfidf is suposed to be better than countvectorizer
model_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('classifier', MultinomialNB())
])

# train it
model_pipeline.fit(X_train, y_train)

# make predictshuns
predictions = model_pipeline.predict(X_test)

# ===== results =====

print("--- Howd we do? ---")
print(f"Acuracy: {accuracy_score(y_test, predictions):.2%}\n") # show as percent

# more details
print("Clasification Report:")
print(classification_report(y_test, predictions, target_names=['Ham', 'Spam']))

# show the confuson matrix, looks nicer as a heatmap
cm = confusion_matrix(y_test, predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
            xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.xlabel('Predicted') # what the model guesed
plt.ylabel('Acutal')
plt.title('Confuson Matrix')
plt.show() # show the plot