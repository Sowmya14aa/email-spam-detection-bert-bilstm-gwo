This project presents an advanced Email Spam Detection System that uses BERT-based contextual embeddings, deep learning models such as CNN, LSTM, BiLSTM, and GRU, and optimization algorithms like Grey Wolf Optimization (GWO) and Particle Swarm Optimization (PSO) to improve classification performance. The system processes email text through a complete pipeline of cleaning, embedding generation, deep learning classification, and optimization, with the BiLSTM + BERT model achieving the best accuracy. A Tkinter-based user interface allows users to enter email text and receive instant spam or non-spam predictions, making the solution practical, user-friendly, and efficient for real-time email filtering.

✅ Problems in Older Systems

Keyword-based filters fail to understand semantic meaning.

High false positives and false negatives due to limited context.

Traditional ML models (Naive Bayes, SVM, Random Forest) struggle with evolving spam patterns.

Poor performance on short or ambiguous emails.

Easily bypassed using misspellings, obfuscation, or hidden text.

Lack of adaptability to new spam techniques.

✅ Proposed Model

Use BERT to generate contextualized embeddings of email text.

Train multiple deep learning architectures (CNN, LSTM, BiLSTM, GRU).

Apply GWO for selecting the most important features.

Use PSO for tuning hyperparameters to reach optimal performance.

Evaluate all models and select the best-performing configuration.

Provide a Tkinter GUI for real-time spam prediction.

✅ Technologies Used

Python

NLP Libraries: NLTK, SpaCy, HuggingFace Transformers

Deep Learning: TensorFlow, Keras

Machine Learning: Scikit-learn

Optimization Algorithms: PyGWO, PySwarms

User Interface: Tkinter

Data Handling: NumPy, Pandas

Visualization: Matplotlib, Seaborn

✅ Dependencies to Install Before Running

Create a virtual environment (optional but recommended):

python -m venv env
source env/bin/activate   # Linux/Mac
env\Scripts\activate      # Windows


Install required libraries:

pip install numpy pandas scikit-learn tensorflow keras nltk spacy matplotlib seaborn transformers pygwo pyswarms


For Tkinter UI (Windows/Linux usually included):

pip install tk


If using Jupyter Notebook:

pip install notebook
