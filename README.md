✈️ Twitter Sentiment Analysis 🌟

This project focuses on sentiment analysis for airline-related tweets using the RoBERTa model. The goal is to classify tweets into positive, negative, and neutral sentiments while addressing challenges like informal language, slang, and class imbalance.

🚀 Features

🗂 Dataset: Uses the Twitter US Airline Sentiment dataset (~15,000 tweets).
🧹 Preprocessing: Cleaning, tokenization, and handling of class imbalance.
🤖 Model: Fine-tuned RoBERTa, compared against Logistic Regression and Naive Bayes.
📊 Evaluation: Precision, Recall, F1-Score, and Accuracy metrics.
🛠 Explainability: Integrated SHAP analysis for understanding model decisions.
📈 Results

RoBERTa demonstrated superior performance:

✅ Accuracy: 84.3%
🏆 F1-Score: 85.2%
📁 Repository Contents

🗃️ Tweets.csv: Dataset of airline tweets.
🧑‍💻 twitter_sentiment_analysis.py: Script for fine-tuning and evaluating RoBERTa.
🔍 evaluate_checkpoints.py: Evaluates saved model checkpoints.
🤔 inference.py: Performs sentiment prediction on new data.
📊 visualization.py: Visualizes results and SHAP analysis.
🖼️ Figures: Graphical representations of results.
🛠️ Prerequisites

🐍 Python: Version 3.7 or higher
📦 Libraries:
PyTorch
Transformers (HuggingFace)
SHAP
Matplotlib
⚙️ Getting Started

Clone the repository:
git clone https://github.com/Rishi4227/Twitter_Sentiment_Analysis.git
cd Twitter_Sentiment_Analysis
Install dependencies:
pip install -r requirements.txt
Run the sentiment analysis:
python twitter_sentiment_analysis.py
🚀 Future Enhancements

🤝 Explore hybrid models like RoBERTa-GRU.
⚡ Optimize performance with lightweight models such as DistilBERT.
🌐 Integrate multi-modal data (e.g., text and images) for comprehensive analysis.
🙏 Credits

This project was implemented as part of academic coursework/research.
The Twitter US Airline Sentiment dataset was sourced from Kaggle.
