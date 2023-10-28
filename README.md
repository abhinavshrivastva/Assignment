## Introduction 
The dataset contains of 30k Hindi sentences which may or may not be NSFW. The dataset is balanced with nearly half of examples as NSFW. 
For Example,

मुझे आंटी की बातों का मर्म ये समझ में आया कि इस सब गुस्से की वजह रात को अंकल ज्यादा देर तक आंटी की चुदाई नहीं कर पाते हैं और आंटी को शांत नहीं कर पाते हैं - 1
सब तुम्हारी वजह से - 0

## Data Preprocesing

Installed the Pandas library and imported it. Read a CSV file ('30k_sample.csv') into a Pandas DataFrame and displayed the initial rows. Removed sentences with only spaces to clean the DataFrame. Eliminated emojis from the 'sentence' column using a regex-based 'remove_emojis' function. Displayed the initial rows of the DataFrame after emoji removal.Utilized the Pandas library and TensorFlow's Keras preprocessing tools to tokenize a dataset. The dataset, named 'data,' was used to create tokenized sentences and corresponding labels. A Tokenizer was initialized to process the text data. For each sentence, whitespace-separated words were tokenized, and the Tokenizer was updated with these tokenized sentences. The result was a new DataFrame called 'tokenized_data' with tokenized sentences and labels. The final DataFrame had the number of rows and columns displayed, with the first five rows also shown. This process allowed text data to be converted into a format suitable for machine learning applications.

The vectorization process converts text data into numerical features suitable for machine learning models.The CountVectorizer was used to convert the text data into a numerical format. It created a vocabulary of words from the 'sentence' column and represented each sentence as a vector with counts of these words. The 'max_features' parameter was set to limit the vocabulary size to 10,000 words.The resulting 'X_train_vectorized' and 'X_test_vectorized' matrices contain the numerical representations of the training and testing text data, respectively.

## Model Selection 

### Logistic Regression:

Once the data was vectorized, a Logistic Regression model was applied to the training data ('X_train_vectorized' and 'y_train') as a starting point. The model was trained to predict the 'label' (0 or 1) based on the vectorized text data.Accuracy was calculated to evaluate the model's performance on the test data with a decent starting score of 71%.

### Naive Bayes:

The Naive Bayes classifier, in this case, a Binomial Naive Bayes, was applied using a pipeline that included Count Vectorization. The model was trained on the same dataset used earlier, where 'sentence' is the text data, and 'label' is the binary classification label.

However, the performance of the Binomial Naive Bayes model did not perform as well as the previously used Logistic Regression model. The accuracy score for the Naive Bayes model was approximately 63.33%, which is significantly lower compared to the basic Logistic Regression model used earlier.

This lower accuracy may be attributed to the assumptions of Naive Bayes, particularly its independence assumptions. In text classification tasks, where the order and relationships between words can be crucial, Naive Bayes may struggle to capture the intricacies of the data compared to Logistic Regression. The choice of classifier can greatly impact model performance, and in this case, Logistic Regression outperformed Naive Bayes for the given text classification task.

### XGBoost :

The XGBoost classifier was employed to handle text classification tasks using the 'sentence' column for text data and the 'label' column for labels. The dataset was split into training and testing sets, just as in previous tasks, with 80% for training and 20% for testing.

To convert the text data into suitable features, the TF-IDF (Term Frequency-Inverse Document Frequency) vectorization technique was applied. The TF-IDF vectorizer created numerical representations of the text data, and the maximum number of features was set to 10,000.

Subsequently, an XGBoost classifier was utilized for training the model on the TF-IDF transformed training data ('X_train_tfidf' and 'y_train'). The model then made predictions on the test data, and its performance was evaluated.

The XGBoost model achieved an accuracy of approximately 69.53%, which is an improvement over the Multinomial Naive Bayes model but still lower than the Logistic Regression model. This demonstrates that XGBoost, a gradient boosting algorithm, can provide better predictive performance for text classification tasks compared to Naive Bayes but may still be outperformed by simpler models like Logistic Regression in some cases.

### Decision Tree:

A Decision Tree Classifier was utilized for text classification tasks, similar to the previous models. The dataset was split into training and testing sets, with an 80% training and 20% testing split, just as before.

A pipeline was created to include TF-IDF Vectorization and the Decision Tree Classifier. The TF-IDF vectorizer transformed the text data into numerical features, and the Decision Tree Classifier was used to build the classification model.

The performance of the Decision Tree Classifier was evaluated, and it achieved an accuracy of approximately 63.33%. This accuracy is consistent with the performance of the Multinomial Naive Bayes model, but it remains lower than the Logistic Regression and XGBoost models for the given text classification task.


###  CNN :

In this text classification task, a Convolutional Neural Network (CNN) model was employed to effectively handle the data. CNNs are suitable when the local order of words is not crucial for text classification, as they can detect specific features within the text.

The text data was tokenized and converted into sequences, with a maximum sequence length set to 100. Padding was applied to ensure consistent input size.

The CNN model was constructed, consisting of an embedding layer, convolutional layers, max-pooling, and fully connected layers with dropout. It was compiled with appropriate loss and metrics.

The model was then trained on the training data, and its performance was evaluated on the test data. Remarkably, the CNN model achieved an accuracy of approximately 98.73%, indicating its strong capability in distinguishing between NSFW and SFW terms within the text.

The impressive performance of the Convolutional Neural Network (CNN) in text classification is attributed to its ability to capture significant features within the text data, especially in scenarios where the local order of words may not be as crucial. The CNN leverages filters to identify specific patterns or features, such as NSFW or SFW terms, within sentences.

The model's deep architecture allows it to learn complex relationships between words and their context, making it highly effective in discerning between different classes. The result, with an accuracy of approximately 98.73%, reflects its robust predictive capabilities and superior performance compared to traditional models like Logistic Regression. This makes the CNN a valuable choice for tasks where semantic information and feature extraction play a pivotal role in text classification.


In summary, the CNN model's impressive performance showcases its effectiveness in text classification when the local order of words is not a critical factor, as demonstrated by the high accuracy of 98.73%.

### Transformer Architecture of Facebook Model FastText:

FastText is a popular text classification and representation learning model developed by Facebook AI Research. It's known for its efficiency, scalability, and ability to handle languages with rich morphological structures like Hindi. Here are some key aspects of FastText:

**Architecture**: FastText uses a shallow neural network architecture. It represents words as continuous bag of n-grams (character-level or subword-level) rather than individual words. This approach captures not only word semantics but also subword information, which is particularly beneficial for morphologically complex languages like Hindi. The model can also handle out-of-vocabulary words using subword embeddings.

**Training on Huge Corpora**: FastText is trained on massive text corpora, making it capable of learning rich and robust word embeddings. For Hindi, training on a large corpus helps it capture the nuances of the language, including variations in script, vocabulary, and grammar. This extensive training data ensures that the model can work well with diverse and extensive Hindi text.

**Morphological Awareness**: FastText's subword embeddings enable it to be morphologically aware. It can represent and recognize word forms and variations, which is essential for understanding the highly inflected nature of Hindi.

**Multilingual Capabilities**: FastText is designed to work with multiple languages, and it can effectively handle code-switching and mixed-language text. This makes it a valuable tool for multilingual applications in regions like India, where multiple languages are often used together.

**Text Classification**: FastText is commonly used for text classification tasks. It can be fine-tuned on specific classification tasks, such as sentiment analysis, topic classification, and more, making it versatile for a wide range of applications.

In summary, FastText's architecture and training methodology, combined with its ability to handle morphologically rich languages like Hindi, make it a powerful tool for natural language processing and text classification. It has been used successfully in various real-world applications, including social media analysis, content recommendation, and more.

Remarkably, the model achieved an impressive accuracy of approximately 98.14%, indicating its ability to correctly classify text data. 


##Conclusion :

In conclusion, FastText is a powerful text classification model, particularly suitable for languages with complex morphological structures like Hindi. Its rich subword embeddings and extensive training on massive corpora make it a robust choice for a wide range of NLP tasks. 

However, when comparing FastText with CNN, there are certain advantages to using CNN. CNN's simplicity and the fact that it can handle text classification tasks effectively without the need for heavy subword embeddings make it a more straightforward choice for many applications. Additionally, CNN models often have faster computation times, which can be crucial in real-time or resource-constrained scenarios.

Moreover, FastText models can be relatively large, often exceeding 6GB in size, making them less efficient for deployment in resource-constrained environments. Ultimately, the choice between FastText and CNN depends on the specific requirements and constraints of the task at hand.

##Future Work  :

Future work in the domain of NSFW (Not Safe For Work) text detection presents exciting opportunities for further research and model development. Building upon the foundation of extensive training data, it's possible to create a BERT-like model that excels at understanding the nuances of NSFW and non-NSFW content in text. The core idea would be to leverage a massive corpus of NSFW and non-NSFW text, encompassing a wide range of languages and contexts, to train a model with an even deeper understanding of what constitutes inappropriate or sensitive content. This model could be designed to recognize subtle or context-dependent variations in NSFW text, providing a higher level of accuracy in classification.

Moreover, in future developments, the model can be enhanced to detect and relate sentences that contain slangs or other non-PG friendly content. This expansion would be especially valuable in scenarios where content moderation is essential, such as in social media platforms, online communities, or content filtering systems. The ability to not only identify NSFW content but also understand the context in which it is used can improve the accuracy of classification and reduce false positives.

By incorporating advanced natural language processing techniques, such as contextual embeddings and deep learning architectures, future models can offer more sophisticated and context-aware NSFW detection, thereby contributing to safer and more reliable online environments. The ever-evolving landscape of online content and communication demands ongoing research and development in this crucial area of NLP.

