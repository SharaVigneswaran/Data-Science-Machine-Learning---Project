<div align="center">
  
# Welcome to LogoRank 
</div>
<div align="center">
  <div style="position: relative; text-align: right;">
  <img src="LogoRank.jpeg" alt="LogoRank" style="width: 200px; height: auto; position: absolute; top: 10px; right: 10px;">
</div>
</div>

**LogoRank** is a visionary startup. Our mission is to enhance language learning experiences and support learners in progressing at their own pace using innovative technology.

## The Idea Behind LogoRank

Reading materials that align with one's language proficiency is crucial in mastering a new language. However, finding texts that match a learner's level can be challenging. LogoRank aims to simplify this process by developing a tool that predicts the difficulty of French texts for English speakers, facilitating a more tailored learning experience.

### The Potential Impact

For learners, being presented with too-difficult texts can hinder learning, while too-easy texts may not offer enough challenges to facilitate growth. LogoRank addresses this by providing text recommendations that are just right for the user's current level. This approach enhances learning efficiency and boosts learner confidence and engagement.

### Our Goals

Our long-term goal is to integrate LogoRank into the daily learning routine of language learners worldwide, making it an indispensable tool in their language learning journey. Before testing our App, here is how we created an algorithm just for you!

## Application Development Process

Our journey in developing the application involved a meticulous process of testing various models, integrating sophisticated language models, and implementing data augmentation techniques to enhance performance and accuracy. Here's an overview of our development process:

### Initial Model Testing
We commenced our application development by evaluating a suite of fundamental machine learning models, including logistic regression, neural networks, and k-nearest neighbors (KNN). This initial phase aimed to establish a baseline understanding of model performance and to identify the strengths and limitations of each approach.

During this phase, we discovered that while these "basic" models provided reasonable performance on certain tasks, they often struggled to capture the nuanced complexities inherent in natural language processing tasks. Their performance was limited by factors such as feature engineering requirements, model capacity, and scalability issues, prompting us to explore more advanced methodologies.

### Integration of Large Language Models
Recognizing the transformative potential of large language models, we transitioned to leveraging state-of-the-art architectures such as RoBERTa, Open AI etc. leveraging the powerful capabilities offered by these models to process and understand natural language data.

By integrating BERT and CamenBERT into our application pipeline, we observed substantial improvements in performance across various metrics, including accuracy, precision, and recall. The model demonstrated a remarkable ability to understand context, discern subtle nuances, and generate accurate predictions, thereby elevating the overall efficacy of our application.

### Data Augmentation with ChatGPT
In addition to leveraging advanced models, we employed innovative data augmentation techniques to enhance the diversity and robustness of our training data. Leveraging the conversational abilities of ChatGPT, we generated synthetic data instances to supplement our existing dataset. We added 50 lignes to each difficulty from A1 to C2 expending the dataset from 4'800 to 5'100 lines. 

This approach enabled us to create a more comprehensive and varied training dataset, encompassing a wider range of linguistic patterns, styles, and contexts. By augmenting our data in this manner, we mitigated the risk of overfitting, enhanced model generalization, and improved overall performance on unseen data.

### Conclusion
In summary, our application development process encompassed a comprehensive journey of exploration, experimentation, and innovation. By iteratively testing different models, integrating cutting-edge language technologies, and embracing novel data augmentation strategies, we were able to create a robust and effective application capable of delivering accurate and reliable predictions in real-world scenarios. Our commitment to continuous improvement and adaptation ensured that our application remained at the forefront of advancements in machine learning and natural language processing, poised to address evolving challenges and opportunities in the digital landscape.

### Reporting Table Initial Models

| Metric     | Logistic Regression | KNN    | Decision Tree | Random Forest | Neural Network | Neural Network (CNN) | XGBoost      | 
|------------|---------------------|--------|---------------|---------------|----------------|----------------------|--------------|
| Precision  | 0.4374              | 0.4003 | 0.2932        | 0.3706        | 0.4391         | 0.4735               | 0.4021       | 
| Recall     | 0.4427              | 0.3552 | 0.2938        | 0.3729        | 0.4375         | 0.4646               | 0.4094       |
| F1-score   | 0.4347              | 0.3303 | 0.2767        | 0.3649        | 0.4375         | 0.4614               | 0.4000       | 
| Accuracy   | 0.4427              | 0.3552 | 0.2938        | 0.3729        | 0.4375         | 0.4646               | 0.4094       |

### Reporting Table Large Language Models

| Metric     | RoBERTa      | OpenAI Embeddings  |
|------------|--------------|--------------------|
| Precision  | 0.5054       | 0.4706             |
| Recall     | 0.4750       | 0.4813             |
| F1-score   | 0.4729       | 0.4703             |
| Accuracy   | 0.4750       | 0.4813             |

BERT
CamenBERT
CamenBERT augmented 


## The Most accurate Model

The model with the highest accuracy is augmented CamenBERT ... 

The confusion matrix shows ...

Here is a snipp of the code ...

The model behaves by ...

### Erronerous Predictions

During the development of our machine learning model, we encountered several significant challenges. 

**1. Service Overload and Long Running Time**
A critical issue was the long running time on Google Colab, exacerbated by our dependency on external APIs for generating text embeddings. Particularly, we faced a 'ServiceUnavailableError' indicating that the OpenAI's server was temporarily overloaded or under maintenance (as shown in the error screenshot). 

The primary issue indicated by the error is that the server hosting the model (in this case, OpenAI's server for embeddings) was temporarily unavailable or overloaded. This can happen during periods of high demand or server maintenance. It reflected the dependency on third-party services and the need for robust error handling and retry mechanisms in our code.

Moreover, our large languge machine learning models, especially those involving extended datasets and complex computations, had long running times. This can be exacerbated when dependent on external services where network latency and server response times add to the overall execution time.

To address such challenges in future implementations, we had to start using Colab Pro and execute the code with better performing GPUs.

<div align="center">
  <div style="position: relative; text-align: right;">
  <img src="Overload error.png" alt="LogoRank" style="width: 700px; height: auto; position: absolute; top: 10px; right: 10px;">
</div>
</div>

Secondly, Overfitting of data ...

Thirdly, Non representative augmented dataset ...

Finally, pretrained models that were not accurate ...


### Features of the LogoRank Application 


link to the video 

