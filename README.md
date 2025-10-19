
## Racommendation System, Agnetic AI and ML in Production       

This portfolio summarizes my exploration on Racommendation System (RS) with Agnetic AI and ML in Production practice.  

### Racommendation System  

推荐系统中最重要的问题因具体应用场景而异，但综合通用性、影响力和技术挑战性，以下三个问题通常被认为是Top 3：

1. 冷启动问题（Cold Start Problem）
重要性：新用户或新项目缺乏历史数据，导致推荐系统难以生成准确推荐。这是许多场景（如新用户注册、平台新增内容）的核心挑战。
原因：冷启动直接影响用户体验和平台留存率，解决不好会导致用户流失或内容曝光不足。
[RS_coldstart_graphRAG_LLM](https://github.com/arcadianlyric/RS_coldstart_graphRAG_LLM)
video -> goods  
The global expansion of e-commerce and the rise of new brands mean that vertical market segments—such as those served by live-stream TikTok shops like POPMART—face increased challenges with the "cold start" problem.  

This project implements a multimodal Graph-based Retrieval-Augmented Generation (GraphRAG) system using Deep Lake and MBTI personalized LLM. It is designed to tackle the cold-start problem in e-commerce by providing intelligent product recommendations based on a user's video viewing history. The system mimics a TikTok Shop scenario where user interests, inferred from watched videos, are used to recommend relevant products.
***Keywords***
Cold-Start, LLM, Multimodal, GraphRAG, MBTI Personalization, Deep Lake, Langchain  


2. 预测点击率（CTR Prediction）
重要性：CTR预测是推荐系统的核心任务，直接决定推荐内容的点击概率和平台收益（如广告、电商）。
原因：高准确度的CTR预测能提升用户满意度和商业价值，是推荐系统性能的关键指标。
[RS_movie](https://github.com/arcadianlyric/RS_movies)  
video -> video recommandataion  
A comprehensive end-to-end Three-Layer Architecture recommendation system, featuring multiple classsic recommendation algorithms and A/B testing capabilities.  
***Keywords***
TensorFlow, PySpark, Feature enginnering, embedding, DeepFM, NeuralCF, CTR Prediction, A/B test, Hybrid Recommendation   

3. 多样性与探索性（Diversity and Exploration）
重要性：过度优化短期点击可能导致推荐内容单一，降低用户长期兴趣。通过多样性和探索性推荐，可以发现用户潜在兴趣并提升长期参与度。
原因：平衡探索-利用（Exploration-Exploitation）是推荐系统可持续发展的关键，尤其在动态环境中（如新闻、短视频）。
[RS_movie](https://github.com/arcadianlyric/RS_movies)  
***Keywords***
Exploration-Exploitation, Multi-Armed Bandit

there are various ways to achive recommandation, sometime simple and computationaly quick means (for example, if user views a creator, recommand 3 more posts from this creator) may outperform sophisticated algorithm, so it take deep insight of the user behavior/product to achive good recommandation system.  

based on latest advancment in recosys2025, KDD2025, etc., there 2 major trends: generative, multi-modal based recommendation. combined with other (user bahavior modeling etc.) More users will have interactive/generative requests to platform, so AI recsys assistant is needed. 
generative, LLM, user needs reflected as language; long tail and cold start  
multi-modal, social media posts/comments/view, shopping review, UGC/non-UGC view etc.

## Agentic AI   
integration of agentic AI in 现有的pipeline, 实现垂直领域数字升级, for example in Biotech    
[Agentic Variants Curator](https://github.com/arcadianlyric/PhasedVariants_AgenticCurator)
an agentic workflow enhanced with a RAG-Enhanced LLM Agent, aimed at addressing the challenges of variant curation (detailed gene functions and variant impacts within the phased VCF):  requires significant manual interpretation by skilled variant curators to extract biological and clinical meaning, time-consuming, limits throughput, and can vary between curators .  
***Keywords***
Agentic, LLM, RAG/Langchain, FAISS, Haplotype Phasing, Gene/Variant Curation, Knowledge Graph, Literature Retrieval/Augmented 

### ML in Production & system design  
Cloud native, scalable ML pipeline  
[MLOps_taxi](https://github.com/arcadianlyric/MLops_taxi)
end to end solution(from raw data, EDA, data cleaning, product metrics design, algorithm design, evaluation, report and decision making) of MLOps platform built to predict taxi tips, integrating Kubeflow, Feast, KFServing, monitoring, and stream processing capabilities.
***Keywords***

### Data analysis and EDA     
[Google analytics certificate](https://github.com/arcadianlyric/Google_data_analytics_Bike_share_growth_hacking)


### Product Insights       
In the new era of AI, the future of applied software/ML development is "One-Person Teams". AI can largely handle coding, the 'how' part of a project, allowing engineers to focus on 'why' and 'what'. to become a 'T' shaped talent, except for digging deep in algorithm and system design, I am also trying to embrace:
- **Personal Branding**: Building influence through tech blogs, journal clubs, or hackathons.
- **Business and Product Insight**: Cutting through complexity to address core needs and drive innovation, with some PM interviews now requiring rapid prototyping via “vibe coding” (informal, rapid coding for prototypes).
- **Aesthetic Taste**: AI can generate hundreds of UI designs, but human judgment is needed to select the most promising for A/B testing.
- **Experience**: AI can compile data (e.g., listing stock price drivers), but humans must weigh these factors and make decisions based on years of experience.      

### Competetion and Hackthon  
1. Kaggle  
[Real-Time Market Data Forecasting](https://github.com/arcadianlyric/kaggle_js)
[Problematic Internet Use](https://github.com/arcadianlyric/kaggle_cmi)
2. Hackathon  
[ml_NGS_prediction_Hackathon](https://github.com/arcadianlyric/ml_NGS_prediction_Hackathon)
A graduate school to predict aging related disease risk.  

### Certificates and Practices     
1. [Full-Stack Nanodegree](https://github.com/arcadianlyric/udacity_fullstack)  
2. [Google analytics certificate](https://github.com/arcadianlyric/Google_data_analytics_Bike_share_growth_hacking)
3. [ML in Production](https://www.coursera.org/account/accomplishments/verify/1ZDF3VKIHSAX)  

### Reference
1. [Hacking Growth](https://books.google.com/books/about/Hacking_Growth.html?id=izG5DAAAQBAJ)  
2. [Netflix TechBlog](https://netflixtechblog.com/)  
3. [Deep Learning Recommandation System](https://books.google.com/books/about/Deep_Learning_Recommender_Systems.html?id=ap_v0AEACAAJ)  
4. [A/B test](https://www.udacity.com/enrollment/ud979)  