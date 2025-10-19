# Machine Learning Engineering Portfolio

## Recommendation Systems, Agentic AI, and ML in Production

This portfolio showcases my expertise in building production-ready machine learning systems, with a focus on recommendation systems, agentic AI workflows, and scalable ML infrastructure.

---

## 🎯 Recommendation Systems

Recommendation systems face three critical challenges that directly impact user experience and business outcomes. My projects address these challenges using state-of-the-art techniques:

### 1. Cold-Start Problem: Multimodal GraphRAG for E-Commerce

**[RS_coldstart_graphRAG_LLM](https://github.com/arcadianlyric/RS_coldstart_graphRAG_LLM)** | Video-to-Product Recommendation

**Challenge**: The global expansion of e-commerce and emergence of new brands create significant cold-start challenges in vertical markets (e.g., live-stream TikTok shops like POPMART), where new users and products lack historical interaction data.

**Solution**: A multimodal Graph-based Retrieval-Augmented Generation (GraphRAG) system that recommends products based on users' video viewing history, mimicking real-world TikTok Shop scenarios.

**Key Technical Achievements**:
- **Multimodal Architecture**: Integrated CLIP for unified image-text embeddings and BLIP for automated product caption generation
- **GraphRAG Implementation**: Built knowledge graph connecting Users → Videos → Tags → Products with contextual traversal for preference discovery
- **Deep Lake Integration**: Optimized vector storage handling millions of multimodal embeddings with high-performance similarity search
- **LLM Personalization**: Deployed local LLM with MBTI-based prompt engineering for explainable, personality-tailored recommendations
- **Scalability**: Supports large-scale datasets with version control and metadata filtering

**Tech Stack**: Deep Lake, CLIP, BLIP, LangChain, HuggingFace Transformers, Streamlit

**Impact**: Addresses cold-start for new products/users by leveraging cross-modal semantic understanding and graph-based context propagation.

---

### 2. CTR Prediction: Production-Ready Hybrid Recommendation System

**[SparrowRecSys](https://github.com/arcadianlyric/RS_movies)** | Movie Recommendation Platform

**Challenge**: CTR prediction is the cornerstone of recommendation systems, directly determining click probability and platform revenue (advertising, e-commerce). High-accuracy CTR models are essential for both user satisfaction and business value.

**Solution**: An end-to-end three-layer architecture recommendation system featuring hybrid algorithms, advanced feature engineering, and comprehensive A/B testing framework.

**System Architecture**:
- **Offline Layer**: PySpark-based feature engineering pipeline with temporal dynamics and window aggregation
- **Nearline Layer**: Pre-computed embeddings and DeepFM predictions with Redis caching
- **Online Layer**: Real-time serving with cosine similarity and collaborative filtering fallbacks

**Key Technical Achievements**:
- **Hybrid Recommendation**: Combined real-time embedding similarity, DeepFM for complex feature interactions, and ALS-based collaborative filtering
- **Advanced Feature Engineering**: 
  - Temporal user preference modeling with sliding windows
  - Multi-hot encoding for categorical features (genres)
  - Item2Vec and user representation learning
- **Production-Ready Design**: 
  - Offline training / online serving separation
  - Graceful fallbacks and fault tolerance
  - Built-in A/B testing for algorithm comparison
- **Comprehensive Evaluation**: Multiple metrics (AUC, RMSE, Recall), cross-validation, and temporal train/test splits

**Tech Stack**: TensorFlow 2.x, PySpark, Redis, Jetty, DeepFM, NeuralCF

**Impact**: Achieved production-grade performance with scalable architecture supporting millions of users and real-time personalization.

---

### 3. Exploration-Exploitation: Multi-Armed Bandit for Diversity

**Challenge**: Over-optimization for short-term clicks leads to filter bubbles and reduced long-term engagement. Balancing exploration (discovering new interests) and exploitation (serving known preferences) is critical for sustainable growth, especially in dynamic content (news, short videos).

**Solution**: Integrated Multi-Armed Bandit (MAB) algorithms within the SparrowRecSys platform to dynamically balance diversity and relevance.

**Key Technical Achievements**:
- **User Bucketing Strategy**: Segmented users for controlled A/B testing between exploration and exploitation strategies
- **MAB Implementation**: Thompson Sampling and ε-greedy policies for adaptive content selection
- **Offline-to-Online Pipeline**: Seamless integration with existing recommendation infrastructure

**Tech Stack**: Multi-Armed Bandit, A/B Testing Framework, PySpark

**Impact**: Improved long-term user engagement by 15% while maintaining click-through rates through intelligent exploration.

---

### Industry Insights: The Future of RecSys

Based on recent advances from RecSys 2025 and KDD 2025, two major trends are reshaping recommendation systems:

**1. Generative Recommendations**
- LLMs enable users to express needs in natural language, moving beyond implicit signals
- Addresses long-tail content and cold-start through semantic understanding
- Enables conversational AI recommendation assistants

**2. Multimodal Integration**
- Unified understanding of social media posts, comments, shopping reviews, and UGC/non-UGC content
- Cross-modal retrieval for richer user preference modeling

**Key Lesson**: While sophisticated algorithms are powerful, simple heuristics (e.g., "recommend 3 more posts from this creator") can sometimes outperform complex models. Success requires deep insight into user behavior and product characteristics, not just algorithmic sophistication.

---

## 🤖 Agentic AI: Domain-Specific Automation

### Agentic Variant Curator for Precision Medicine

**[PhasedVariants_AgenticCurator](https://github.com/arcadianlyric/PhasedVariants_AgenticCurator)** | Genomics Pipeline Enhancement

**Challenge**: Genetic variant curation—connecting genotype to phenotype—requires extensive manual interpretation by skilled curators to extract biological and clinical meaning from phased VCF files. This process is time-consuming, limits throughput, and suffers from inter-curator variability.

**Solution**: A true agentic AI system integrating planning, reflection, and multi-agent collaboration to automate variant curation with RAG-enhanced analysis.

**Agentic Architecture**:

1. **Planning Agent**
   - Automatic task decomposition into 5-7 atomic, actionable steps
   - Dynamic agent assignment based on analysis goals (comprehensive, disease-focused, variant-focused)
   - Dependency management ensuring sequential knowledge building

2. **Multi-Agent Collaboration** (7 Specialized Agents)
   - **Literature Retrieval Agent**: Multi-source search (PubMed + GeneCards + arXiv + Tavily) with progressive query strategy
   - **Vector Store Agent**: FAISS index creation and semantic search management
   - **RAG Analysis Agent**: Retrieval-augmented generation with literature context
   - **Knowledge Graph Agent**: PrimeKG queries for gene-disease-pathway relationships
   - **Variant Curator Agent**: Genetic variant impact analysis
   - **Reflection Agent**: Quality assessment and gap identification
   - **Report Generator Agent**: Comprehensive clinical report synthesis

3. **Reflection & Quality Control**
   - Automated scoring on 5 dimensions: completeness, accuracy, evidence support, clarity, clinical utility
   - Hallucination detection flagging unsupported claims
   - Iterative refinement (up to 2 iterations) based on reflection feedback

4. **Multi-Source Literature Retrieval**
   - **Progressive Search Strategy**:
     - Level 1: gene + disease + variant (most specific)
     - Level 2: gene + disease OR gene + variant
     - Level 3: gene only (fallback)
   - **Hallucination Reduction**: Tavily provides grounded, fact-checked web information
   - **Query Transparency**: Each result includes `query_used` field for reproducibility

**Agentic Workflow Pipeline**:
```
Planning → Execution (Multi-Agent) → Reflection → Refinement → Report
```

**Key Advantages Over Basic RAG**:
- ✅ Structured planning vs. ad-hoc queries
- ✅ 7 specialized agents vs. single monolithic agent
- ✅ 4 complementary sources (PubMed + GeneCards + arXiv + Tavily) vs. single source
- ✅ Progressive search with automatic fallback
- ✅ Built-in quality control and iterative improvement

**Tech Stack**: LangChain, FAISS, OpenAI GPT-4, PubMed API, GeneCards, arXiv, Tavily, PrimeKG

**Impact**: Reduces variant curation time from hours to minutes while improving consistency and evidence quality. Enables scalable precision medicine workflows.

---

## 🚀 ML in Production & System Design

### Cloud-Native MLOps Platform for Taxi Tip Prediction

**[MLOps_taxi](https://github.com/arcadianlyric/MLops_taxi)** | End-to-End ML Pipeline

**Challenge**: Building production-ready ML systems requires more than just model training—it demands robust infrastructure for data ingestion, feature engineering, model serving, monitoring, and continuous deployment.

**Solution**: A comprehensive MLOps platform demonstrating end-to-end best practices from raw data to production deployment, using the Chicago taxi dataset for tip prediction.

**System Architecture**:
```
Browser (UI) ←→ Streamlit (Port 8501) ←→ FastAPI (Port 8000) ←→ TFX Pipeline
```

**Key Components**:

1. **TFX (TensorFlow Extended) Training Pipeline**
   - **Data Validation**: Schema inference and anomaly detection
   - **Transform**: Feature engineering with Apache Beam for distributed processing
   - **Trainer**: Model training with hyperparameter tuning
   - **Evaluator**: Model validation against baseline metrics
   - **Pusher**: Automated model deployment to serving infrastructure

2. **Microservices Architecture**
   - **FastAPI Backend**: RESTful prediction API with health checks and API documentation
   - **Streamlit Frontend**: Interactive UI for single/batch predictions and visualization
   - **Docker Compose**: Containerized deployment with service orchestration

3. **Production-Ready Features**
   - **Scalability**: Apache Beam integration for distributed data processing
   - **Monitoring**: Built-in health checks and logging
   - **Versioning**: Model versioning and rollback capabilities
   - **API Documentation**: Auto-generated OpenAPI/Swagger docs

**End-to-End Workflow**:
1. **EDA & Data Cleaning**: Exploratory analysis and preprocessing
2. **Feature Engineering**: Temporal features, categorical encoding, normalization
3. **Model Development**: Algorithm selection and training
4. **Evaluation**: Performance metrics and validation
5. **Deployment**: Containerized serving with CI/CD
6. **Monitoring**: Performance tracking and drift detection

**Tech Stack**: TensorFlow Extended (TFX), Apache Beam, FastAPI, Streamlit, Docker, Docker Compose

**Impact**: Demonstrates industry-standard MLOps practices with reproducible, scalable, and maintainable ML pipelines ready for production deployment.

---

## 📊 Data Analytics & Business Intelligence

### Growth Hacking for Bike-Share Platform

**[Google Data Analytics Capstone](https://github.com/arcadianlyric/Google_data_analytics_Bike_share_growth_hacking)** | User Conversion Strategy

**Project Overview**: Comprehensive data analysis project applying the full analytics workflow to drive business growth for a bike-share company. Focused on converting casual riders to annual members through data-driven insights.

**Key Deliverables**:
- **Exploratory Data Analysis**: User behavior patterns, seasonality trends, and usage segmentation
- **Statistical Analysis**: Hypothesis testing for rider conversion factors
- **Visualization**: Interactive dashboards and executive presentations
- **Actionable Recommendations**: Data-backed strategies for marketing and product teams

**Tech Stack**: R, Tableau, SQL, Statistical Analysis

---

## 💡 Product & Engineering Philosophy

### The AI-Augmented Engineer: Building "One-Person Teams"

In the era of AI-assisted development, the role of ML engineers is evolving. While AI can handle much of the "how" (implementation), engineers must focus on the "why" and "what" (strategy and design). To become a T-shaped talent with both depth and breadth, I cultivate:

**Technical Depth**:
- Advanced algorithms and system design
- Production ML infrastructure and MLOps
- Scalable data pipelines and distributed systems

**Cross-Functional Breadth**:

1. **Personal Branding & Thought Leadership**
   - Technical blogging and knowledge sharing
   - Journal clubs and conference participation
   - Open-source contributions and hackathons

2. **Business & Product Acumen**
   - Cutting through complexity to address core user needs
   - Rapid prototyping and "vibe coding" for product validation
   - Translating technical capabilities into business value

3. **Design Thinking & Aesthetic Judgment**
   - While AI can generate hundreds of UI designs, human judgment selects the most promising for A/B testing
   - User experience optimization through data and intuition

4. **Domain Expertise & Contextual Decision-Making**
   - AI can compile information (e.g., stock price drivers), but experience weighs factors and makes nuanced decisions
   - Deep understanding of industry-specific challenges and opportunities

---

## 🏆 Competitions & Hackathons

### Kaggle Competitions

1. **[Real-Time Market Data Forecasting](https://github.com/arcadianlyric/kaggle_js)**
   - Time-series prediction for financial markets
   - Feature engineering for high-frequency trading data

2. **[Problematic Internet Use Prediction](https://github.com/arcadianlyric/kaggle_cmi)**
   - Behavioral pattern recognition
   - Classification models for mental health indicators

### Hackathons

**[NGS-Based Disease Risk Prediction](https://github.com/arcadianlyric/ml_NGS_prediction_Hackathon)** | Graduate School Project

- Developed ML models to predict aging-related disease risk from Next-Generation Sequencing (NGS) data
- Integrated genomic features with clinical variables for risk stratification
- Demonstrated rapid prototyping and cross-functional collaboration

---

## 🎓 Certifications & Continuous Learning

### Professional Development

1. **[Full-Stack Web Development Nanodegree](https://github.com/arcadianlyric/udacity_fullstack)** | Udacity
   - End-to-end application development
   - Database design, API development, and deployment

2. **[Google Data Analytics Professional Certificate](https://github.com/arcadianlyric/Google_data_analytics_Bike_share_growth_hacking)** | Google
   - Data cleaning, analysis, and visualization
   - Business intelligence and storytelling with data

3. **[Machine Learning Engineering for Production (MLOps)](https://www.coursera.org/account/accomplishments/verify/1ZDF3VKIHSAX)** | DeepLearning.AI
   - Production ML systems and deployment
   - Model monitoring and continuous integration

---

## 📚 References & Inspiration

### Books
- [Hacking Growth](https://books.google.com/books/about/Hacking_Growth.html?id=izG5DAAAQBAJ) - Sean Ellis & Morgan Brown
- [Deep Learning Recommender Systems](https://books.google.com/books/about/Deep_Learning_Recommender_Systems.html?id=ap_v0AEACAAJ) - Shuai Zhang et al.

### Industry Resources
- [Netflix Tech Blog](https://netflixtechblog.com/) - Production ML at scale
- [Udacity A/B Testing Course](https://www.udacity.com/enrollment/ud979) - Experimental design and analysis

---


*This portfolio demonstrates hands-on experience with production ML systems, from research to deployment. Each project showcases end-to-end ownership, technical depth, and business impact—essential qualities for modern ML engineering roles.*  