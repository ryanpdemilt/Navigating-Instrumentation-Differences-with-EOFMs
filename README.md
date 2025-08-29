# Navigating-Instrumentation-Differences-with-EOFMs
Repository for replicating experiments for the paper submitted to Neurips 2025 Workshop ML 4 Physical Sciences

## Abstract

Earth Observation Foundation Models (EOFMs) have exploded in prevalence as tools for processing the massive volumes of remotely sensed and other earth observation data, and for delivering impact on the many essential earth monitoring tasks. An emerging trend posits using the outputs of pre-trained models as 'embeddings' which summarize high dimensional data to be used for generic tasks such as similarity search and content-specific queries. However, most EOFM models are trained only on single modalities of data and then applied or benchmarked by matching bands across different modalities. It is not clear from existing work what impact diverse sensor architectures have on the internal representations of the present suite of EOFMs. We show in this work that the representation space of EOFMs is highly sensitive to sensor architecture and that understanding this difference gives a vital perspective on the pitfalls of current EOFM design and signals for how to move forward as model developers, users, and a community guided by robust remote-sensing science.

## Data Download

Data is available in archived format at: [Drive Link](https://drive.google.com/file/d/1rtFTd26mRLLokgR9dbV429SXEPWJoaHh/view?usp=sharing)

## Model Weights

Model weights will be automatically downloaded

## Acknowledgements

Thanks to the [PANGAEA](https://github.com/VMarsocci/pangaea-bench) for the Encoder and Dataset code and for the evaluation setup used in this work.