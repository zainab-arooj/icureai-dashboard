PROTOTYPE 
            iCUREAI: Predictive CRISPR & Clinical Acceleration

iCUREAI is a unified, multi-modal simulation environment designed to shift gene therapy validation from the benchtop to the desktop in 60 seconds. By making CRISPR outcomes highly predictive, this platform aims to de-risk complex gene therapies (such as treatments for Sickle Cell Disease) before dedicating immense time and resources to clinical trials.

Key Architectural Modules
1. Unified Genomic & Cellular Validation: Integrates a patient’s variant data (VCF) with a dynamic, interactive 3D simulation of HBB protein structural changes.
2. Lineage Tracking via scRNA-seq: Offers real-time UMAP visualization tracking hematopoietic stem cell (HSC) differentiation trajectories into Erythroid Progenitors, allowing you to map out cellular effects before editing.
3. Explainable AI (XAI) for Risk Metrics: Quantifies mission-critical metrics like "gRNA Off-Target Risk" and "Mutation Severity" using transparent SHAP feature contributions, bringing regulatory-grade confidence to the prediction pipeline.
4. Global Equity Ethics Submodule: Simulates deployment strategies and economic healthcare hurdles across different regions based on global GDP per capita to meet strategic distribution goals.

Project Structure & Execution

The core implementation is contained entirely within a streamlined web dashboard interface:

1. `app.py`: The main application script housing the interactive UI, visualization layouts, and simulation logic.
2. `requirements.txt`: The complete list of python dependencies required to execute the pipeline locally.

How to Run Locally

Clone this repository:
   ```bash
   git clone [https://github.com/zainab-arooj/icureai-dashboard.git](https://github.com/zainab-arooj/icureai-dashboard.git)
   cd icureai-dashboard

1. Install the necessary dependencies
Bash
pip install -r requirements.txt

2. Spin up the application interface:
streamlit run app.py
(Note: Adjust the runner command if using a different framework like Dash or Gradio instead of Streamlit).
```

iCUREAI Platform Demonstration

https://github.com/user-attachments/assets/652dc890-c130-4ec0-8131-3019a856df55



